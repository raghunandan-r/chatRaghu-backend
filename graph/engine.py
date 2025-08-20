from typing import Optional, Any, Dict, List, AsyncGenerator, Tuple
from datetime import datetime, timezone
from opik import track, opik_context
from graph.config import GraphConfig
from utils.retry import async_retry_with_backoff
from utils.logger import logger
from evaluation_models import ConversationFlow, EnrichedNodeExecutionLog
from graph.models import MessagesState, StreamingResponse
from graph.schemas import GenerationResponse
from graph.retrieval import RetrieveTool


class GraphEngine:
    def __init__(
        self,
        nodes: dict,
        edges: dict,
        entry_point: str,
        instructor_client,
        queue_manager=None,
        config: GraphConfig | None = None,
    ):
        self.nodes = nodes
        self.edges = edges
        self.entry_point = entry_point
        self.client = instructor_client
        self.queue_manager = queue_manager
        self.config = config or GraphConfig()


    def _build_enriched_log(
        self,
        *,
        node_name: str,
        start_time: datetime,
        end_time: datetime,
        system_prompt: Optional[str] = None,
        input: Dict[str, Any],
        output: Dict[str, Any],
        retrieved_docs: Optional[List[Dict[str, Any]]] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        return EnrichedNodeExecutionLog(
            node_name=node_name,
            start_time=start_time,
            end_time=end_time,
            input=input,
            output=output,
            retrieved_docs=retrieved_docs or [],
            system_prompt=system_prompt,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def _emit_final_audit_and_enqueue(self, *, conversation_flow):

        # Compute totals, attach usage, and enqueue to evaluation via queue_manager
        if self.queue_manager:
            await self.queue_manager.enqueue_response(conversation_flow)

    # TODO: when we branch to multiple LLM providers, or clarifications, mid-stream decisions and tool calling in streaming 
    async def _consume_stream(self, stream):    
        pass
    
    @track(capture_output=False, capture_input=False)
    async def execute_stream(
        self,
        initial_state,  # MessagesState
        run_id: str,
        turn_index: int,
    ) -> AsyncGenerator[Tuple["StreamingResponse", Dict[str, Any]], None]:
        """
        Orchestrate: build → call → postprocess → log → route.
        """

        opik_context.update_current_trace(            
            thread_id=initial_state.thread_id,
            input={"user_query": initial_state.user_query},
        )

        # Initialize conversation flow tracking
        conversation_flow = ConversationFlow(
            run_id=run_id,
            thread_id=initial_state.thread_id,
            turn_index=turn_index,
            user_query=initial_state.user_query,
            graph_version="v1",
            latency_ms=None,
            time_to_first_token_ms=None,
            total_prompt_tokens=0,
            total_completion_tokens=0,
        )


        current_node = self.entry_point
        state = initial_state
        while current_node and current_node != "END":
            adapter = self.nodes[current_node]
            start_time = datetime.now(timezone.utc)

            # 1) build the prompt via adapter
            bundle = adapter.build_prompt(state)
            messages = bundle.messages
            logger.info("DEBUG: messages", extra={"messages": messages})

            # 2) call the llm
            if bundle.should_stream:
                stream = self.client.chat.completions.create_partial(
                    model=self.config.default_model,
                    messages=messages,                    
                    stream=True,
                    temperature=self.config.default_temperature,
                    response_model=GenerationResponse,
                )

                final_text, usage = "", None
                async for partial in stream:  # Async for with awaited stream
                    if hasattr(partial, "text") and partial.text:
                        new = partial.text
                        delta = new[len(final_text):]
                        if delta:
                            yield StreamingResponse(content=delta), {"node":"generate_answer"}
                        final_text = new
                        
                        if not conversation_flow.time_to_first_token_ms:
                            
                            conversation_flow.time_to_first_token_ms = (
                                datetime.now(timezone.utc) - conversation_flow.start_time
                            ).total_seconds() * 1000
                            logger.debug(
                                "First token received for generate_answer",
                                extra={
                                    "node": adapter.name,
                                    "thread_id": initial_state.thread_id,
                                    "ttft_ms": conversation_flow.time_to_first_token_ms,
                                },
                            )

                    if hasattr(partial, "usage") and partial.usage:
                        usage = partial.usage
                
                validated = GenerationResponse(text=final_text)
                completion = type("Usage", (), {"usage": usage})
            
            else:
                response_model = getattr(adapter, "response_model", None)
                assert response_model is not None, f"{adapter.name} must define response_model"
                
                validated, completion = await self.client.chat.completions.create_with_completion(
                    model=self.config.default_model,
                    response_model=response_model,
                    messages=messages,
                    temperature=self.config.default_temperature,
                    max_retries=self.config.llm_retry_count,
                )

            # 6) routing and side-effects (retrieval)

            next_edge = self.edges[current_node]
            decision = getattr(validated, "decision", None)


            # 7) routing driven by graph definition
            next_map = self.edges[current_node]
            next_node = next_map.get(decision, next_map.get("default"))

            # TODO: move this to the adapter, but need to move decision to adapter too..
            # Side-effects tied to (node, decision) before routing
            if adapter.name == "query_or_respond" and decision == "RETRIEVE":
                query = state.meta.get(
                    "refined_query",
                    state.user_query,
                )
                retriever = RetrieveTool()
                results = await retriever.execute(query)
                state.meta["retrieved_docs"] = [r.to_dict() for r in results]
                state.meta["context_mode"] = "rag"
            elif adapter.name == "query_or_respond" and decision == "SUFFICIENT":
                state.meta["context_mode"] = "history"
            elif adapter.name == "relevance_check" and decision == "IRRELEVANT":
                state.meta["context_mode"] = "deflection"


            # 3) postprocess the mutate state
            state = await adapter.postprocess(state, validated)

            # 4) token accounting and audit log
            
            prompt_tokens = getattr(getattr(completion, "usage", None), "prompt_tokens", None)
            completion_tokens = getattr(getattr(completion, "usage", None), "completion_tokens", None)
            end_time = datetime.now(timezone.utc)
            node_log = self._build_enriched_log(
                node_name=adapter.name,
                start_time=start_time,
                end_time=end_time,
                system_prompt="\n".join([msg.get("content") for msg in messages if msg.get("role") == "system"]),
                input={"conversation_history": [msg for msg in messages if msg.get("role") != "system"]},
                output=validated.model_dump() if hasattr(validated, "model_dump") else {"text": getattr(validated, "text", str(validated))},
                retrieved_docs=state.meta.get("retrieved_docs", []),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            # Aggregate tokens from node logs
            conversation_flow.total_prompt_tokens += prompt_tokens if prompt_tokens is not None else 0
            conversation_flow.total_completion_tokens += completion_tokens if completion_tokens is not None else 0
            conversation_flow.node_executions.append(node_log)
            conversation_flow.latency_ms = (end_time - start_time).total_seconds() * 1000


            if not next_node or next_node == "END":
                break
            current_node = next_node
        
        opik_context.update_current_trace(
                        output={"full_response": validated.text},            
                        )
        conversation_flow.final_response = validated.text
        await self._emit_final_audit_and_enqueue(                                        
                    conversation_flow=conversation_flow,
                    )


