from typing import Optional, Any, Dict, List, AsyncGenerator, Tuple, Union
from datetime import datetime, timezone
from opik import track, opik_context
from graph.config import GraphConfigDefault, GraphConfigImmi
from utils.logger import logger
from .evaluation_models import ConversationFlow, EnrichedNodeExecutionLog
from graph.models import MessagesState, StreamingResponse
from graph.schemas import GenerationResponse
from contextlib import suppress
import asyncio


class AsyncAuditService:
    """Async service for non-blocking audit and logging operations"""

    def __init__(self, queue_manager=None):
        self.queue_manager = queue_manager
        self._background_tasks = set()

    async def _build_enriched_log_async(
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

    async def log_node_execution(
        self,
        *,
        conversation_flow: ConversationFlow,
        node_name: str,
        start_time: datetime,
        messages: List[Dict[str, Any]],
        validated_response: Any,
        state: MessagesState,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ):
        """Async node execution logging"""
        try:
            end_time = datetime.now(timezone.utc)

            # Build log asynchronously
            node_log = await self._build_enriched_log_async(
                node_name=node_name,
                start_time=start_time,
                end_time=end_time,
                system_prompt="\n".join(
                    [
                        msg.get("content")
                        for msg in messages
                        if msg.get("role") == "system"
                    ]
                ),
                input={
                    "conversation_history": [
                        msg for msg in messages if msg.get("role") != "system"
                    ]
                },
                output=validated_response.model_dump()
                if hasattr(validated_response, "model_dump")
                else {
                    "text": getattr(validated_response, "text", str(validated_response))
                },
                retrieved_docs=state.meta.get("retrieved_docs", []),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            # Update conversation flow atomically
            conversation_flow.total_prompt_tokens += (
                prompt_tokens if prompt_tokens is not None else 0
            )
            conversation_flow.total_completion_tokens += (
                completion_tokens if completion_tokens is not None else 0
            )
            conversation_flow.node_executions.append(node_log)
            conversation_flow.latency_ms = (
                end_time - start_time
            ).total_seconds() * 1000

        except Exception as e:
            logger.error(
                "Failed to log node execution",
                extra={"error": str(e), "node": node_name},
            )

    async def finalize_conversation_flow(
        self, conversation_flow: ConversationFlow, final_response: str
    ):
        """Async final conversation flow processing"""
        try:
            conversation_flow.final_response = final_response

            # Emit audit and enqueue if queue manager exists
            if self.queue_manager:
                await self.queue_manager.enqueue_response(conversation_flow)

        except Exception as e:
            logger.error(
                "Failed to finalize conversation flow", extra={"error": str(e)}
            )

    def fire_and_forget_audit(self, coro):
        """Create background task for audit operation"""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        # Suppress exceptions to prevent audit failures from affecting main flow
        def _handle_audit_exception(task):
            with suppress(Exception):
                if task.exception():
                    logger.error(
                        "Audit operation failed", extra={"error": str(task.exception())}
                    )

        task.add_done_callback(_handle_audit_exception)


class GraphEngine:
    def __init__(
        self,
        nodes: dict,
        edges: dict,
        entry_point: str,
        instructor_client,
        queue_manager=None,
        config: Union[GraphConfigDefault, GraphConfigImmi, None] = None,
    ):
        self.nodes = nodes
        self.edges = edges
        self.entry_point = entry_point
        self.client = instructor_client
        self.queue_manager = queue_manager
        self.config = config
        self.audit_service = AsyncAuditService(queue_manager=queue_manager)

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
            graph_version=self.config.graph_version,
            latency_ms=None,
            time_to_first_token_ms=None,
            total_prompt_tokens=0,
            total_completion_tokens=0,
        )

        current_node = self.entry_point
        state = initial_state
        state.meta["graph_type"] = self.config.graph_type

        while current_node and current_node != "END":
            adapter = self.nodes[current_node]
            start_time = datetime.now(timezone.utc)

            # 1) build the prompt via adapter
            bundle = await adapter.build_prompt(state)
            messages = bundle.messages
            logger.info("DEBUG: messages", extra={"messages": messages})

            # 2) call the llm
            if bundle.should_stream:
                stream = self.client.chat.completions.create_partial(
                    model=self.config.default_model,
                    messages=messages,
                    stream=True,
                    temperature=self.config.default_temperature,
                    max_tokens=4096,
                    response_model=GenerationResponse,
                    extra_body={
                        "provider": {
                            "order": ["azure", "openai"],
                            "data_collection": "deny",
                        },
                        "use_context": True,
                    },
                )

                final_text, usage = "", None
                async for partial in stream:  # Async for with awaited stream
                    if hasattr(partial, "text") and partial.text:
                        new = partial.text
                        delta = new[len(final_text) :]
                        if delta:
                            yield StreamingResponse(content=delta), {
                                "node": "generate_answer"
                            }
                        final_text = new

                        if not conversation_flow.time_to_first_token_ms:
                            conversation_flow.time_to_first_token_ms = (
                                datetime.now(timezone.utc)
                                - conversation_flow.start_time
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
                assert (
                    response_model is not None
                ), f"{adapter.name} must define response_model"

                (
                    validated,
                    completion,
                ) = await self.client.chat.completions.create_with_completion(
                    model=self.config.thinking_model,
                    response_model=response_model,
                    messages=messages,
                    temperature=self.config.default_temperature,
                    max_retries=self.config.llm_retry_count,
                    max_tokens=4096,
                    extra_body={
                        "provider": {
                            "order": ["cerebras", "groq"],
                            "data_collection": "deny",
                        },
                        "use_context": True,
                    },
                )

            # 3) routing and side-effects (retrieval)
            decision = getattr(validated, "decision", None)

            next_map = self.edges[current_node]
            next_node = next_map.get(decision, next_map.get("default"))

            # 3) postprocess the mutate state
            state = await adapter.postprocess(state, validated)

            # 4) token accounting and audit log,
            prompt_tokens = getattr(
                getattr(completion, "usage", None), "prompt_tokens", None
            )
            completion_tokens = getattr(
                getattr(completion, "usage", None), "completion_tokens", None
            )

            self.audit_service.fire_and_forget_audit(
                self.audit_service.log_node_execution(
                    conversation_flow=conversation_flow,
                    node_name=adapter.name,
                    start_time=start_time,
                    messages=messages,
                    validated_response=validated,
                    state=state,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )

            if not next_node or next_node == "END":
                break
            current_node = next_node

        # Handle different response types for final trace update
        if hasattr(validated, "text"):
            final_response = validated.text
        elif hasattr(validated, "decision"):
            final_response = validated.decision
        else:
            final_response = str(validated)

        opik_context.update_current_trace(
            output={"full_response": final_response},
        )
        conversation_flow.final_response = final_response
        self.audit_service.fire_and_forget_audit(
            self.audit_service.finalize_conversation_flow(
                conversation_flow, final_response
            )
        )
