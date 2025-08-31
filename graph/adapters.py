from __future__ import annotations

from typing import Any, Dict, List
from dataclasses import dataclass
from graph.retrieval import RetrieveTool
from opik import track, opik_context
from utils.logger import logger

from graph.models import (
    MessagesState,
    AIMessage,
    ToolMessage,
)
from graph.schemas import (
    RoutingDecision,
    GenerationResponse,
)
from graph.utils import (
    render_system_prompt_,
    build_conversation_history,
)


@dataclass
class PromptBundle:
    messages: List[Dict[str, Any]]
    should_stream: bool = False


class RouterAdapter:
    """Router adapter: decides which strategy to use to respond to the query.
    - Non-streaming
    - Returns RoutingDecision via engine's non-streaming LLM call
    """

    name: str = "router"
    response_model = RoutingDecision

    async def build_prompt(self, state: MessagesState) -> PromptBundle:
        system_prompt = render_system_prompt_(
            user_query=state.user_query,
            name=self.name,
            decision=next(
                (
                    msg.content
                    for msg in reversed(state.messages)
                    if isinstance(msg, AIMessage)
                ),
                "default",
            ),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            *build_conversation_history(
                state, max_history=24, include_tool_content=True
            ),
        ]

        logger.debug("Built router prompt", extra={"system_prompt": system_prompt})
        return PromptBundle(messages=messages, should_stream=False)

    @track(capture_output=False, capture_input=False)
    async def postprocess(
        self, state: MessagesState, validated_response: RoutingDecision
    ) -> MessagesState:
        logger.info(
            "Routing decision",
            extra={
                "thread_id": state.thread_id,
                "decision": validated_response.decision,
                "query_for_retrieval": validated_response.query_for_retrieval,
            },
        )

        state.messages.append(AIMessage(content=validated_response.decision))
        # Opik for logging span under trace.
        opik_context.update_current_span(
            name=self.name,
            input={"user_query": state.user_query},
            output={"response": validated_response.decision},
        )

        return state


class GenerateSimpleResponseAdapter:
    """Generator adapter: streams the final answer in a single node.
    - Streaming
    - Engine renders system prompt via render_generate_answer() based on context_mode
    - Engine is responsible for mapping stream deltas to StreamingResponse
    """

    name: str = "generate_simple_response"
    response_model = GenerationResponse

    async def build_prompt(self, state: MessagesState) -> PromptBundle:
        system_prompt = render_system_prompt_(
            user_query=state.user_query,
            name=self.name,
            decision=next(
                (
                    msg.content
                    for msg in reversed(state.messages)
                    if isinstance(msg, AIMessage)
                ),
                "default",
            ),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            *build_conversation_history(
                state, max_history=24, include_tool_content=False
            ),
        ]
        logger.debug(
            "Built generate_simple_response prompt (streaming)",
            extra={"system_prompt": system_prompt},
        )

        return PromptBundle(messages=messages, should_stream=True)

    @track(capture_output=False, capture_input=False)
    async def postprocess(
        self, state: MessagesState, validated_response: GenerationResponse
    ) -> MessagesState:
        logger.info(
            "Generated final simple response (postprocess)",
            extra={"thread_id": state.thread_id, "text_": validated_response.text},
        )
        # Opik for logging span under trace.
        opik_context.update_current_span(
            name=self.name,
            input={"user_query": state.user_query},
            output={"response": validated_response.text},
        )
        state.messages.append(AIMessage(content=validated_response.text))
        await state.update_thread_store()

        return state


class GenerateAnswerWithHistoryAdapter:
    """Generator adapter: streams the final answer in a single node.
    - Streaming
    - Engine renders system prompt via render_generate_answer() based on context_mode
    - Engine is responsible for mapping stream deltas to StreamingResponse
    """

    name: str = "generate_answer_with_history"
    response_model = GenerationResponse

    async def build_prompt(self, state: MessagesState) -> PromptBundle:
        system_prompt = render_system_prompt_(
            user_query=state.user_query, name=self.name, decision="answer_with_history"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            *build_conversation_history(
                state, max_history=24, include_tool_content=False
            ),
        ]
        logger.debug(
            "Built generate_answer_with_history prompt (streaming)",
            extra={"system_prompt": system_prompt},
        )

        return PromptBundle(messages=messages, should_stream=True)

    @track(capture_output=False, capture_input=False)
    async def postprocess(
        self, state: MessagesState, validated_response: GenerationResponse
    ) -> MessagesState:
        logger.info(
            "Generated final answer with history (postprocess)",
            extra={"thread_id": state.thread_id, "text_": validated_response.text},
        )
        # Opik for logging span under trace.
        opik_context.update_current_span(
            name=self.name,
            input={"user_query": state.user_query},
            output={"response": validated_response.text},
        )
        state.messages.append(AIMessage(content=validated_response.text))
        await state.update_thread_store()

        return state


class GenerateAnswerWithRagAdapter:
    """Generator adapter: streams the final answer in a single node.
    - Streaming
    - Engine renders system prompt via render_generate_answer() based on context_mode
    - Engine is responsible for mapping stream deltas to StreamingResponse
    """

    name: str = "generate_answer_with_rag"
    response_model = GenerationResponse

    async def build_prompt(self, state: MessagesState) -> PromptBundle:
        system_prompt = render_system_prompt_(
            user_query=state.user_query, name=self.name, decision="retrieve_and_answer"
        )

        query = state.meta.get(
            "refined_query",
            state.user_query,
        )
        retriever = RetrieveTool()
        results = await retriever.execute(query)
        state.meta["retrieved_docs"] = [r.to_dict() for r in results]

        state.messages.append(
            ToolMessage(
                content="",
                tool_name="retrieve",
                input={"query": state.user_query},
                output=state.meta.get("retrieved_docs", []),
            )
        )

        messages = [
            {"role": "system", "content": system_prompt},
            *build_conversation_history(
                state, max_history=24, include_tool_content=True
            ),
        ]
        logger.debug(
            "Built generate_answer_with_rag prompt (streaming)",
            extra={"system_prompt": system_prompt},
        )

        return PromptBundle(messages=messages, should_stream=True)

    @track(capture_output=False, capture_input=False)
    async def postprocess(
        self, state: MessagesState, validated_response: GenerationResponse
    ) -> MessagesState:
        logger.info(
            "Generated final answer with rag (postprocess)",
            extra={"thread_id": state.thread_id, "text_": validated_response.text},
        )
        # Opik for logging span under trace.
        opik_context.update_current_span(
            name=self.name,
            input={"user_query": state.user_query},
            output={"response": validated_response.text},
            metadata={"retrieved_docs": state.meta.get("retrieved_docs", [])},
        )
        state.messages.append(AIMessage(content=validated_response.text))
        await state.update_thread_store()

        return state
