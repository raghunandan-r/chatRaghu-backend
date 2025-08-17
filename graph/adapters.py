from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from opik import track,opik_context
from utils.logger import logger

from graph.models import MessagesState, SystemMessage, AIMessage, ToolMessage, HumanMessage
from graph.schemas import (
    RelevanceDecision,
    RoutingDecision,
    DeflectionCategoryDecision,
    GenerationResponse,
)
from graph.utils import (
    render_simple_template,
    render_deflection_categorizer,
    build_conversation_history,
    render_prompt_generate_answer,
)

@dataclass
class PromptBundle:
    messages: List[Dict[str, Any]]
    should_stream: bool = False


class RelevanceCheckAdapter:
    """Gate adapter: classifies a query as RELEVANT or IRRELEVANT.
    - Non-streaming
    - Returns RelevanceDecision via engine's non-streaming LLM call
    """

    name: str = "relevance_check"
    response_model = RelevanceDecision

    def build_prompt(self, state: MessagesState) -> PromptBundle:
        system_prompt = render_simple_template("relevance_check", query=state.user_query)
        messages = [
            {"role": "system", "content": system_prompt},
            *build_conversation_history(state, max_history=24, include_tool_content=True),            
        ]
        
        logger.debug("Built relevance_check prompt", extra={"system_prompt": system_prompt})
        return PromptBundle(messages=messages, should_stream=False)

    @track(capture_output=False, capture_input=False)
    async def postprocess(self, state: MessagesState, validated_response: RelevanceDecision) -> MessagesState:
        logger.info(
            "Relevance decision",
            extra={"thread_id": state.thread_id, "decision": validated_response.decision},
        )
        # Opik for logging span under trace.
        opik_context.update_current_span(
            name=self.name,
            input={"user_query": state.user_query},
            output={"response": validated_response.decision},            
            )
        state.messages.append(AIMessage(content=validated_response.decision))

        return state

    async def augment_metadata(self, state: MessagesState, start_time, end_time) -> Dict[str, Any]:
        return {}


class QueryOrRespondAdapter:
    """Gate adapter: decides whether to retrieve or respond from history.
    - Non-streaming
    - Returns RoutingDecision via engine's non-streaming LLM call
    """

    name: str = "query_or_respond"
    response_model = RoutingDecision

    def build_prompt(self, state: MessagesState) -> PromptBundle:        
        system_prompt = render_simple_template("query_or_respond", query=state.user_query)
        messages = [
            {"role": "system", "content": system_prompt},
            *build_conversation_history(state, max_history=24, include_tool_content=True),            
        ]
        logger.debug("Built query_or_respond prompt", extra={"system_prompt": system_prompt})
        return PromptBundle(messages=messages, should_stream=False)

    @track(capture_output=False, capture_input=False)
    async def postprocess(self, state: MessagesState, validated_response: RoutingDecision) -> MessagesState:
        logger.info(
            "Routing decision",
            extra={
                "thread_id": state.thread_id,
                "decision": validated_response.decision,
                "query_for_retrieval": validated_response.query_for_retrieval,
            },
        )

        # Opik for logging span under trace.
        opik_context.update_current_span(
            name=self.name,
            input={"user_query": state.user_query},
            output={"response": validated_response.decision},
            metadata={"retrieved_docs": state.meta.get("retrieved_docs", [])},
            )
        state.messages.append(AIMessage(content=validated_response.decision))
        state.messages.append(ToolMessage(
            content="",
            tool_name="retrieve",
            input={"query": state.user_query},
            output=state.meta.get("retrieved_docs", []),
        ))
        
        return state

    async def augment_metadata(self, state: MessagesState, start_time, end_time) -> Dict[str, Any]:
        return {}


class DeflectionCategorizerAdapter:
    """Classifier adapter: assigns a deflection category for out-of-scope queries.
    - Non-streaming
    - Returns DeflectionCategoryDecision via engine's non-streaming LLM call
    - Dynamic few-shot selection may be injected by the engine; adapter provides a static fallback
    """

    name: str = "deflection_categorizer"
    response_model = DeflectionCategoryDecision

    def build_prompt(self, state: MessagesState) -> PromptBundle:
        # Static few-shots from templates; engine may override with dynamic examples
        system_prompt = render_deflection_categorizer(state.user_query)
        messages = [
            {"role": "system", "content": system_prompt},
            *build_conversation_history(state, max_history=24, include_tool_content=True),
            {"role": "user", "content": state.user_query}
        ]
        logger.debug("Built deflection_categorizer prompt", extra={"thread_id": state.thread_id})
        return PromptBundle(messages=messages, should_stream=False)

    async def postprocess(self, state: MessagesState, validated_response: DeflectionCategoryDecision) -> MessagesState:
        logger.info(
            "Deflection category decided",
            extra={"thread_id": state.thread_id, "category": validated_response.decision},
        )
        # Opik for logging span under trace.
        opik_context.update_current_span(
            name=self.name,
            input={"user_query": state.user_query},
            output={"response": validated_response.decision},
            metadata={"retrieved_docs": state.meta.get("retrieved_docs", [])},
            )
        
        return state

    async def augment_metadata(self, state: MessagesState, start_time, end_time) -> Dict[str, Any]:
        return {}


class GenerateAnswerAdapter:
    """Generator adapter: streams the final answer in a single node.
    - Streaming
    - Engine renders system prompt via render_generate_answer() based on context_mode
    - Engine is responsible for mapping stream deltas to StreamingResponse
    """

    name: str = "generate_answer"
    response_model = GenerationResponse

    def build_prompt(self, state: MessagesState) -> PromptBundle:
        system_prompt = render_prompt_generate_answer(
            state.meta.get("context_mode", ""),
            user_query=state.user_query,            
            deflection_category=state.meta.get("deflection_category", ""),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            *build_conversation_history(state, max_history=24, include_tool_content=True),
            {"role": "user", "content": state.user_query}
        ]
        logger.debug("Built generate_answer prompt (streaming)", extra={"system_prompt": system_prompt})

        return PromptBundle(messages=messages, should_stream=True)

    async def postprocess(self, state: MessagesState, validated_response: GenerationResponse) -> MessagesState:
        logger.info(
            "Generated final answer (postprocess)",
            extra={"thread_id": state.thread_id, "text_len": len(validated_response.text)},
        )
        # Opik for logging span under trace.
        opik_context.update_current_span(
            name=self.name,
            input={"user_query": state.user_query},
            output={"response": validated_response.text},
            metadata={"retrieved_docs": state.meta.get("retrieved_docs", [])},
            )
        state.messages.append(AIMessage(content=validated_response.text))
        state.update_thread_store()

        return state

    async def augment_metadata(self, state: MessagesState, start_time, end_time) -> Dict[str, Any]:
        return {}


