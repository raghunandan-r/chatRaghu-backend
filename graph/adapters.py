from __future__ import annotations

from typing import Any, Dict

from utils.logger import logger

from graph.engine import PromptBundle
from graph.models import MessagesState
from graph.schemas import (
    RelevanceDecision,
    RoutingDecision,
    DeflectionCategoryDecision,
    GenerationResponse,
)
from graph.template_renderer import (
    render_simple_template,
    render_deflection_categorizer,
)


class RelevanceCheckAdapter:
    """Gate adapter: classifies a query as RELEVANT or IRRELEVANT.

    - Non-streaming
    - Returns RelevanceDecision via engine's non-streaming LLM call
    """

    name: str = "relevance_check"
    response_model = RelevanceDecision

    def build_prompt(self, state: MessagesState) -> PromptBundle:
        current_query = next(
            (m.content for m in reversed(state.messages) if getattr(m, "type", "") == "human"),
            "",
        )
        system_prompt = render_simple_template("relevance_check", query=current_query)
        messages = [{"role": "user", "content": current_query}]
        logger.debug("Built relevance_check prompt", extra={"thread_id": state.thread_id})
        return PromptBundle(system_prompt=system_prompt, messages=messages, should_stream=False)

    async def postprocess(self, state: MessagesState, validated_response: RelevanceDecision) -> MessagesState:
        logger.info(
            "Relevance decision",
            extra={"thread_id": state.thread_id, "decision": validated_response.decision},
        )
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
        current_query = next(
            (m.content for m in reversed(state.messages) if getattr(m, "type", "") == "human"),
            "",
        )
        system_prompt = render_simple_template("query_or_respond", query=current_query)
        messages = [{"role": "user", "content": current_query}]
        logger.debug("Built query_or_respond prompt", extra={"thread_id": state.thread_id})
        return PromptBundle(system_prompt=system_prompt, messages=messages, should_stream=False)

    async def postprocess(self, state: MessagesState, validated_response: RoutingDecision) -> MessagesState:
        logger.info(
            "Routing decision",
            extra={
                "thread_id": state.thread_id,
                "decision": validated_response.decision,
                "query_for_retrieval": validated_response.query_for_retrieval,
            },
        )
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
        current_query = next(
            (m.content for m in reversed(state.messages) if getattr(m, "type", "") == "human"),
            "",
        )
        # Static few-shots from templates; engine may override with dynamic examples
        system_prompt = render_deflection_categorizer(current_query)
        logger.debug(
            "Built deflection_categorizer prompt",
            extra={"thread_id": state.thread_id},
        )
        return PromptBundle(system_prompt=system_prompt, messages=[], should_stream=False)

    async def postprocess(self, state: MessagesState, validated_response: DeflectionCategoryDecision) -> MessagesState:
        logger.info(
            "Deflection category decided",
            extra={"thread_id": state.thread_id, "category": validated_response.category},
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
        logger.debug("Built generate_answer prompt (streaming)", extra={"thread_id": state.thread_id})
        return PromptBundle(system_prompt=None, messages=[], should_stream=True)

    async def postprocess(self, state: MessagesState, validated_response: GenerationResponse) -> MessagesState:
        logger.info(
            "Generated final answer (postprocess)",
            extra={"thread_id": state.thread_id, "text_len": len(validated_response.text)},
        )
        return state

    async def augment_metadata(self, state: MessagesState, start_time, end_time) -> Dict[str, Any]:
        return {}


