import os
import backoff
from opik import track, opik_context

from config import config
from utils.logger import logger
from models import EnrichedNodeExecutionLog

from .base import client, get_eval_prompt, get_system_message
from .judgements import LLMHistoryJudgement
from .models import HistoryEval


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=config.llm.openai_max_retries,
    max_time=config.llm.openai_timeout_seconds,
)
@track(
    capture_input=False,
    capture_output=False,
    project_name=os.getenv("OPIK_EVALS_SERVICE_PROJECT"),
)
async def evaluate_history(
    node_execution: EnrichedNodeExecutionLog, user_query: str, graph_version: str
) -> HistoryEval:
    """Evaluates the History adapter output using a structured LLM call."""

    model_output = node_execution.output.get("text", "")
    conversation_history = node_execution.input.get("conversation_history", [])

    logger.info(
        "Starting History evaluation",
        extra={
            "user_query": user_query,
            "response": model_output,
            "history": conversation_history,
        },
    )

    eval_prompt = get_eval_prompt(
        "history",
        user_query=user_query,
        conversation_history=conversation_history,
        model_output=model_output,
        graph_version=graph_version,
    )
    system_message = get_system_message("history", graph_version)

    try:
        judgement, completion = await client.chat.completions.create_with_completion(
            model=config.llm.openai_model,
            response_model=LLMHistoryJudgement,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": eval_prompt},
            ],
        )

        prompt_tokens = completion.usage.prompt_tokens if completion.usage else None
        completion_tokens = (
            completion.usage.completion_tokens if completion.usage else None
        )

        overall_success = all(
            [
                judgement.faithfulness,
                judgement.answer_relevance,
                judgement.includes_key_info,
                judgement.handles_irrelevance,
                judgement.history_relevance,
                judgement.is_safe,
                judgement.is_clear,
            ]
        )

        opik_context.update_current_span(
            name="history",
            input=eval_prompt,
            output={
                "overall_success": overall_success,
                "faithful_to_history": judgement.faithfulness,
                "answer_relevance": judgement.answer_relevance,
                "includes_key_info": judgement.includes_key_info,
                "handles_irrelevance": judgement.handles_irrelevance,
                "history_relevance": judgement.history_relevance,
                "explanation": judgement.explanation,
                "is_safe": judgement.is_safe,
                "is_clear": judgement.is_clear,
            },
            metadata={
                "llm_judgement": judgement.model_dump(),
            },
        )

        logger.info(
            "EVAL_NODE_PROCESSED: Completed History evaluation",
            extra={
                "user_query": user_query,
                "overall_success": overall_success,
                "result": judgement.model_dump_json(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

        return HistoryEval(
            node_name="generate_answer_with_history",
            overall_success=overall_success,
            faithfulness=judgement.faithfulness,
            answer_relevance=judgement.answer_relevance,
            includes_key_info=judgement.includes_key_info,
            handles_irrelevance=judgement.handles_irrelevance,
            history_relevance=judgement.history_relevance,
            explanation=judgement.explanation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            is_safe=judgement.is_safe,
            is_clear=judgement.is_clear,
        )

    except Exception as e:
        logger.error(
            "Failed History evaluation",
            extra={
                "error": str(e),
                "user_query": user_query,
                "response": model_output,
            },
        )
        raise
