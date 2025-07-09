import os
import backoff
from opik import track, opik_context

from config import config
from utils.logger import logger
from models import EnrichedNodeExecutionLog

from .base import client, get_eval_prompt, get_system_message, get_main_graph_prompt
from .judgements import LLMRelevanceJudgement
from .models import RelevanceCheckEval


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=config.llm.openai_max_retries,
    max_time=config.llm.openai_timeout_seconds,
)
@track(capture_input=True, project_name=os.getenv("OPIK_EVALS_SERVICE_PROJECT"))
async def evaluate_relevance_check(
    node_execution: EnrichedNodeExecutionLog, user_query: str
) -> RelevanceCheckEval:
    """Evaluates the relevance_check node output using a structured LLM call."""

    model_output = node_execution.output.get("messages", [{}])[0].get("content", "")
    conversation_history = node_execution.input.get("conversation_history", [])
    original_system_prompt = get_main_graph_prompt("relevance_check")

    logger.info(
        "Starting relevance_check evaluation",
        extra={"user_query": user_query, "model_output": model_output},
    )

    eval_prompt = get_eval_prompt(
        "relevance_check",
        original_system_prompt=original_system_prompt,
        user_query=user_query,
        conversation_history=conversation_history,
        model_output=model_output,
    )
    system_message = get_system_message("relevance_check")

    try:
        judgement, completion = await client.chat.completions.create_with_completion(
            model=config.llm.openai_model,
            response_model=LLMRelevanceJudgement,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": eval_prompt},
            ],
        )

        prompt_tokens = completion.usage.prompt_tokens if completion.usage else None
        completion_tokens = (
            completion.usage.completion_tokens if completion.usage else None
        )

        opik_context.update_current_span(
            name="relevance_check",
            input={"query": user_query, "history": conversation_history},
            output={"classification": judgement.classification_correct},
            metadata={
                "system_prompt": eval_prompt,
                "llm_judgement": judgement.model_dump(),
            },
        )
        logger.info(
            "EVAL_NODE_PROCESSED: Completed relevance_check evaluation",
            extra={
                "user_query": user_query,
                "model_output": model_output,
                "result": judgement.model_dump_json(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

        return RelevanceCheckEval(
            node_name="relevance_check",
            overall_success=judgement.classification_correct,
            classification=model_output,
            format_valid=judgement.format_valid,
            explanation=judgement.explanation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    except Exception as e:
        logger.error(
            "Failed relevance_check evaluation",
            extra={"error": str(e), "user_query": user_query},
        )
        raise
