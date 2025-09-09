import os
import backoff
from opik import track, opik_context

from config import config
from utils.logger import logger
from models import EnrichedNodeExecutionLog

from .base import client, get_eval_prompt, get_system_message
from .judgements import LLMSimpleResponseJudgement
from .models import SimpleResponseEval


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
async def evaluate_simple_response(
    node_execution: EnrichedNodeExecutionLog, user_query: str, graph_version: str
) -> SimpleResponseEval:
    """Evaluates the SimpleResponse adapter output using a structured LLM call."""

    model_output = node_execution.output.get("text", "")
    conversation_history = node_execution.input.get("conversation_history", [])

    logger.info(
        "Starting SimpleResponse evaluation",
        extra={
            "user_query": user_query,
            "response": model_output,
        },
    )

    eval_prompt = get_eval_prompt(
        "simple_response",
        user_query=user_query,
        conversation_history=conversation_history,
        org_system_prompt=node_execution.system_prompt,
        model_output=model_output,
        graph_version=graph_version,
    )
    system_message = get_system_message("simple_response", graph_version)

    if not eval_prompt or not system_message:
        logger.error(
            "Missing evaluator prompts; skipping evaluation",
            extra={"evaluator": "simple_response"},
        )
        return SimpleResponseEval(
            node_name="generate_simple_response",
            overall_success=False,
            handles_irrelevance=False,
            response_appropriateness=False,
            is_safe=False,
            is_clear=False,
            explanation="Evaluator prompts missing; evaluation skipped.",
        )

    try:
        judgement, completion = await client.chat.completions.create_with_completion(
            model=config.llm.openai_model,
            response_model=LLMSimpleResponseJudgement,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": eval_prompt},
            ],
            max_tokens=4096,
        )

        prompt_tokens = completion.usage.prompt_tokens if completion.usage else None
        completion_tokens = (
            completion.usage.completion_tokens if completion.usage else None
        )

        overall_success = all(
            [
                judgement.response_appropriateness,
                judgement.handles_irrelevance,
            ]
        )

        opik_context.update_current_span(
            name="simple_response",
            input=eval_prompt,
            output={
                "overall_success": overall_success,
                "response_appropriateness": judgement.response_appropriateness,
                "handles_irrelevance": judgement.handles_irrelevance,
                "explanation": judgement.explanation,
                "is_safe": judgement.is_safe,
                "is_clear": judgement.is_clear,
            },
            metadata={
                "llm_judgement": judgement.model_dump(),
            },
        )

        logger.info(
            "EVAL_NODE_PROCESSED: Completed SimpleResponse evaluation",
            extra={
                "user_query": user_query,
                "overall_success": overall_success,
                "result": judgement.model_dump_json(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "is_safe": judgement.is_safe,
                "is_clear": judgement.is_clear,
            },
        )

        return SimpleResponseEval(
            node_name="generate_simple_response",
            overall_success=overall_success,
            handles_irrelevance=judgement.handles_irrelevance,
            response_appropriateness=judgement.response_appropriateness,
            explanation=judgement.explanation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            is_safe=judgement.is_safe,
            is_clear=judgement.is_clear,
        )

    except Exception as e:
        logger.error(
            "Failed SimpleResponse evaluation",
            extra={
                "error": str(e),
                "user_query": user_query,
                "response": model_output,
            },
        )
        raise
