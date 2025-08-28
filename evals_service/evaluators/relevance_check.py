import os
import backoff
from opik import track, opik_context

from config import config
from utils.logger import logger
from models import EnrichedNodeExecutionLog

from .base import client, get_eval_prompt, get_system_message
from .judgements import LLMRouterJudgement
from .models import RouterEval


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
async def evaluate_router(
    node_execution: EnrichedNodeExecutionLog, user_query: str
) -> RouterEval:
    """Evaluates the router node output using a structured LLM call."""

    model_output = node_execution.output.get("decision", "")
    conversation_history = node_execution.input.get("conversation_history", [])
    # original_system_prompt = get_main_graph_prompt("relevance_check")

    logger.info(
        "Starting router evaluation",
        extra={"user_query": user_query, "model_output": model_output},
    )

    eval_prompt = get_eval_prompt(
        "router",
        # original_system_prompt=original_system_prompt,
        user_query=user_query,
        conversation_history=conversation_history,
        model_output=model_output,
    )
    system_message = get_system_message("router")

    if not eval_prompt or not system_message:
        logger.error(
            "Missing evaluator prompts; skipping evaluation",
            extra={"evaluator": "router"},
        )
        return RouterEval(
            node_name="router",
            overall_success=False,
            routing_correct=False,
            explanation="Evaluator prompts missing; evaluation skipped.",
        )

    try:
        judgement, completion = await client.chat.completions.create_with_completion(
            model=config.llm.openai_model,
            response_model=LLMRouterJudgement,
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
            name="router",
            input=eval_prompt,
            output={
                "overall_success": judgement.routing_correct,
                "routing_correct": judgement.routing_correct,
                "explanation": judgement.explanation,
            },
            metadata={
                "system_prompt": system_message,
                "llm_judgement": judgement.model_dump(),
            },
        )
        logger.info(
            "EVAL_NODE_PROCESSED: Completed router evaluation",
            extra={
                "user_query": user_query,
                "model_output": model_output,
                "result": judgement.model_dump_json(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

        return RouterEval(
            node_name="router",
            overall_success=judgement.routing_correct,
            routing_correct=judgement.routing_correct,
            explanation=judgement.explanation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    except Exception as e:
        logger.error(
            "Failed router evaluation",
            extra={"error": str(e), "user_query": user_query},
        )
        raise
