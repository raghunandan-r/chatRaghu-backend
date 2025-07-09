import os
import backoff
from opik import track, opik_context

from config import config
from utils.logger import logger
from models import EnrichedNodeExecutionLog

from .base import client, get_eval_prompt, get_system_message, get_main_graph_prompt
from .judgements import LLMGenerateWithPersonaJudgement
from .models import GenerateWithPersonaEval


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=config.llm.openai_max_retries,
    max_time=config.llm.openai_timeout_seconds,
)
@track(capture_input=True, project_name=os.getenv("OPIK_EVALS_SERVICE_PROJECT"))
async def evaluate_generate_with_persona(
    node_execution: EnrichedNodeExecutionLog, user_query: str
) -> GenerateWithPersonaEval:
    """Evaluates the generate_with_persona node output using a structured LLM call."""

    model_output = node_execution.output.get("response", "")
    previous_ai_message = ""
    message_history = []
    if node_execution.input and "messages" in node_execution.input:
        messages = node_execution.input.get("messages", [])
        for message in messages:
            msg_type = message.get("type")
            content = message.get("content", "")
            message_history.append(f"{msg_type}: {content}")
        for message in reversed(messages):
            if message.get("type") == "ai":
                previous_ai_message = message.get("content", "")
                break

    system_prompt = node_execution.system_prompt or get_main_graph_prompt(
        "generate_with_persona"
    )

    logger.info(
        "Starting generate_with_persona evaluation",
        extra={"user_query": user_query, "response_length": len(model_output)},
    )

    message_history_str = "\n".join(message_history)

    eval_prompt = get_eval_prompt(
        "generate_with_persona",
        system_prompt=system_prompt,
        message_history_str=message_history_str,
        user_query=user_query,
        previous_ai_message=previous_ai_message or "None",
        model_output=model_output,
    )
    system_message = get_system_message("generate_with_persona")

    try:
        judgement, completion = await client.chat.completions.create_with_completion(
            model=config.llm.openai_model,
            response_model=LLMGenerateWithPersonaJudgement,
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
                judgement.persona_adherence,
                judgement.follows_rules,
                judgement.faithfulness,
                judgement.answer_relevance,
                judgement.handles_irrelevance,
                judgement.context_relevance,
            ]
        )

        opik_context.update_current_span(
            name="generate_with_persona",
            input={
                "query": user_query,
                "response_length": len(model_output),
                "message_history": message_history_str,
            },
            output={
                "overall_success": overall_success,
                "persona_adherence": judgement.persona_adherence,
                "follows_rules": judgement.follows_rules,
            },
            metadata={
                "system_prompt": system_prompt,
                "llm_judgement": judgement.model_dump(),
            },
        )

        logger.info(
            "EVAL_NODE_PROCESSED: Completed generate_with_persona evaluation",
            extra={
                "user_query": user_query,
                "response_length": len(model_output),
                "overall_success": overall_success,
                "result": judgement.model_dump_json(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

        return GenerateWithPersonaEval(
            node_name="generate_with_persona",
            overall_success=overall_success,
            persona_adherence=judgement.persona_adherence,
            follows_rules=judgement.follows_rules,
            faithfulness=judgement.faithfulness,
            answer_relevance=judgement.answer_relevance,
            handles_irrelevance=judgement.handles_irrelevance,
            context_relevance=judgement.context_relevance,
            explanation=judgement.explanation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    except Exception as e:
        logger.error(
            "Failed generate_with_persona evaluation",
            extra={"error": str(e), "user_query": user_query},
        )
        raise
