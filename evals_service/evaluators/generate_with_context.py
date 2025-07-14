import os
import json
import backoff
from opik import track, opik_context

from config import config
from utils.logger import logger
from models import EnrichedNodeExecutionLog

from .base import client, get_eval_prompt, get_system_message
from .judgements import LLMGenerateWithContextJudgement
from .models import GenerateWithContextEval


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=config.llm.openai_max_retries,
    max_time=config.llm.openai_timeout_seconds,
)
@track(capture_input=True, project_name=os.getenv("OPIK_EVALS_SERVICE_PROJECT"))
async def evaluate_generate_with_context(
    node_execution: EnrichedNodeExecutionLog, user_query: str
) -> GenerateWithContextEval:
    """Evaluates the generate_with_context node output using a structured LLM call."""

    model_output = node_execution.output.get("messages", [{}])[0].get("content", "")
    retrieved_docs = node_execution.retrieved_docs or []
    # original_system_prompt = get_main_graph_prompt("generate_with_context")
    conversation_history = node_execution.input.get("conversation_history", [])

    logger.info(
        "Starting generate_with_context evaluation",
        extra={
            "user_query": user_query,
            "response_length": len(model_output),
            "docs_count": len(retrieved_docs),
        },
    )

    docs_content = []
    for i, doc in enumerate(retrieved_docs):
        content = doc.get("content", "")
        score = doc.get("score", 0.0)
        metadata = doc.get("metadata", {})
        docs_content.append(
            f"Document {i+1} (Score: {score:.3f}):\n{content}\n"
            f"Metadata: {json.dumps(metadata, indent=2)}\n"
        )
    docs_text = "\n".join(docs_content) if docs_content else "No documents retrieved."

    eval_prompt = get_eval_prompt(
        "generate_with_context",
        # original_system_prompt=original_system_prompt,
        user_query=user_query,
        conversation_history=conversation_history,
        docs_text=docs_text,
        model_output=model_output,
    )
    system_message = get_system_message("generate_with_context")

    try:
        judgement, completion = await client.chat.completions.create_with_completion(
            model=config.llm.openai_model,
            response_model=LLMGenerateWithContextJudgement,
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
                judgement.context_relevance,
            ]
        )

        opik_context.update_current_span(
            name="generate_with_context",
            input={
                "query": user_query,
                "response_length": len(model_output),
                "docs_count": len(retrieved_docs),
            },
            output={
                "overall_success": overall_success,
                "faithful": judgement.faithfulness,
                "answer_relevance": judgement.answer_relevance,
                "includes_key_info": judgement.includes_key_info,
                "handles_irrelevance": judgement.handles_irrelevance,
                "context_relevance": judgement.context_relevance,
            },
            metadata={
                # "system_prompt": original_system_prompt,
                "llm_judgement": judgement.model_dump(),
                "docs_scores": [doc.get("score", 0.0) for doc in retrieved_docs],
            },
        )

        logger.info(
            "EVAL_NODE_PROCESSED: Completed generate_with_context evaluation",
            extra={
                "user_query": user_query,
                "response_length": len(model_output),
                "overall_success": overall_success,
                "result": judgement.model_dump_json(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

        return GenerateWithContextEval(
            node_name="generate_with_context",
            overall_success=overall_success,
            faithfulness=judgement.faithfulness,
            answer_relevance=judgement.answer_relevance,
            includes_key_info=judgement.includes_key_info,
            handles_irrelevance=judgement.handles_irrelevance,
            context_relevance=judgement.context_relevance,
            explanation=judgement.explanation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    except Exception as e:
        logger.error(
            "Failed generate_with_context evaluation",
            extra={
                "error": str(e),
                "user_query": user_query,
                "response_length": len(model_output),
                "docs_count": len(retrieved_docs),
            },
        )
        raise
