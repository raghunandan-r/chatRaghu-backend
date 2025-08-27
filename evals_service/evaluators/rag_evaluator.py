import os
import json
import backoff
from opik import track, opik_context

from config import config
from utils.logger import logger
from models import EnrichedNodeExecutionLog

from .base import client, get_eval_prompt, get_system_message
from .judgements import LLMRAGJudgement
from .models import RAGEval


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=config.llm.openai_max_retries,
    max_time=config.llm.openai_timeout_seconds,
)
@track(capture_input=False, capture_output=False, project_name=os.getenv("OPIK_EVALS_SERVICE_PROJECT"))
async def evaluate_rag(
    node_execution: EnrichedNodeExecutionLog, user_query: str
) -> RAGEval:
    """Evaluates the RAG adapter output using a structured LLM call."""

    model_output = node_execution.output.get("text", "")
    retrieved_docs = node_execution.retrieved_docs or []
    conversation_history = node_execution.input.get("conversation_history", [])

    logger.info(
        "Starting RAG evaluation",
        extra={
            "user_query": user_query,
            "response": model_output,
            "docs": retrieved_docs,
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
        "rag",
        user_query=user_query,
        conversation_history=conversation_history,
        docs_text=docs_text,
        model_output=model_output,
    )
    system_message = get_system_message("rag")

    try:
        judgement, completion = await client.chat.completions.create_with_completion(
            model=config.llm.openai_model,
            response_model=LLMRAGJudgement,
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
                judgement.document_relevance,
            ]
        )

        opik_context.update_current_span(
            name="rag",
            input=eval_prompt,
            output={
                "overall_success": overall_success,
                "faithful": judgement.faithfulness,
                "answer_relevance": judgement.answer_relevance,
                "includes_key_info": judgement.includes_key_info,
                "handles_irrelevance": judgement.handles_irrelevance,
                "document_relevance": judgement.document_relevance,
                "explanation": judgement.explanation,
            },
            metadata={
                "llm_judgement": judgement.model_dump(),
                "docs_scores": [doc.get("score", 0.0) for doc in retrieved_docs],
            },
        )

        logger.info(
            "EVAL_NODE_PROCESSED: Completed RAG evaluation",
            extra={
                "user_query": user_query,                
                "overall_success": overall_success,
                "result": judgement.model_dump_json(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

        return RAGEval(
            node_name="generate_answer_with_rag",
            overall_success=overall_success,
            faithfulness=judgement.faithfulness,
            answer_relevance=judgement.answer_relevance,
            includes_key_info=judgement.includes_key_info,
            handles_irrelevance=judgement.handles_irrelevance,
            document_relevance=judgement.document_relevance,
            explanation=judgement.explanation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    except Exception as e:
        logger.error(
            "Failed RAG evaluation",
            extra={
                "error": str(e),
                "user_query": user_query,
                "response": model_output,
                "docs": retrieved_docs,
            },
        )
        raise
