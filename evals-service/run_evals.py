from pathlib import Path
from dotenv import load_dotenv
from utils.logger import logger
from typing import Dict, List, Optional
import backoff
import time
from datetime import datetime
from models import EvaluationResult, RetryConfig, ConversationFlow
from evaluators import (
    evaluate_relevance_check,
    # evaluate_query_or_respond,
    # evaluate_few_shot_selector,
    # evaluate_generate_with_context,
    # evaluate_generate_with_persona,
)
from config import config
import traceback

# Load environment variables from .env file
# Get the project root directory (assuming evals is in project root)
project_root = Path(__file__).parent
env_path = project_root / ".env"
load_dotenv(env_path)


def extract_category_from_response(response: str) -> str:
    """
    Extract category from few-shot selector response.

    Expected format: "Category: OFFICIAL\nStyle: ..."
    Returns: "OFFICIAL" or "OFFICIAL" as fallback
    """
    if not response:
        return "OFFICIAL"

    # Look for "Category: " pattern
    if "Category: " in response:
        category_line = response.split("Category: ")[1].split("\n")[0].strip()
        # Validate it's one of the expected categories
        if category_line in ["OFFICIAL", "JEST", "HACK"]:
            return category_line

    # Fallback: try to extract from the beginning of the response
    response_upper = response.upper().strip()
    for category in ["OFFICIAL", "JEST", "HACK"]:
        if category in response_upper:
            return category

    # Default fallback
    logger.warning(f"Could not extract category from response: {response[:100]}...")
    return "OFFICIAL"


class AsyncEvaluator:
    """
    Stateless evaluator that processes conversation flows and returns evaluation results.
    Used by the FastAPI app for evaluation processing.
    """

    def __init__(self):
        self.retry_config = RetryConfig()
        self._evaluation_count = 0
        self._error_count = 0

        logger.info("Initialized AsyncEvaluator")

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=config.service.max_retry_attempts,
        max_time=30,
    )
    async def evaluate_response(
        self,
        thread_id: str,
        query: str,
        response: str,
        retrieved_docs: Optional[List[Dict[str, str]]] = None,
        conversation_flow: Optional[ConversationFlow] = None,
    ) -> EvaluationResult:
        """
        Evaluates a complete conversation flow through all executed nodes.
        """
        try:
            # Start evaluation timing
            eval_start_time = time.monotonic()

            logger.info(
                f"EVALUATOR_START: Starting evaluation for thread_id={thread_id}",
                extra={
                    "thread_id": thread_id,
                    "node_count": len(conversation_flow.node_executions)
                    if conversation_flow
                    else 0,
                },
            )

            evaluations = {}
            eval_token_usage = {"prompt": 0, "completion": 0}

            # We are only evaluating the 'relevance_check' node for now.
            for node_execution in conversation_flow.node_executions:
                if node_execution.node_name == "relevance_check":
                    eval_result = await evaluate_relevance_check(
                        node_execution=node_execution, user_query=query
                    )
                    evaluations[node_execution.node_name] = eval_result.model_dump()

                    # TODO: When evaluators are updated to return token usage, capture it here
                    # eval_token_usage["prompt"] += eval_result.prompt_tokens or 0
                    # eval_token_usage["completion"] += eval_result.completion_tokens or 0

                    logger.info(
                        f"EVALUATOR_NODE_PROCESSED: Processed node '{node_execution.node_name}' for thread_id={thread_id}",
                        extra={
                            "thread_id": thread_id,
                            "node_name": node_execution.node_name,
                        },
                    )
                else:
                    logger.info(
                        f"EVALUATOR_NODE_SKIPPED: Skipping node '{node_execution.node_name}' for thread_id={thread_id}",
                        extra={
                            "thread_id": thread_id,
                            "node_name": node_execution.node_name,
                        },
                    )

            # Calculate evaluation latency
            eval_latency_ms = (time.monotonic() - eval_start_time) * 1000

            logger.info(
                f"EVALUATOR_SUCCESS: Evaluation completed for thread_id={thread_id}",
                extra={
                    "thread_id": thread_id,
                    "result": evaluations,
                    "eval_latency_ms": eval_latency_ms,
                },
            )

            # Return the enhanced structured result
            return EvaluationResult(
                # Identifiers from the graph flow
                run_id=conversation_flow.run_id,
                thread_id=thread_id,
                turn_index=conversation_flow.turn_index,
                # Timestamps & Latency
                timestamp_start=conversation_flow.start_time,
                timestamp_end=datetime.utcnow(),
                graph_latency_ms=conversation_flow.latency_ms,
                time_to_first_token_ms=conversation_flow.time_to_first_token_ms,
                evaluation_latency_ms=eval_latency_ms,
                # Core conversation data
                query=query,
                response=response,
                retrieved_docs=retrieved_docs,
                # Token Counts
                graph_total_prompt_tokens=conversation_flow.total_prompt_tokens,
                graph_total_completion_tokens=conversation_flow.total_completion_tokens,
                eval_total_prompt_tokens=eval_token_usage["prompt"],
                eval_total_completion_tokens=eval_token_usage["completion"],
                # The actual evaluation results
                evaluations=evaluations,
                # Overall metadata
                metadata={"overall_success": bool(evaluations)},
            )

        except Exception as e:
            self._error_count += 1
            logger.error(
                f"EVALUATOR_FAILURE: Evaluation failed for thread_id={thread_id}",
                extra={
                    "thread_id": thread_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            # Return a failure result with enhanced structure
            return EvaluationResult(
                # Identifiers - use defaults if conversation_flow is None
                run_id=conversation_flow.run_id if conversation_flow else "unknown",
                thread_id=thread_id,
                turn_index=conversation_flow.turn_index if conversation_flow else 0,
                # Timestamps & Latency - use current time if conversation_flow is None
                timestamp_start=conversation_flow.start_time
                if conversation_flow
                else datetime.utcnow(),
                timestamp_end=datetime.utcnow(),
                graph_latency_ms=conversation_flow.latency_ms
                if conversation_flow
                else None,
                time_to_first_token_ms=conversation_flow.time_to_first_token_ms
                if conversation_flow
                else None,
                evaluation_latency_ms=None,  # Could not be calculated due to error
                # Core conversation data
                query=query,
                response=response,
                retrieved_docs=retrieved_docs,
                # Token Counts - use values from conversation_flow if available
                graph_total_prompt_tokens=conversation_flow.total_prompt_tokens
                if conversation_flow
                else None,
                graph_total_completion_tokens=conversation_flow.total_completion_tokens
                if conversation_flow
                else None,
                eval_total_prompt_tokens=None,
                eval_total_completion_tokens=None,
                # The actual evaluation results
                evaluations={},
                # Overall metadata
                metadata={"overall_success": False, "error": str(e)},
            )

    async def health_check(self) -> dict:
        """Perform health check on evaluator."""
        try:
            # Test LLM connectivity by making a simple API call
            # This is a basic health check - in production you might want more sophisticated checks
            return {
                "evaluator_healthy": True,
                "evaluation_count": self._evaluation_count,
                "error_count": self._error_count,
                "llm_model": config.llm.openai_model,
                "llm_configured": bool(config.llm.openai_api_key),
            }
        except Exception as e:
            logger.error("Evaluator health check failed", extra={"error": str(e)})
            return {"evaluator_healthy": False, "error": str(e)}

    async def get_metrics(self) -> dict:
        """Get evaluator metrics."""
        return {
            "evaluation_count": self._evaluation_count,
            "error_count": self._error_count,
            "success_rate": (
                (self._evaluation_count - self._error_count)
                / max(self._evaluation_count, 1)
            ),
            "llm_model": config.llm.openai_model,
            "max_retry_attempts": config.service.max_retry_attempts,
        }
