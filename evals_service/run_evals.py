"""
Main evaluation runner for processing node outputs.
"""

import backoff
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from pathlib import Path
from utils.logger import logger
from models import (
    EvaluationResult,
    RetryConfig,
    ConversationFlow,
    EnrichedNodeExecutionLog,
)
from evaluators.models import NodeEvaluation
from evaluators import EVALUATOR_REGISTRY
from config import config

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
        conversation_flow: ConversationFlow,
    ) -> EvaluationResult:
        """
        Evaluates a complete conversation flow through all executed nodes.
        """
        try:
            # Start evaluation timing
            eval_start_time = time.monotonic()

            logger.info(
                f"EVALUATOR_START: Starting evaluation for thread_id={conversation_flow.thread_id}",
                extra={
                    "thread_id": conversation_flow.thread_id,
                    "node_count": len(conversation_flow.node_executions)
                    if conversation_flow
                    else 0,
                },
            )

            evaluations = []  # Changed to list for flattened structure
            eval_token_usage = {"prompt": 0, "completion": 0}

            # Evaluate relevant nodes in the conversation flow using the registry
            for node_execution in conversation_flow.node_executions:
                node_name = node_execution.node_name
                evaluator_functions = EVALUATOR_REGISTRY.get(node_name, [])

                if not evaluator_functions:
                    logger.info(
                        f"EVALUATOR_NODE_SKIPPED: No evaluators registered for node '{node_name}' for thread_id={conversation_flow.thread_id}",
                        extra={"thread_id": conversation_flow.thread_id, "node_name": node_name},
                    )
                    continue

                for evaluator_func in evaluator_functions:
                    eval_name = evaluator_func.__name__
                    try:
                        eval_result = await evaluator_func(
                            node_execution=node_execution, user_query=conversation_flow.user_query
                        )

                        # Get the evaluation result as dict
                        eval_dump = eval_result.model_dump()

                        # Truncate long explanation strings
                        if "explanation" in eval_dump and isinstance(
                            eval_dump["explanation"], str
                        ):
                            eval_dump["explanation"] = (
                                eval_dump["explanation"][:500] + "..."
                                if len(eval_dump["explanation"]) > 500
                                else eval_dump["explanation"]
                            )

                        # Flatten the result structure
                        flattened_eval = {
                            "node_name": node_name,
                            "evaluator_name": eval_name,
                            **eval_dump,
                        }
                        evaluations.append(flattened_eval)

                        # Centrally aggregate token usage
                        eval_token_usage["prompt"] += eval_result.prompt_tokens or 0
                        eval_token_usage["completion"] += (
                            eval_result.completion_tokens or 0
                        )

                    except Exception as e:
                        logger.error(
                            f"EVALUATOR_SUB_EVAL_FAILED: Evaluator '{eval_name}' failed for node '{node_name}'",
                            extra={
                                "thread_id": conversation_flow.thread_id,
                                "node_name": node_name,
                                "evaluator": eval_name,
                                "error": str(e),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        # Record the failure in flattened structure
                        evaluations.append(
                            {
                                "node_name": node_name,
                                "evaluator_name": eval_name,
                                "error": str(e),
                                "error_type": type(e).__name__,
                                "status": "failed",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                        continue

            # Calculate evaluation latency
            eval_latency_ms = (time.monotonic() - eval_start_time) * 1000

            # Calculate total latency
            total_latency_ms = (
                (datetime.now(timezone.utc) - conversation_flow.start_time).total_seconds()
                * 1000
                if conversation_flow and conversation_flow.start_time
                else None
            )

            # Calculate overall success based on all evaluation results
            def check_all_evaluations_success(evaluations_list):
                """Check if all evaluation results have overall_success=True"""
                if not evaluations_list:
                    return False

                for result in evaluations_list:
                    # Skip failed evaluations
                    if "error" in result or result.get("status") == "failed":
                        continue
                    # Check if overall_success is False
                    if result.get("overall_success") is False:
                        return False
                return True

            overall_success = check_all_evaluations_success(evaluations)

            logger.info(
                f"EVALUATOR_SUCCESS: Evaluation completed for thread_id={conversation_flow.thread_id}",
                extra={
                    "thread_id": conversation_flow.thread_id,
                    "result": evaluations,
                    "eval_latency_ms": eval_latency_ms,
                    "total_latency_ms": total_latency_ms,
                    "eval_prompt_tokens": eval_token_usage["prompt"],
                    "eval_completion_tokens": eval_token_usage["completion"],
                    "total_eval_tokens": eval_token_usage["prompt"]
                    + eval_token_usage["completion"],
                    "overall_success": overall_success,
                },
            )

            # Return the enhanced structured result
            return EvaluationResult(
                # Identifiers from the graph flow
                run_id=conversation_flow.run_id,
                thread_id=conversation_flow.thread_id,
                turn_index=conversation_flow.turn_index,
                graph_version=conversation_flow.graph_version,
                # Timestamps & Latency
                timestamp_start=conversation_flow.start_time,
                timestamp_end=datetime.now(timezone.utc),
                graph_latency_ms=conversation_flow.latency_ms,
                time_to_first_token_ms=conversation_flow.time_to_first_token_ms,
                evaluation_latency_ms=eval_latency_ms,
                total_latency_ms=total_latency_ms,
                # Core conversation data
                query=conversation_flow.user_query,
                response=conversation_flow.final_response,
                retrieved_docs=conversation_flow.node_executions[-1].retrieved_docs,
                # Token Counts
                graph_total_prompt_tokens=conversation_flow.total_prompt_tokens,
                graph_total_completion_tokens=conversation_flow.total_completion_tokens,
                eval_total_prompt_tokens=eval_token_usage["prompt"],
                eval_total_completion_tokens=eval_token_usage["completion"],
                # The actual evaluation results
                evaluations=evaluations,
                # Overall metadata
                metadata={
                    "overall_success": overall_success,
                    "evaluation_token_usage": eval_token_usage,
                },
            )

        except Exception as e:
            self._error_count += 1
            logger.error(
                f"EVALUATOR_FAILURE: Evaluation failed for thread_id={conversation_flow.thread_id}",
                extra={
                    "thread_id": conversation_flow.thread_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            # Return a failure result with enhanced structure
            return EvaluationResult(
                # Identifiers - use defaults if conversation_flow is None
                run_id=conversation_flow.run_id if conversation_flow else "unknown",
                thread_id=conversation_flow.thread_id if conversation_flow else "unknown",
                turn_index=conversation_flow.turn_index if conversation_flow else 0,
                # Timestamps & Latency - use current time if conversation_flow is None
                timestamp_start=conversation_flow.start_time
                if conversation_flow
                else datetime.now(timezone.utc),
                timestamp_end=datetime.now(timezone.utc),
                graph_latency_ms=conversation_flow.latency_ms
                if conversation_flow
                else None,
                time_to_first_token_ms=conversation_flow.time_to_first_token_ms
                if conversation_flow
                else None,
                evaluation_latency_ms=None,  # Could not be calculated due to error
                total_latency_ms=None,  # Could not be calculated due to error
                # Core conversation data
                query=conversation_flow.user_query,
                response=conversation_flow.final_response,
                retrieved_docs=conversation_flow.node_executions[-1].retrieved_docs,
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
                metadata={
                    "overall_success": False,
                    "error": str(e),
                    "evaluation_token_usage": {"prompt": 0, "completion": 0},
                },
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


async def evaluate_response(
    node_execution: EnrichedNodeExecutionLog,
    user_query: str,
) -> Tuple[Dict[str, NodeEvaluation], bool]:
    """
    Evaluates a node's output using registered evaluators.

    Args:
        node_execution: The enriched node execution log containing input/output
        user_query: The original user query that triggered this execution

    Returns:
        Tuple of:
        - Dict mapping evaluator names to their evaluation results
        - Boolean indicating if all evaluations passed
    """
    node_name = node_execution.node_name
    evaluator_functions = EVALUATOR_REGISTRY.get(node_name, [])

    if not evaluator_functions:
        logger.warning(f"No evaluators registered for node: {node_name}")
        return {}, True

    node_eval_results = {}
    all_passed = True

    for evaluator_func in evaluator_functions:
        eval_name = evaluator_func.__name__
        try:
            result = await evaluator_func(node_execution, user_query)
            node_eval_results[eval_name] = result
            if not result.overall_success:
                all_passed = False

        except Exception as e:
            logger.error(
                f"Failed to run evaluator {eval_name}",
                extra={
                    "error": str(e),
                    "node_name": node_name,
                    "evaluator": eval_name,
                },
            )
            all_passed = False

    return node_eval_results, all_passed
