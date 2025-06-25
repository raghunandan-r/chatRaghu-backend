import os
from pathlib import Path
from dotenv import load_dotenv
from utils.logger import logger
import asyncio
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import backoff
from models import EvaluationResult, RetryConfig, ConversationFlow
from evaluators import (
    evaluate_relevance_check,
    evaluate_query_or_respond,
    evaluate_few_shot_selector,
    evaluate_generate_with_context,
    evaluate_generate_with_persona,
)

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
    def __init__(self, storage_path: str = "./eval_results"):
        self.storage_path = storage_path
        self.retry_config = RetryConfig()
        self.results_queue = asyncio.Queue()
        self._setup_storage()

    def _setup_storage(self):
        """Setup local storage and S3 client if configured"""
        os.makedirs(self.storage_path, exist_ok=True)
        # TODO: Add S3 client setup when needed

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=30)
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

        Args:
            thread_id: Unique identifier for the conversation
            query: Original user query
            response: Final response generated
            retrieved_docs: Any documents retrieved during RAG
            conversation_flow: Complete flow of conversation through nodes
        """
        logger.info(
            "Starting conversation flow evaluation",
            extra={
                "thread_id": thread_id,
                "node_count": len(conversation_flow.node_executions)
                if conversation_flow
                else 0,
            },
        )

        evaluation_results = []

        try:
            if not conversation_flow or not conversation_flow.node_executions:
                raise ValueError("No conversation flow data provided for evaluation")

            # Process each node execution in order
            for node_execution in conversation_flow.node_executions:
                node_name = node_execution.node_name
                node_input = node_execution.input
                node_output = node_execution.output
                node_retrieved_docs = node_execution.retrieved_docs
                node_system_prompt = node_execution.system_prompt

                logger.info(
                    f"Evaluating {node_name} node",
                    extra={
                        "thread_id": thread_id,
                        "node_name": node_name,
                        "has_retrieved_docs": bool(node_retrieved_docs),
                        "has_system_prompt": bool(node_system_prompt),
                    },
                )

                if node_name == "relevance_check":
                    result = await evaluate_relevance_check(
                        user_query=query,
                        conversation_history=node_input.get("conversation_history", []),
                        model_output=node_output.get("messages", [{}])[0].get(
                            "content", ""
                        )
                        if node_output.get("messages")
                        else "",
                        rules=node_input.get("rules", ""),
                    )
                    evaluation_results.append(result)

                elif node_name == "query_or_respond":
                    result = await evaluate_query_or_respond(
                        query=query,
                        history=node_input.get("conversation_history", []),
                        model_output=node_output.get("messages", [{}])[0].get(
                            "content", ""
                        )
                        if node_output.get("messages")
                        else "",
                        rules=node_input.get("rules", ""),
                    )
                    evaluation_results.append(result)

                elif node_name == "few_shot_selector":
                    # Extract just the category from the multi-line response
                    full_response = (
                        node_output.get("messages", [{}])[0].get("content", "")
                        if node_output.get("messages")
                        else ""
                    )
                    category = extract_category_from_response(full_response)

                    logger.info(
                        "Extracted category from few_shot_selector",
                        extra={
                            "thread_id": thread_id,
                            "full_response": full_response[:100] + "..."
                            if len(full_response) > 100
                            else full_response,
                            "extracted_category": category,
                        },
                    )

                    result = await evaluate_few_shot_selector(
                        query=query,
                        category=category,  # Pass just the category
                        response_style=node_output.get("response_style", ""),
                        examples=node_input.get("examples", []),
                    )
                    evaluation_results.append(result)

                elif node_name == "generate_with_retrieved_context":
                    # Use node-specific retrieved docs if available, otherwise fall back to global
                    docs_to_use = (
                        node_retrieved_docs if node_retrieved_docs else retrieved_docs
                    )
                    docs_content = []
                    if docs_to_use:
                        for doc in docs_to_use:
                            if isinstance(doc, dict):
                                docs_content.append(doc.get("content", ""))
                            else:
                                docs_content.append(str(doc))

                    result = await evaluate_generate_with_context(
                        query=query,
                        response=node_output.get("messages", [{}])[0].get("content", "")
                        if node_output.get("messages")
                        else "",
                        docs_content=docs_content,
                    )
                    evaluation_results.append(result)

                elif node_name == "generate_with_persona":
                    # Find the last AI message from conversation history
                    last_ai_message = None
                    if conversation_flow.node_executions:
                        for prev_node in reversed(
                            conversation_flow.node_executions[:-1]
                        ):
                            if prev_node.output and "response" in prev_node.output:
                                last_ai_message = prev_node.output["response"]
                                break

                    result = await evaluate_generate_with_persona(
                        response=node_output.get("response", ""),
                        category=node_input.get(
                            "category", "OFFICIAL"
                        ),  # Default to OFFICIAL if not specified
                        last_ai_message=last_ai_message,
                        rules=node_input.get("rules", ""),
                    )
                    evaluation_results.append(result)

            # Calculate overall success and aggregate metrics
            overall_success = all(result.success for result in evaluation_results)
            aggregated_metrics = {}

            # Aggregate metrics from all evaluations
            for result in evaluation_results:
                for metric_name, metric_value in result.metrics.items():
                    qualified_metric_name = f"{result.node_name}_{metric_name}"
                    aggregated_metrics[qualified_metric_name] = metric_value

            # Create final evaluation result
            eval_result = EvaluationResult(
                thread_id=thread_id,
                timestamp=datetime.utcnow(),
                query=query,
                response=response,
                retrieved_docs=retrieved_docs,
                scores=aggregated_metrics,
                metadata={
                    "evaluation_version": "1.0",
                    "overall_success": overall_success,
                    "node_results": [
                        {
                            "node_name": result.node_name,
                            "success": result.success,
                            "explanation": result.explanation,
                        }
                        for result in evaluation_results
                    ],
                    "conversation_flow": conversation_flow.model_dump(mode="json")
                    if conversation_flow
                    else None,
                },
            )

            logger.info(
                "Completed conversation flow evaluation",
                extra={
                    "thread_id": thread_id,
                    "overall_success": overall_success,
                    "evaluated_nodes": [r.node_name for r in evaluation_results],
                },
            )

            return eval_result

        except Exception as e:
            logger.error(
                "Failed to evaluate conversation flow",
                extra={"thread_id": thread_id, "error": str(e)},
            )
            raise

    async def _store_results_worker(self):
        """Background worker to store evaluation results"""
        while True:
            try:
                # Collect results for batch processing
                results = []
                try:
                    while len(results) < 100:  # batch size
                        result = await asyncio.wait_for(
                            self.results_queue.get(), timeout=5.0
                        )
                        results.append(result)
                except asyncio.TimeoutError:
                    pass  # Process whatever we have

                if results:
                    # Convert to DataFrame
                    df = pd.DataFrame([r.model_dump(mode="json") for r in results])

                    # Save to parquet
                    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    filename = f"eval_results_{timestamp}.parquet"
                    df.to_parquet(
                        os.path.join(self.storage_path, filename), compression="snappy"
                    )

                    # TODO: Upload to S3 when configured

                    logger.info(
                        f"Stored {len(results)} evaluation results",
                        extra={"filename": filename, "record_count": len(results)},
                    )

            except Exception as e:
                logger.error("Error in results storage worker", extra={"error": str(e)})
                await asyncio.sleep(5)  # Back off on error

    async def start(self):
        """Start the background storage worker"""
        self._storage_task = asyncio.create_task(self._store_results_worker())

    async def stop(self):
        """Stop the background storage worker"""
        if hasattr(self, "_storage_task"):
            self._storage_task.cancel()
            try:
                await self._storage_task
            except asyncio.CancelledError:
                pass
