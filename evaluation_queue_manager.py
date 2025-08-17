"""
Simplified queue manager for the main ChatRaghu backend service.

This module provides a queue manager interface that communicates with
the separate evaluation service instead of handling evaluations locally.
"""

import asyncio
from typing import Optional
from utils.logger import logger
from evaluation_models import ConversationFlow
from evaluation_client import get_evaluation_client


class EvaluationQueueManager:
    """Simplified queue manager that delegates to evaluation service"""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._evaluation_client = None

    async def enqueue_response(self, message: ConversationFlow):
        """Enqueue a response for evaluation via the evaluation service"""
        await self.queue.put(message)
        logger.info(
            "Enqueued response for evaluation",
            extra={"thread_id": message.thread_id, "queue_size": self.queue.qsize()},
        )

    async def _process_message(self, message: ConversationFlow):
        """Process a message by sending it to the evaluation service"""
        try:
            if self._evaluation_client is None:
                self._evaluation_client = await get_evaluation_client()

            logger.info(
                "Sending conversation flow to evaluation service",
                extra={
                    "thread_id": message.thread_id,
                    "node_count": len(message.node_executions),
                },
            )

            # Send to evaluation service
            await self._evaluation_client.evaluate_conversation_async(message)

            logger.info(
                "Successfully sent evaluation request to service",
                extra={"thread_id": message.thread_id},
            )

        except Exception as e:
            logger.error(
                f"Evaluation service request failed, RETRYING!", # TODO: retry only if its an LLM API call error
                extra={"error": str(e), "thread_id": message.thread_id},
            )

    async def _worker(self):
        """Background worker to process evaluation requests"""
        while True:
            message = None
            try:
                # Wait for the next item; allow the task to be cancelled cleanly
                message = await self.queue.get()
            except asyncio.CancelledError:
                # Task is being cancelled (e.g., on shutdown). Leave the loop gracefully.
                logger.info("Evaluation queue worker received cancellation signal")
                break

            try:
                await self._process_message(message)
            except Exception as e:
                logger.error("Worker error", extra={"error": str(e)})
            finally:
                # Mark the task done **only** if we actually retrieved an item
                if message is not None:
                    self.queue.task_done()

    async def start(self, evaluator=None):
        """Start the evaluation queue worker"""
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("Started evaluation queue worker")

    async def stop(self):
        """Stop the evaluation queue worker"""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                logger.info("Evaluation queue worker stopped")
