import asyncio
from typing import Optional
from utils.logger import logger
from models import ResponseMessage


class EvaluationQueueManager:
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    async def enqueue_response(self, message: ResponseMessage):
        await self.queue.put(message)
        logger.info(
            "Enqueued response for evaluation",
            extra={"thread_id": message.thread_id, "queue_size": self.queue.qsize()},
        )

    async def _process_message(self, message: ResponseMessage, evaluator):
        try:
            logger.info(
                "Processing conversation flow",
                extra={
                    "thread_id": message.thread_id,
                    "node_count": len(message.conversation_flow.node_executions),
                    "execution_path": [
                        f"{node.node_name}({node.output.get('next_edge', 'N/A')})"
                        for node in message.conversation_flow.node_executions
                    ],
                },
            )

            await evaluator.evaluate_response(
                thread_id=message.thread_id,
                query=message.query,
                response=message.response,
                retrieved_docs=message.retrieved_docs,
                conversation_flow=message.conversation_flow,
            )

            logger.info(
                "Successfully evaluated response",
                extra={"thread_id": message.thread_id},
            )
        except Exception as e:
            if message.retry_count < message.max_retries:
                message.retry_count += 1
                await self.queue.put(message)
                logger.warning(
                    f"Evaluation failed, retrying {message.retry_count}/{message.max_retries}",
                    extra={"thread_id": message.thread_id, "error": str(e)},
                )
            else:
                logger.error(
                    f"Evaluation failed after {message.max_retries} retries",
                    extra={
                        "error": str(e),
                        "message_data": message.model_dump(mode="json"),
                    },
                )

    async def _worker(self, evaluator):
        while True:
            try:
                message = await self.queue.get()
                await self._process_message(message, evaluator)
            except Exception as e:
                logger.error("Worker error", extra={"error": str(e)})
            finally:
                self.queue.task_done()

    async def start(self, evaluator):
        self._worker_task = asyncio.create_task(self._worker(evaluator))
        logger.info("Started evaluation queue worker")

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                logger.info("Evaluation queue worker stopped")
