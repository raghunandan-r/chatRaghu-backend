import asyncio
from typing import Optional
from utils.logger import logger
from models import ConversationFlow
from config import config


class DualQueueManager:
    """
    Manages dual queues for audit logging and evaluation processing.
    Separates concerns between immediate audit logging and background evaluation.
    """

    def __init__(self):
        # Separate queues for different purposes
        self.audit_queue: asyncio.Queue = asyncio.Queue(
            maxsize=config.service.max_queue_size
        )
        self.eval_queue: asyncio.Queue = asyncio.Queue(
            maxsize=config.service.max_queue_size
        )

        # Storage managers (will be set by the app)
        self.audit_storage = None
        self.results_storage = None

        # Worker tasks
        self._audit_worker_task: Optional[asyncio.Task] = None
        self._eval_worker_task: Optional[asyncio.Task] = None

        # Metrics
        self._audit_processed = 0
        self._eval_processed = 0
        self._audit_errors = 0
        self._eval_errors = 0

        logger.info(
            "Initialized DualQueueManager",
            extra={
                "max_queue_size": config.service.max_queue_size,
                "queue_worker_count": config.service.queue_worker_count,
            },
        )

    def set_storage_managers(self, audit_storage, results_storage):
        """Set the storage managers for audit and evaluation results."""
        self.audit_storage = audit_storage
        self.results_storage = results_storage
        logger.info("Storage managers configured for DualQueueManager")

    async def enqueue_audit(self, request: ConversationFlow):
        """Enqueue a request for immediate audit logging."""
        try:
            await self.audit_queue.put(request)
            logger.info(
                "Enqueued request for audit logging",
                extra={
                    "thread_id": request.thread_id,
                    "audit_queue_size": self.audit_queue.qsize(),
                },
            )
        except asyncio.QueueFull:
            self._audit_errors += 1
            logger.error(
                "Audit queue is full, dropping request",
                extra={"thread_id": request.thread_id},
            )
            raise

    async def enqueue_evaluation(self, message: ConversationFlow):
        """Enqueue a response for evaluation processing."""
        try:
            await self.eval_queue.put(message)
            logger.info(
                "Enqueued response for evaluation",
                extra={
                    "thread_id": message.thread_id,
                    "eval_queue_size": self.eval_queue.qsize(),
                },
            )
        except asyncio.QueueFull:
            self._eval_errors += 1
            logger.error(
                "Evaluation queue is full, dropping request",
                extra={"thread_id": message.thread_id},
            )
            raise

    async def _audit_worker(self):
        """Worker for processing audit requests."""
        logger.info("Started audit queue worker")

        while True:
            request = None
            try:
                request = await self.audit_queue.get()
            except asyncio.CancelledError:
                logger.info("Audit worker cancelled")
                break

            try:
                await self._process_audit_request(request)
                self._audit_processed += 1
            except Exception as e:
                self._audit_errors += 1
                logger.error("Error in audit worker", extra={"error": str(e)})
            finally:
                if request is not None:
                    self.audit_queue.task_done()

    async def _eval_worker(self, evaluator):
        """Worker for processing evaluation requests."""
        logger.info("Started evaluation queue worker")

        while True:
            message = None
            try:
                message = await self.eval_queue.get()
            except asyncio.CancelledError:
                logger.info("Evaluation worker cancelled")
                break

            try:
                await self._process_evaluation_request(message, evaluator)
                self._eval_processed += 1

            except Exception as e:
                self._eval_errors += 1
                logger.error("Error in evaluation worker", extra={"error": str(e)})
            finally:
                if message is not None:
                    self.eval_queue.task_done()

    async def _process_audit_request(self, request: ConversationFlow):
        """Process an audit request and store it."""
        try:
            logger.info(
                "Processing audit request",
                extra={
                    "thread_id": request.thread_id,
                    "nodes_to_evaluate": request.node_executions,
                },
            )

            # Store the audit request if storage manager is available, picked up by StorageManager._storage_worker()
            if self.audit_storage:
                await self.audit_storage.queue.put(request)
                logger.info(
                    "Queued audit request for storage",
                    extra={"thread_id": request.thread_id},
                )
            else:
                logger.warning("No audit storage manager configured")

        except Exception as e:
            logger.error(
                "Failed to process audit request",
                extra={"thread_id": request.thread_id, "error": str(e)},
            )
            raise

    async def _process_evaluation_request(self, message: ConversationFlow, evaluator):
        """Process an evaluation request and store the result."""
        try:
            logger.info(
                "Processing evaluation request",
                extra={
                    "thread_id": message.thread_id,
                    "nodes_to_evaluate": message.node_executions,
                    "execution_path": [
                        f"{node.node_name}({node.output.get('next_edge', 'N/A')})"
                        for node in message.node_executions
                    ],
                },
            )

            # Perform the evaluation
            evaluation_result = await evaluator.evaluate_response(message)

            # Store the evaluation result if storage manager is available
            if self.results_storage:
                await self.results_storage.queue.put(evaluation_result)
                logger.info(
                    "Queued evaluation result for storage",
                    extra={
                        "thread_id": message.thread_id,
                        "overall_success": evaluation_result.metadata.get(
                            "overall_success", False
                        ),
                    },
                )
            else:
                logger.warning("No results storage manager configured")

            logger.info(
                "Successfully evaluated response",
                extra={"thread_id": message.thread_id},
            )
        except Exception as e:
            logger.error(
                "Evaluation failed after in queue_manager._process_evaluation_request",
                extra={
                    "error": str(e),
                    "message_data": message.to_dict(),
                },
            )

    async def start(self, evaluator):
        """Start both queue workers."""
        self._audit_worker_task = asyncio.create_task(self._audit_worker())
        self._eval_worker_task = asyncio.create_task(self._eval_worker(evaluator))
        logger.info("Started dual queue workers")

    async def stop(self):
        """Stop both queue workers."""
        if self._audit_worker_task:
            self._audit_worker_task.cancel()
            try:
                await self._audit_worker_task
            except asyncio.CancelledError:
                pass

        if self._eval_worker_task:
            self._eval_worker_task.cancel()
            try:
                await self._eval_worker_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped dual queue workers")

    async def health_check(self) -> dict:
        """Perform health check on queue system."""
        try:
            audit_worker_healthy = (
                self._audit_worker_task is not None
                and not self._audit_worker_task.done()
            )
            eval_worker_healthy = (
                self._eval_worker_task is not None and not self._eval_worker_task.done()
            )

            return {
                "queue_manager_healthy": audit_worker_healthy and eval_worker_healthy,
                "audit_worker_healthy": audit_worker_healthy,
                "eval_worker_healthy": eval_worker_healthy,
                "audit_queue_size": self.audit_queue.qsize(),
                "eval_queue_size": self.eval_queue.qsize(),
                "audit_processed": self._audit_processed,
                "eval_processed": self._eval_processed,
                "audit_errors": self._audit_errors,
                "eval_errors": self._eval_errors,
            }
        except Exception as e:
            logger.error("Queue manager health check failed", extra={"error": str(e)})
            return {"queue_manager_healthy": False, "error": str(e)}

    async def get_metrics(self) -> dict:
        """Get queue manager metrics."""
        return {
            "audit_queue_size": self.audit_queue.qsize(),
            "eval_queue_size": self.eval_queue.qsize(),
            "audit_processed": self._audit_processed,
            "eval_processed": self._eval_processed,
            "audit_errors": self._audit_errors,
            "eval_errors": self._eval_errors,
            "max_queue_size": config.service.max_queue_size,
        }


# Legacy compatibility - keep the old class name for backward compatibility
EvaluationQueueManager = DualQueueManager
