from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import asyncio

# Import evaluation modules
from models import (
    EvaluationRequest,
    ResponseMessage,
)
from queue_manager import DualQueueManager
from run_evals import AsyncEvaluator
from storage import StorageManager, LocalStorageBackend, GCSStorageBackend
from utils.logger import logger
from config import config

# Load environment variables
project_root = Path(__file__).parent
env_path = project_root / ".env"
load_dotenv(env_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""

    # Initialize components
    app.state.evaluator = AsyncEvaluator()
    app.state.queue_manager = DualQueueManager()

    # Create storage backends
    # Determine storage backend based on configuration (local vs GCS)
    if (
        config.storage.storage_backend.lower() == "gcs"
        and config.storage.gcs_audit_bucket_name
        and config.storage.gcs_eval_results_bucket_name
    ):
        logger.info(
            "Using separate GCS buckets for audit and result storage",
            extra={
                "audit_bucket": config.storage.gcs_audit_bucket_name,
                "results_bucket": config.storage.gcs_eval_results_bucket_name,
            },
        )

        # Create separate backend instances for true isolation
        audit_backend = GCSStorageBackend(
            bucket_name=config.storage.gcs_audit_bucket_name
        )
        results_backend = GCSStorageBackend(
            bucket_name=config.storage.gcs_eval_results_bucket_name
        )
    else:
        # Fallback to local storage (default behaviour)
        logger.info(
            "Using LocalStorageBackend for audit and result storage",
            extra={"backend_type": "LocalStorageBackend"},
        )

    audit_backend = LocalStorageBackend(config.storage.audit_data_path)
    results_backend = LocalStorageBackend(config.storage.eval_results_path)

    # Create separate storage queues
    audit_storage_queue = asyncio.Queue(maxsize=config.service.max_queue_size)
    results_storage_queue = asyncio.Queue(maxsize=config.service.max_queue_size)

    # Create storage managers with separate queues
    app.state.audit_storage = StorageManager(
        queue=audit_storage_queue,
        storage_backend=audit_backend,
        file_prefix="audit_request",
        path_prefix="audit_data",
        batch_size=config.storage.batch_size,
        write_timeout=config.storage.write_timeout_seconds,
    )

    app.state.results_storage = StorageManager(
        queue=results_storage_queue,
        storage_backend=results_backend,
        file_prefix="eval_result",
        path_prefix="eval_results",
        batch_size=config.storage.batch_size,
        write_timeout=config.storage.write_timeout_seconds,
    )

    # Connect storage managers to queue manager
    app.state.queue_manager.set_storage_managers(
        app.state.audit_storage, app.state.results_storage
    )

    # Start all components
    await app.state.audit_storage.start()
    await app.state.results_storage.start()
    await app.state.queue_manager.start(app.state.evaluator)

    logger.info(
        "Evaluation service started successfully",
        extra={
            "service_name": config.service.service_name,
            "version": config.service.service_version,
            "environment": config.service.environment,
        },
    )

    yield

    # Shutdown all components
    logger.info("Shutting down evaluation service...")
    await app.state.queue_manager.stop()
    await app.state.results_storage.stop()
    await app.state.audit_storage.stop()
    logger.info("Evaluation service shutdown complete")


app = FastAPI(
    title="ChatRaghu Evaluation Service",
    description="FastAPI service for evaluating conversation flows and responses",
    version=config.service.service_version,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=config.cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EvaluationResponse(BaseModel):
    """Response model for evaluation results"""

    thread_id: str
    success: bool
    message: str
    timestamp: datetime


async def run_and_queue_evaluation(
    request: EvaluationRequest,
    queue_manager: DualQueueManager,
):
    """Wrapper function to run evaluation and queue the result."""
    logger.info(
        f"EVAL_APP_LOG: Starting background task for thread_id={request.thread_id}",
        extra={"thread_id": request.thread_id},
    )
    try:
        # Create response message for evaluation
        response_message = ResponseMessage(
            thread_id=request.thread_id,
            query=request.query,
            response=request.response,
            retrieved_docs=request.retrieved_docs,
            conversation_flow=request.conversation_flow,
        )

        # Queue for evaluation processing
        logger.info(
            f"EVAL_APP_LOG: Enqueueing for evaluation processing for thread_id={request.thread_id}",
            extra={"thread_id": request.thread_id},
        )
        await queue_manager.enqueue_evaluation(response_message)
        logger.info(
            f"EVAL_APP_LOG: Successfully enqueued for evaluation for thread_id={request.thread_id}",
            extra={"thread_id": request.thread_id},
        )

    except Exception as e:
        logger.error(
            "Failed during background evaluation setup",
            extra={"thread_id": request.thread_id, "error": str(e)},
        )


@app.post("/evaluate", response_model=EvaluationResponse, status_code=202)
async def evaluate_conversation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
):
    """
    Evaluate a conversation flow and response asynchronously.

    This endpoint accepts a conversation evaluation request, immediately logs it for audit,
    and schedules the evaluation to run in the background. Returns immediately with a 202 status.
    """
    try:
        logger.info(
            f"EVAL_APP_LOG: Received evaluation request for thread_id={request.thread_id}",
            extra={
                "thread_id": request.thread_id,
                "node_count": len(request.conversation_flow.node_executions),
            },
        )

        # 1. Immediately queue the raw request for audit logging
        logger.info(
            f"EVAL_APP_LOG: Enqueueing for audit logging for thread_id={request.thread_id}",
            extra={"thread_id": request.thread_id},
        )
        await app.state.queue_manager.enqueue_audit(request)
        logger.info(
            f"EVAL_APP_LOG: Successfully enqueued for audit for thread_id={request.thread_id}",
            extra={"thread_id": request.thread_id},
        )

        # 2. Schedule the evaluation to run in the background
        background_tasks.add_task(
            run_and_queue_evaluation, request, app.state.queue_manager
        )

        logger.info(
            f"EVAL_APP_LOG: Background task scheduled for thread_id={request.thread_id}",
            extra={"thread_id": request.thread_id},
        )
        return EvaluationResponse(
            thread_id=request.thread_id,
            success=True,
            message="Evaluation request accepted and is being processed.",
            timestamp=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(
            f"EVAL_APP_LOG: Failed to process evaluation request for thread_id={request.thread_id}",
            extra={"thread_id": request.thread_id, "error": str(e)},
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Collect health status from all components
        evaluator_health = await app.state.evaluator.health_check()
        queue_health = await app.state.queue_manager.health_check()
        audit_storage_health = await app.state.audit_storage.health_check()
        results_storage_health = await app.state.results_storage.health_check()

        # Determine overall health
        overall_healthy = all(
            [
                evaluator_health.get("evaluator_healthy", False),
                queue_health.get("queue_manager_healthy", False),
                audit_storage_health.get("storage_manager_healthy", False),
                results_storage_health.get("storage_manager_healthy", False),
            ]
        )

        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "service": config.service.service_name,
            "version": config.service.service_version,
            "environment": config.service.environment,
            "timestamp": datetime.utcnow(),
            "components": {
                "evaluator": evaluator_health,
                "queue_manager": queue_health,
                "audit_storage": audit_storage_health,
                "results_storage": results_storage_health,
            },
            "overall_healthy": overall_healthy,
        }
    except Exception as e:
        logger.error("Health check failed", extra={"error": str(e)})
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.utcnow()}


@app.get("/metrics")
async def get_metrics():
    """Get comprehensive service metrics"""
    try:
        evaluator_metrics = await app.state.evaluator.get_metrics()
        queue_metrics = await app.state.queue_manager.get_metrics()
        audit_storage_metrics = await app.state.audit_storage.get_metrics()
        results_storage_metrics = await app.state.results_storage.get_metrics()

        return {
            "service": {
                "name": config.service.service_name,
                "version": config.service.service_version,
                "environment": config.service.environment,
                "timestamp": datetime.utcnow(),
            },
            "components": {
                "evaluator": evaluator_metrics,
                "queue_manager": queue_metrics,
                "audit_storage": audit_storage_metrics,
                "results_storage": results_storage_metrics,
            },
            "configuration": {
                "batch_size": config.storage.batch_size,
                "write_timeout": config.storage.write_timeout_seconds,
                "max_queue_size": config.service.max_queue_size,
                "llm_model": config.llm.openai_model,
            },
        }
    except Exception as e:
        logger.error("Failed to get metrics", extra={"error": str(e)})
        return {"error": str(e), "timestamp": datetime.utcnow()}


@app.get("/config")
async def get_config():
    """Get current service configuration (without sensitive data)"""
    return {
        "service": {
            "name": config.service.service_name,
            "version": config.service.service_version,
            "environment": config.service.environment,
            "max_retry_attempts": config.service.max_retry_attempts,
            "max_queue_size": config.service.max_queue_size,
        },
        "storage": {
            "backend": config.storage.storage_backend,
            "audit_data_path": config.storage.audit_data_path,
            "eval_results_path": config.storage.eval_results_path,
            "batch_size": config.storage.batch_size,
            "write_timeout": config.storage.write_timeout_seconds,
        },
        "llm": {
            "model": config.llm.openai_model,
            "max_retries": config.llm.openai_max_retries,
            "timeout": config.llm.openai_timeout_seconds,
        },
        "api": {"host": config.api_host, "port": config.api_port},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host=config.api_host, port=config.api_port, workers=config.api_workers
    )
