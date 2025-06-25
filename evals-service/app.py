from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Import evaluation modules
from models import EvaluationResult, ConversationFlow, ResponseMessage
from queue_manager import EvaluationQueueManager
from run_evals import AsyncEvaluator
from utils.logger import logger

# Load environment variables
project_root = Path(__file__).parent
env_path = project_root / ".env"
load_dotenv(env_path)

app = FastAPI(
    title="ChatRaghu Evaluation Service",
    description="FastAPI service for evaluating conversation flows and responses",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
evaluator = AsyncEvaluator()
queue_manager = EvaluationQueueManager()


@app.on_event("startup")
async def startup_event():
    """Initialize evaluation service on startup"""
    logger.info("Starting ChatRaghu Evaluation Service")
    await evaluator.start()
    await queue_manager.start(evaluator)
    logger.info("Evaluation service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down ChatRaghu Evaluation Service")
    await evaluator.stop()
    await queue_manager.stop()
    logger.info("Evaluation service shutdown complete")


class EvaluationRequest(BaseModel):
    """Request model for evaluation"""

    thread_id: str
    query: str
    response: str
    retrieved_docs: Optional[List[Dict[str, str]]] = None
    conversation_flow: ConversationFlow


class EvaluationResponse(BaseModel):
    """Response model for evaluation results"""

    thread_id: str
    success: bool
    evaluation_result: Optional[EvaluationResult] = None
    error: Optional[str] = None
    timestamp: datetime


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_conversation(
    request: EvaluationRequest, background_tasks: BackgroundTasks
):
    """
    Evaluate a conversation flow and response.

    This endpoint accepts a conversation evaluation request and processes it
    asynchronously. The evaluation results are stored and can be retrieved later.
    """
    try:
        logger.info(
            "Received evaluation request",
            extra={
                "thread_id": request.thread_id,
                "node_count": len(request.conversation_flow.node_executions),
            },
        )

        # Create response message for queue processing
        response_message = ResponseMessage(
            thread_id=request.thread_id,
            query=request.query,
            response=request.response,
            retrieved_docs=request.retrieved_docs,
            conversation_flow=request.conversation_flow,
        )

        # Add to background queue for processing
        background_tasks.add_task(queue_manager.enqueue_response, response_message)

        return EvaluationResponse(
            thread_id=request.thread_id,
            success=True,
            evaluation_result=None,
            error=None,
            timestamp=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(
            "Failed to process evaluation request",
            extra={"thread_id": request.thread_id, "error": str(e)},
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/sync", response_model=EvaluationResponse)
async def evaluate_conversation_sync(request: EvaluationRequest):
    """
    Evaluate a conversation flow and response synchronously.

    This endpoint processes the evaluation immediately and returns the results.
    Use this for testing or when immediate results are needed.
    """
    try:
        logger.info(
            "Received synchronous evaluation request",
            extra={
                "thread_id": request.thread_id,
                "node_count": len(request.conversation_flow.node_executions),
            },
        )

        # Perform evaluation synchronously
        evaluation_result = await evaluator.evaluate_response(
            thread_id=request.thread_id,
            query=request.query,
            response=request.response,
            retrieved_docs=request.retrieved_docs,
            conversation_flow=request.conversation_flow,
        )

        return EvaluationResponse(
            thread_id=request.thread_id,
            success=True,
            evaluation_result=evaluation_result,
            error=None,
            timestamp=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(
            "Failed to process synchronous evaluation request",
            extra={"thread_id": request.thread_id, "error": str(e)},
        )
        return EvaluationResponse(
            thread_id=request.thread_id,
            success=False,
            evaluation_result=None,
            error=str(e),
            timestamp=datetime.utcnow(),
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ChatRaghu Evaluation Service",
        "timestamp": datetime.utcnow(),
        "queue_size": queue_manager.queue.qsize() if queue_manager.queue else 0,
    }


@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    return {
        "queue_size": queue_manager.queue.qsize() if queue_manager.queue else 0,
        "worker_running": queue_manager._worker_task is not None
        and not queue_manager._worker_task.done(),
        "storage_path": evaluator.storage_path,
        "timestamp": datetime.utcnow(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
