import os
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Header,
    Request,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
from contextlib import asynccontextmanager
import uvicorn
from graph import (
    MessagesState,
    HumanMessage,
    get_evaluation_client,
    close_evaluation_client,
    EvaluationQueueManager,
    create_engine_default,
    create_engine_immi,
)
from utils.logger import logger

# import redis.asyncio as redis
from upstash_redis.asyncio import Redis
from graph.models import set_global_redis_client, SESSION_STORAGE_MODE
from openai import AsyncOpenAI
import instructor

# Load .env files only if they exist
if os.path.exists(".env"):
    load_dotenv(".env")
    load_dotenv(".env.development")

# Sentry initialization is now handled in the logger utility
from utils.logger import initialize_sentry

initialize_sentry()


class ChatRequest(BaseModel):
    content: str
    thread_id: str
    stream_id: str
    query_type: str


class ErrorResponse(BaseModel):
    error: str


KEEP_ALIVE_TIMEOUT = 15


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Redis and other services
    # app.state.redis = redis.from_url(os.getenv("UPSTASH_REDIS_REST_URL"))
    app.state.redis = Redis.from_env()
    app.state.evaluation_client = await get_evaluation_client()
    app.state.queue_manager = EvaluationQueueManager()

    # Set the global Redis client for session storage
    if SESSION_STORAGE_MODE == "redis":
        set_global_redis_client(app.state.redis)
    else:
        set_global_redis_client(None)
    instructor_client = instructor.from_openai(
        AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    )
    app.state.resume_engine = create_engine_default(
        instructor_client=instructor_client, queue_manager=app.state.queue_manager
    )
    app.state.immi_engine = create_engine_immi(
        instructor_client=instructor_client, queue_manager=app.state.queue_manager
    )

    await app.state.queue_manager.start()
    yield

    # Shutdown
    await app.state.redis.close()
    await app.state.queue_manager.stop()
    await close_evaluation_client()


# Initialize FastAPI
app = FastAPI(
    title="ChatRaghu Stream Generator",
    description="API for generating LLM streams and writing them to Redis",
    version="3.0.1",
    lifespan=lifespan,
)

# CORS configuration
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)

# Security: API Key for authenticating the proxy
PROXY_API_KEY = os.getenv("VALID_API_KEYS")


async def verify_proxy_key(
    x_api_key: str = Header(..., description="API key from frontend proxy")
):
    """Dependency to verify the proxy API key."""
    if not PROXY_API_KEY or x_api_key not in PROXY_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key")


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Connection"] = "keep-alive"
    response.headers["Keep-Alive"] = f"timeout={KEEP_ALIVE_TIMEOUT}, max=100"
    return response


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log incoming requests and outgoing responses."""
    if request.url.path == "/favicon.ico":
        return await call_next(request)

    logger.info(
        "Incoming request",
        extra={
            "method": request.method,
            "url": str(request.url),
        },
    )
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()

    logger.info(
        "Request completed",
        extra={
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "process_time": f"{process_time:.4f}s",
        },
    )
    return response


async def generate_and_persist_stream(
    engine, redis: Redis, thread_id: str, stream_id: str, initial_state: MessagesState
):
    """
    Executes the GraphEngine stream and writes each chunk to a Redis Stream.
    This function is designed to run in the background and be resilient to Redis failures.
    """
    stream_key = f"stream:{stream_id}"
    redis_is_functional = True  # Assume Redis is working initially

    try:
        run_id = f"run_{stream_id}"
        turn_index = (
            len(
                [msg for msg in initial_state.messages if isinstance(msg, HumanMessage)]
            )
            - 1
        )

        stream_gen = engine.execute_stream(initial_state, run_id, turn_index)

        async for chunk, meta in stream_gen:
            if chunk.type == "content" and chunk.content and redis_is_functional:
                try:
                    # This is the primary operation that can fail.
                    # await redis.xadd(stream_key, "*",{"chunk": str(chunk.content).encode()})
                    await redis.execute(
                        ["XADD", stream_key, "*", "chunk", str(chunk.content)]
                    )
                except Exception as redis_error:
                    redis_is_functional = False
                    logger.error(
                        "Redis connection failed during stream write. Halting stream persistence for this request.",
                        extra={"thread_id": thread_id, "error": str(redis_error)},
                    )
                    # Do not attempt any more Redis writes for this request.

    except Exception as e:
        logger.error(
            "Unhandled error during stream generation logic",
            extra={"thread_id": thread_id, "error": str(e)},
        )
        # If Redis was still functional before this broader error, try to write an error message.
        if redis_is_functional:
            try:
                error_message = f"[STREAM_GENERATION_ERROR: {e}]"
                # await redis.xadd(stream_key, "*", {"chunk": error_message.encode()})
                await redis.execute(["XADD", stream_key, "*", "chunk", error_message])
            except Exception as final_redis_error:
                logger.error(
                    "Redis failed while trying to write final error message.",
                    extra={"thread_id": thread_id, "error": str(final_redis_error)},
                )
    finally:
        # Only attempt to finalize the stream if Redis was working.
        if redis_is_functional:
            try:
                # Always write an end-of-stream marker so consumers know when to stop.
                # await redis.xadd(stream_key, "*", {"chunk": b"[END_OF_STREAM]"})
                await redis.execute(
                    ["XADD", stream_key, "*", "chunk", "[END_OF_STREAM]"]
                )
                # Set a 1-hour expiry on the stream to automatically clean up old data.
                # await redis.expire(stream_key, 3600)
                await redis.execute(["EXPIRE", stream_key, "3600"])
                logger.info(f"Finished and cleaned up stream for {thread_id}")
            except Exception as final_redis_error:
                logger.error(
                    "Redis failed during final cleanup.",
                    extra={"thread_id": thread_id, "error": str(final_redis_error)},
                )


@app.post(
    "/generate-stream",
    dependencies=[Depends(verify_proxy_key)],
    responses={
        200: {"description": "Confirmation that stream generation has started"},
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def start_stream_generation(
    request: Request,
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
):
    """
    Validates the request from the proxy, instantly returns a confirmation,
    and starts the GraphEngine stream generation in a background task.
    """
    if chat_request.query_type == "immi":
        engine = request.app.state.immi_engine
    else:
        engine = request.app.state.resume_engine
    redis = request.app.state.redis

    try:
        new_message = HumanMessage(content=chat_request.content)
        initial_state = await MessagesState.from_thread(
            chat_request.thread_id, new_message
        )

        # Schedule the long-running stream generation in the background
        background_tasks.add_task(
            generate_and_persist_stream,
            engine,
            redis,
            chat_request.thread_id,
            chat_request.stream_id,
            initial_state,
        )

        # Return an immediate confirmation to the proxy
        return JSONResponse(
            content={
                "status": "generation_started",
                "thread_id": chat_request.thread_id,
            },
            status_code=200,
        )
    except Exception as e:
        logger.error(
            "Failed to start stream generation",
            extra={"thread_id": chat_request.thread_id, "error": str(e)},
        )
        raise HTTPException(
            status_code=500, detail="Internal server error starting stream generation"
        )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


application = app

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=3000,
        timeout_keep_alive=120,
        timeout_graceful_shutdown=30,
        log_level="debug",
        access_log=True,
    )
