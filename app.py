import asyncio  
import json
import os
from dotenv import load_dotenv
import re
from threading import Thread
from fastapi import FastAPI, HTTPException, Depends, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Type, Tuple, AsyncGenerator
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import time
import graph.graph as graph
# from graph.graph import warm_up_cache  # Import the warm_up function
from utils.logger import logger, log_request_info
from langsmith import Client
# from evals.evaluators import RaghuPersonaEvaluator, RelevanceEvaluator
import uvicorn
import sys
import sentry_sdk
from sentry_sdk import capture_exception, capture_message
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from starlette.types import Message

# Load .env files only if they exist
if os.path.exists('.env'):
    load_dotenv('.env')
    load_dotenv('.env.development')

if os.getenv("ENVIRONMENT") == "prod":
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),    
        environment="production",  
        traces_sample_rate=1.0,   
        profiles_sample_rate=1.0, 
        integrations=[
            FastApiIntegration(),
            AsyncioIntegration(),
        ],
        send_default_pii=True,
        _experiments={
            "continuous_profiling_auto_start": True,        
        },
    )


class ClientMessage(BaseModel):
    role: str
    content: str
    thread_id: Optional[str] = None

class ChatRequest(BaseModel):
    messages: List[ClientMessage]  # Accept last messages

    @field_validator('messages')
    def validate_messages(cls, v):
        # Check for potentially harmful content
        if not v:
            raise ValueError("At least one msg content is required")
        if v and re.search(r'<[^>]*script', v[-1].content, re.IGNORECASE):
            raise ValueError("Invalid message content")
        return v

class ChatResponse(BaseModel):
    response: str

class ErrorResponse(BaseModel):
    error: str

# Global storage (consider using Redis for production)
class Storage:
    api_key_usage: Dict[str, List[datetime]] = {}
    request_history: Dict[str, List[datetime]] = {}

    @classmethod
    async def cleanup_old_entries(cls):
        now = datetime.now()
        cutoff = now - timedelta(seconds=600)
        
        # Cleanup api_key_usage
        for api_key in list(cls.api_key_usage.keys()):
            cls.api_key_usage[api_key] = [
                timestamp for timestamp in cls.api_key_usage[api_key]
                if timestamp > cutoff
            ]
            if not cls.api_key_usage[api_key]:
                del cls.api_key_usage[api_key]

        # Cleanup request_history
        for thread_id in list(cls.request_history.keys()):
            cls.request_history[thread_id] = [
                timestamp for timestamp in cls.request_history[thread_id]
                if timestamp > cutoff
            ]
            if not cls.request_history[thread_id]:
                del cls.request_history[thread_id]

# Update these constants at the top of your file
STREAM_TIMEOUT = 60  # Global 60-second timeout for the entire stream
KEEP_ALIVE_TIMEOUT = 15

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    # Startup
    cleanup_task = asyncio.create_task(Storage.cleanup_old_entries())  # Schedule the cleanup task
    # print("Warming up LLM cache...")
    # await warm_up_cache()
     
    yield
    # Shutdown (if needed)
    cleanup_task.cancel()  # Cancel the cleanup task if the app is shutting down
    try:
        await cleanup_task  # Await the task to ensure it finishes
    except asyncio.CancelledError:
        logger.warning("Cleanup task was cancelled")

# Initialize FastAPI
app = FastAPI(
    title="ChatRaghu API",
    description="API for querying documents and returning LLM-formatted outputs",
    version="1.1.0",
    lifespan=lifespan
)

# Security and rate limiting constants
MAX_API_REQUESTS_PER_MINUTE = 100
MAX_USER_REQUESTS_PER_MINUTE = 30
VALID_API_KEY = set(os.environ.get("VALID_API_KEYS", '').split(','))

# CORS configuration
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["*"],
    expose_headers=["X-Rate-Limit", "Content-Type", "X-Vercel-AI-Data-Stream"],
    max_age=600,
)

# Dependencies
async def verify_api_key(
    x_api_key: str = Header(..., description="API key for authentication")
) -> str:
    if x_api_key not in VALID_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    now = datetime.now()
    if x_api_key in Storage.api_key_usage:
        Storage.api_key_usage[x_api_key] = [
            timestamp for timestamp in Storage.api_key_usage[x_api_key]
            if (now - timestamp).seconds < 60
        ]
        if len(Storage.api_key_usage[x_api_key]) >= MAX_API_REQUESTS_PER_MINUTE:
            raise HTTPException(
                status_code=429,
                detail="API rate limit exceeded"
            )
    
    Storage.api_key_usage[x_api_key] = Storage.api_key_usage.get(x_api_key, []) + [now]
    return x_api_key


async def check_thread_rate_limit(thread_id: str) -> bool:
    now = datetime.now()
    if thread_id in Storage.request_history:
        Storage.request_history[thread_id] = [
            timestamp for timestamp in Storage.request_history[thread_id]
            if (now - timestamp).seconds < 60
        ]
        if len(Storage.request_history[thread_id]) >= MAX_USER_REQUESTS_PER_MINUTE:
            raise HTTPException(
                status_code=429,
                detail="Thread rate limit exceeded"
            )
    
    Storage.request_history[thread_id] = Storage.request_history.get(thread_id, []) + [now]
    return True


# Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Connection"] = "keep-alive"
    response.headers["Keep-Alive"] = f"timeout={KEEP_ALIVE_TIMEOUT}, max=100"
    return response

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    if request.url.path == "/favicon.ico":
        return await call_next(request)

    # Log request
    request_info = await log_request_info(request)
    logger.info("Incoming request", extra=request_info)
    
    # Process request and measure timing
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    
    # Log response
    logger.info("Request completed", extra={
        **request_info,
        "status_code": response.status_code,
        "process_time": process_time
    })
    
    return response

@asynccontextmanager
async def manage_stream_generator(stream_gen, thread_id: str):
    """Context manager to properly handle stream generator lifecycle"""
    try:
        yield stream_gen
    finally:
        try:
            # Don't rely on _is_closing attribute
            if hasattr(stream_gen, 'aclose'):
                try:
                    await asyncio.wait_for(stream_gen.aclose(), timeout=5)
                except asyncio.TimeoutError:
                    logger.warning("Timeout during generator cleanup", extra={"thread_id": thread_id})
                except Exception as e:
                    logger.warning(f"Error during generator cleanup: {str(e)}", extra={"thread_id": thread_id})
        except Exception as e:
            logger.warning(f"Final cleanup error: {str(e)}", extra={"thread_id": thread_id})

async def stream_chunks(stream_gen, thread_id: str, start_time: float, request: Request):
    chunk_count = 0
    last_chunk_time = start_time
    
    try:
        logger.info("[STREAM_DEBUG] Stream starting", extra={
            "thread_id": thread_id,
            "start_time": start_time,
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown")
        })
        
        async with manage_stream_generator(stream_gen, thread_id):
            async for msg, metadata in stream_gen:
                current_time = time.time()
                chunk_count += 1
                chunk_delay = current_time - last_chunk_time
                
                logger.info("[STREAM_DEBUG] Processing chunk", extra={
                    "thread_id": thread_id,
                    "chunk_number": chunk_count,
                    "elapsed_time": current_time - start_time,
                    "chunk_delay": chunk_delay,
                    "langgraph_node": metadata.get('langgraph_node', 'unknown'),
                    "msg_type": type(msg).__name__,
                    "msg_content_length": len(msg.content) if hasattr(msg, 'content') else 0
                })
                
                if isinstance(msg, graph.AIMessageChunk) and metadata.get('langgraph_node') == 'generate_with_persona':
                    try:
                        yield json.dumps({
                            "choices": [{
                                "delta": {"content": msg.content}
                            }]
                        }) + "\n"
                        last_chunk_time = current_time
                        
                    except Exception as e:
                        logger.error("[STREAM_DEBUG] Chunk yield error", extra={
                            "thread_id": thread_id,
                            "chunk_number": chunk_count,
                            "elapsed_time": current_time - start_time,
                            "error": str(e),
                            "error_type": type(e).__name__
                        })
                        raise
                        
    except GeneratorExit as e:
        logger.error("[STREAM_DEBUG] Generator exit", extra={
            "thread_id": thread_id,
            "elapsed_time": time.time() - start_time,
            "chunks_processed": chunk_count,
            "last_chunk_age": time.time() - last_chunk_time,
            "traceback": {
                "file": e.__traceback__.tb_frame.f_code.co_filename,
                "function": e.__traceback__.tb_frame.f_code.co_name,
                "line": e.__traceback__.tb_lineno
            }
        })
        return
    except Exception as e:
        logger.error("[STREAM_DEBUG] Stream error", extra={
            "thread_id": thread_id,
            "elapsed_time": time.time() - start_time,
            "chunks_processed": chunk_count,
            "last_chunk_age": time.time() - last_chunk_time,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": {
                "file": e.__traceback__.tb_frame.f_code.co_filename,
                "function": e.__traceback__.tb_frame.f_code.co_name,
                "line": e.__traceback__.tb_lineno
            }
        })
        raise

@app.post(
    "/api/chat",
    response_model=ChatResponse,
    responses={
        200: {"model": ChatResponse},
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)

async def chat(
    request: Request,
    chat_request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """Main chat endpoint handler"""
    thread_id = chat_request.messages[0].thread_id if chat_request.messages else ''
    
    try:
        # Rate limiting and cache check
        await check_thread_rate_limit(thread_id)
        cache_key = f"{chat_request.messages[0].content}"
        is_cached = graph.llm.cache and await graph.llm.cache.lookup(cache_key) is not None
        
        # Initialize stream
        messages = {"messages": [graph.HumanMessage(content=chat_request.messages[0].content)]}
        config = {"configurable": {"thread_id": thread_id}}
        stream_gen = graph.graph.astream(messages, stream_mode="messages", config=config)
        
        return StreamingResponse(
            stream_chunks(stream_gen, thread_id, time.time(), request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Keep-Alive": f"timeout={KEEP_ALIVE_TIMEOUT}, max=100",
                "X-Vercel-AI-Data-Stream": "v1",
                "X-Cache-Status": "HIT" if is_cached else "MISS",
                "Access-Control-Allow-Headers": "*"
            }
        )
    except Exception as e:
        logger.error("[STREAM_DEBUG] Chat endpoint error", extra={
            "thread_id": thread_id,
            "error": str(e),
            "error_type": type(e).__name__
        })
        raise

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

class ASGILoggingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":  # Skip WebSocket etc.
            return await self.app(scope, receive, send)

        thread_id = None  # We'll set this when we get the request

        async def logged_receive():
            msg = await receive()
            logger.info("[ASGI_DEBUG] Received message", extra={
                "thread_id": thread_id,
                "message_type": msg["type"],
                "asgi_msg": msg
            })
            return msg

        async def logged_send(message: Message):
            logger.info("[ASGI_DEBUG] Sending message", extra={
                "thread_id": thread_id,
                "message_type": message["type"],
                "more_body": message.get("more_body", False)
            })
            await send(message)

        try:
            logger.info("[ASGI_DEBUG] Starting request", extra={
                "scope_type": scope["type"],
                "path": scope.get("path", "unknown")
            })
            await self.app(scope, logged_receive, logged_send)
        except Exception as e:
            logger.error("[ASGI_DEBUG] ASGI error", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise

# Add the middleware to FastAPI
app.add_middleware(ASGILoggingMiddleware)

application = app

if __name__ == "__main__":    
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8080,
        timeout_keep_alive=120,
        timeout_graceful_shutdown=30,
        log_level="debug",
        access_log=True
    )
        
