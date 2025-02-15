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
            if not hasattr(stream_gen, '_is_closing'):
                setattr(stream_gen, '_is_closing', True)
                await asyncio.wait_for(stream_gen.aclose(), timeout=5)
        except asyncio.TimeoutError:
            logger.warning("Timeout while closing stream generator", extra={"thread_id": thread_id})
        except RuntimeError as e:
            if "already running" in str(e) or "didn't stop after athrow()" in str(e):
                logger.warning(f"Generator cleanup race condition: {str(e)}", extra={"thread_id": thread_id})
            else:
                raise
        except Exception as e:
            logger.warning(f"Error during stream generator cleanup: {str(e)}", extra={"thread_id": thread_id})

async def stream_chunks(
    stream_gen,
    thread_id: str,
    start_time: float
) -> AsyncGenerator[str, None]:
    """Handle streaming of chunks with proper error handling and timeout"""
    chunk_count = 0
    total_chars = 0
    last_beat = start_time

    try:
        async with manage_stream_generator(stream_gen, thread_id):
            async with asyncio.timeout(STREAM_TIMEOUT):
                while True:
                    try:
                        msg, metadata = await asyncio.wait_for(
                            asyncio.shield(stream_gen.__anext__()),
                            timeout=30
                        )
                        
                        current_time = time.time()
                        if current_time - start_time > STREAM_TIMEOUT:
                            logger.warning("Stream exceeded maximum duration", extra={
                                "thread_id": thread_id,
                                "duration": current_time - start_time
                            })
                            yield json.dumps({
                                "choices": [{
                                    "delta": {"content": "Stream exceeded maximum duration"}
                                }]
                            }) + "\n"
                            return
                            
                        chunk_count += 1
                        total_chars += len(str(msg))
                        
                        if isinstance(msg, graph.AIMessageChunk) and metadata.get('langgraph_node') == 'generate_with_persona':
                            yield json.dumps({
                                "choices": [{
                                    "delta": {"content": msg.content}
                                }]
                            }) + "\n"
                            last_beat = current_time
                            
                    except StopAsyncIteration:
                        logger.info("Stream completed normally", extra={"thread_id": thread_id})
                        yield json.dumps({"choices": [{"delta": {"content": "[DONE]"}}]}) + "\n"
                        return
                    except asyncio.TimeoutError:
                        logger.error("Stream timeout", extra={"thread_id": thread_id})
                        yield json.dumps({
                            "choices": [{
                                "delta": {"content": "Request timed out"}
                            }]
                        }) + "\n"
                        return
                    except asyncio.CancelledError:
                        logger.info("Stream cancelled by client", extra={
                            "thread_id": thread_id,
                            "duration": time.time() - start_time
                        })
                        return
                        
    except asyncio.TimeoutError:
        logger.warning("Global stream timeout reached", extra={
            "thread_id": thread_id,
            "duration": time.time() - start_time,
            "chunks_sent": chunk_count
        })
        yield json.dumps({
            "choices": [{
                "delta": {"content": "Stream exceeded maximum duration of 60 seconds"}
            }]
        }) + "\n"
        return
    except GeneratorExit:
        logger.info("Generator exit requested", extra={
            "thread_id": thread_id,
            "duration": time.time() - start_time
        })
        return
    except Exception as e:
        error_context = {
            "thread_id": thread_id,
            "error_type": type(e).__name__,
            "error_details": str(e),
            "chunks_sent": chunk_count,
            "total_duration": time.time() - start_time,
        }
        logger.error("Stream error", extra=error_context)
        with sentry_sdk.push_scope() as scope:
            for key, value in error_context.items():
                scope.set_extra(key, value)
            sentry_sdk.capture_exception(e)
        yield json.dumps({
            "choices": [{
                "delta": {"content": "An unexpected error occurred"}
            }]
        }) + "\n"
        return

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
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """Main chat endpoint handler"""
    thread_id = request.messages[0].thread_id if request.messages and request.messages[0].thread_id else ''
    
    try:
        # Rate limiting and cache check
        await check_thread_rate_limit(thread_id)
        cache_key = f"{request.messages[0].content}"
        is_cached = graph.llm.cache and await graph.llm.cache.lookup(cache_key) is not None
        
        # Initialize stream
        messages = {"messages": [graph.HumanMessage(content=request.messages[0].content)]}
        config = {"configurable": {"thread_id": thread_id}}
        stream_gen = graph.graph.astream(messages, stream_mode="messages", config=config)
        
        return StreamingResponse(
            stream_chunks(stream_gen, thread_id, time.time()),
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
        logger.exception("Error initializing chat stream")
        raise HTTPException(status_code=500, detail="Failed to initialize chat stream")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

application = app

if __name__ == "__main__":    

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8080,
        reload=True,
        log_level="info",
        timeout_keep_alive=20,
        timeout_graceful_shutdown=30
    )
        
