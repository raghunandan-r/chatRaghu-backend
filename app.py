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
from typing import List, Dict, Optional, Type, Tuple
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from asyncio import TimeoutError
from functools import wraps

import graph.graph as graph
from graph.graph import warm_up_cache  # Import the warm_up function
from utils.logger import logger, log_request_info

# load_dotenv('.env')
# load_dotenv('.env.development')

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    cleanup_thread = Thread(target=Storage.cleanup_old_entries, daemon=True)
    cleanup_thread.start()        
    # print("Warming up LLM cache...")
    # await warm_up_cache()
     
    yield
    # Shutdown (if needed)

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
    response.headers["Keep-Alive"] = "timeout=75, max=100"
    return response

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
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

# Routes
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
    thread_id = request.messages[0].thread_id if request.messages and request.messages[0].thread_id else ''
    
    logger.info("Chat request received", extra={
        "thread_id": thread_id,
        "message_length": len(request.messages[0].content)
    })
    
    await check_thread_rate_limit(thread_id)    
    cache_key = f"{request.messages[0].content}"
    is_cached = graph.llm.cache and await graph.llm.cache.lookup(cache_key) is not None

    async def event_stream():
        try:
            messages = {"messages": [graph.HumanMessage(content=request.messages[0].content)]}
            config = {"configurable": {"thread_id": thread_id}}
            
            logger.debug("Starting graph stream", extra={
                "thread_id": thread_id,
                "is_cached": is_cached
            })
            
            async for msg, metadata in graph.graph.astream(
                messages,
                stream_mode="messages",
                config=config,
            ):
                if isinstance(msg, graph.AIMessageChunk) and metadata['langgraph_node'] == 'generate_with_persona':
                    logger.debug("Streaming chunk", extra={
                        "thread_id": thread_id,
                        "node": metadata['langgraph_node'],
                        "chunk_length": len(msg.content)
                    })
                    content = msg.content
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': content}}]})}\n\n"
            
            logger.info("Stream completed successfully", extra={"thread_id": thread_id})
            yield "data: [DONE]\n\n"

        except asyncio.TimeoutError:
            logger.error("Stream timeout", extra={"thread_id": thread_id})
            raise HTTPException(status_code=499, detail="Stream timed out")
        except asyncio.CancelledError:
            logger.warning("Stream cancelled by client", extra={"thread_id": thread_id})
            raise HTTPException(status_code=499, detail="Stream was cancelled by client")
        except Exception as e:
            logger.exception(
                "Stream error",
                extra={
                    "thread_id": thread_id,
                    "error_type": e.__class__.__name__,
                    "error_line": e.__traceback__.tb_lineno
                }
            )
            raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Vercel-AI-Data-Stream": "v1",
            "X-Cache-Status": "HIT" if is_cached else "MISS",  # Add cache status header
            "Access-Control-Allow-Headers": "*"
        }
    )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "main:app",
#         host="127.0.0.1",
#         port=8080,
#         reload=True,
#         log_level="info"
#     )

