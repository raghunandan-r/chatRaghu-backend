#!/usr/bin/env python3
"""
Shared fixtures and configuration for ChatRaghu tests.
This file makes fixtures automatically available to all test files in the directory.
"""

import pytest
import pytest_asyncio
import httpx
import os
import logging
from datetime import datetime
from typing import Dict, Any, AsyncGenerator
import sys
from pathlib import Path
from dotenv import load_dotenv

try:
    import pytest_httpx
except ImportError:
    pytest_httpx = None


# Setup path and import models
def _setup_imports():
    """Setup imports with proper path handling"""
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # This import is crucial to set up the shims
    import evals_service

    # Now we can use the shimmed imports
    from models import ConversationFlow, EnrichedNodeExecutionLog

    return ConversationFlow, EnrichedNodeExecutionLog


ConversationFlow, EnrichedNodeExecutionLog = _setup_imports()

load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Test configuration
MAIN_SERVICE_URL = "http://localhost:3000"
EVALUATION_SERVICE_URL = "http://localhost:8001"
TEST_API_KEY = os.getenv("TEST_API_KEY", "test_api_key_123")


class ServiceTestError(Exception):
    """Base exception for service test failures"""

    pass


class ServiceUnavailableError(ServiceTestError):
    """Raised when a service is not available"""

    pass


class APIKeyError(ServiceTestError):
    """Raised when API key validation fails"""

    pass


@pytest_asyncio.fixture
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Async HTTP client fixture with proper error handling"""
    logger.info("Initializing HTTP client for testing")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info("HTTP client initialized successfully")
            yield client
    except Exception as e:
        logger.error(f"Failed to initialize HTTP client: {e}")
        raise ServiceTestError(f"HTTP client initialization failed: {e}")


@pytest.fixture
def valid_api_headers() -> Dict[str, str]:
    """Headers with valid API key"""
    logger.debug(f"Creating headers with API key: {TEST_API_KEY[:8]}...")
    return {"X-API-Key": TEST_API_KEY}


@pytest.fixture
def invalid_api_headers() -> Dict[str, str]:
    """Headers with invalid API key"""
    logger.debug("Creating headers with invalid API key")
    return {"X-API-Key": "invalid_key"}


@pytest.fixture
def sample_chat_request() -> Dict[str, Any]:
    """Sample chat request payload"""
    logger.debug("Creating sample chat request payload")
    return {
        "messages": [
            {
                "role": "user",
                "content": "Hello, this is a test message",
                "thread_id": "test-thread-456",
            }
        ]
    }


@pytest.fixture
def sample_evaluation_request() -> Dict[str, Any]:
    """Sample evaluation request payload"""
    logger.debug("Creating sample evaluation request payload")
    try:
        conversation_flow = ConversationFlow(
            thread_id="test-thread-123",
            user_query="What is the weather like?",
            node_executions=[
                EnrichedNodeExecutionLog(
                    node_name="relevance_check",
                    input={"conversation_history": [], "rules": "Test rules"},
                    output={"messages": [{"content": "RELEVANT"}]},
                    retrieved_docs=None,
                    system_prompt=None,
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    graph_version="v1",
                    tags=["classification"],
                    message_source="ai",
                ),
                EnrichedNodeExecutionLog(
                    node_name="generate_with_persona",
                    input={"category": "OFFICIAL", "rules": "Test rules"},
                    output={"response": "The weather is sunny today."},
                    retrieved_docs=None,
                    system_prompt="You are a helpful assistant.",
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    graph_version="v1",
                    tags=["system_prompt", "persona"],
                    message_source="ai",
                ),
            ],
        )

        return {
            "thread_id": "test-thread-123",
            "query": "What is the weather like?",
            "response": "The weather is sunny today.",
            "conversation_flow": conversation_flow.model_dump(mode="json"),
        }
    except Exception as e:
        logger.error(f"Failed to create sample evaluation request: {e}")
        raise ServiceTestError(f"Sample evaluation request creation failed: {e}")
