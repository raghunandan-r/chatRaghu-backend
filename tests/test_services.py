#!/usr/bin/env python3
"""
Integration tests for ChatRaghu services.
Tests both main service and evaluation service functionality with comprehensive coverage.
"""

import pytest
import pytest_asyncio
import httpx
import asyncio
import os
import logging
from datetime import datetime
from typing import Dict, Any, AsyncGenerator
from evaluation_models import ConversationFlow, EnrichedNodeExecutionLog
from unittest.mock import patch

try:
    import pytest_httpx
except ImportError:
    pytest_httpx = None

# Load environment variables from .env file
from dotenv import load_dotenv

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
    """Custom exception for service test failures"""

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


class TestMainService:
    """Test suite for main ChatRaghu backend service"""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, http_client: httpx.AsyncClient):
        """Test main service health endpoint with comprehensive validation"""
        logger.info("Testing main service health endpoint")

        try:
            response = await http_client.get(f"{MAIN_SERVICE_URL}/health")

            # Validate response structure and content
            assert (
                response.status_code == 200
            ), f"Expected 200, got {response.status_code}"

            response_data = response.json()
            assert "status" in response_data, "Health response missing 'status' field"
            assert (
                response_data["status"] == "healthy"
            ), f"Expected 'healthy', got '{response_data['status']}'"

            logger.info("✓ Main service health check passed")

        except httpx.ConnectError as e:
            logger.error(f"Connection error to main service: {e}")
            raise ServiceUnavailableError(f"Main service not available: {e}")
        except httpx.TimeoutException as e:
            logger.error(f"Timeout connecting to main service: {e}")
            raise ServiceTestError(f"Main service timeout: {e}")
        except Exception as e:
            logger.error(f"Unexpected error testing main service health: {e}")
            raise ServiceTestError(f"Main service health test failed: {e}")

    @pytest.mark.asyncio
    async def test_chat_endpoint_with_valid_api_key(
        self,
        http_client: httpx.AsyncClient,
        valid_api_headers: Dict[str, str],
        sample_chat_request: Dict[str, Any],
    ):
        """Test chat endpoint with valid API key and streaming response validation"""
        logger.info("Testing chat endpoint with valid API key")

        try:
            response = await http_client.post(
                f"{MAIN_SERVICE_URL}/api/chat",
                headers=valid_api_headers,
                json=sample_chat_request,
            )

            # Validate response status and headers
            assert (
                response.status_code == 200
            ), f"Expected 200, got {response.status_code}"
            assert response.headers["content-type"].startswith(
                "text/event-stream"
            ), "Expected streaming response"

            # For streaming responses, read content properly without closing connection
            content = ""
            async for chunk in response.aiter_text():
                content += chunk

            assert len(content) > 0, "Response should not be empty"

            # Log response details for debugging
            logger.info(
                f"✓ Chat endpoint test passed - response length: {len(content)} chars"
            )

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error in chat endpoint test: {e.response.status_code} - {e.response.text}"
            )
            if e.response.status_code == 401:
                raise APIKeyError("API key validation failed")
            else:
                raise ServiceTestError(f"Chat endpoint HTTP error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in chat endpoint test: {e}")
            raise ServiceTestError(f"Chat endpoint test failed: {e}")

    @pytest.mark.asyncio
    async def test_chat_endpoint_without_api_key(
        self, http_client: httpx.AsyncClient, sample_chat_request: Dict[str, Any]
    ):
        """Test chat endpoint without API key - should fail validation"""
        logger.info("Testing chat endpoint without API key")

        try:
            response = await http_client.post(
                f"{MAIN_SERVICE_URL}/api/chat", json=sample_chat_request
            )

            # Should fail with 422 (missing required header)
            assert (
                response.status_code == 422
            ), f"Expected 422 (missing header), got {response.status_code}"

            logger.info("✓ API key validation test passed")

        except Exception as e:
            logger.error(f"Error testing chat endpoint without API key: {e}")
            raise ServiceTestError(f"API key validation test failed: {e}")

    @pytest.mark.asyncio
    async def test_chat_endpoint_with_invalid_api_key(
        self,
        http_client: httpx.AsyncClient,
        invalid_api_headers: Dict[str, str],
        sample_chat_request: Dict[str, Any],
    ):
        """Test chat endpoint with invalid API key - should fail authentication"""
        logger.info("Testing chat endpoint with invalid API key")

        try:
            response = await http_client.post(
                f"{MAIN_SERVICE_URL}/api/chat",
                headers=invalid_api_headers,
                json=sample_chat_request,
            )

            # Should fail with 401 (unauthorized)
            assert (
                response.status_code == 401
            ), f"Expected 401, got {response.status_code}"

            response_data = response.json()
            assert "detail" in response_data, "Error response missing 'detail' field"
            assert (
                "Invalid API key" in response_data["detail"]
            ), "Expected 'Invalid API key' in error message"

            logger.info("✓ Invalid API key test passed")

        except Exception as e:
            logger.error(f"Error testing chat endpoint with invalid API key: {e}")
            raise ServiceTestError(f"Invalid API key test failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.rate_limiting
    async def test_chat_endpoint_rate_limiting(
        self,
        http_client: httpx.AsyncClient,
        valid_api_headers: Dict[str, str],
        sample_chat_request: Dict[str, Any],
    ):
        """Test rate limiting on chat endpoint with comprehensive validation"""
        logger.info("Testing rate limiting on chat endpoint")

        try:
            # Make multiple requests to trigger rate limiting
            responses = []

            for i in range(70):  # More than the rate limit
                logger.debug(f"Making request {i+1}/70")

                response = await http_client.post(
                    f"{MAIN_SERVICE_URL}/api/chat",
                    headers=valid_api_headers,
                    json=sample_chat_request,
                )
                responses.append(response)

                if response.status_code == 429:
                    logger.info(f"Rate limiting triggered after {i+1} requests")
                    break

                # Small delay to avoid overwhelming the service
                await asyncio.sleep(0.1)

            # Validate rate limiting behavior
            rate_limited_responses = [r for r in responses if r.status_code == 429]
            assert (
                len(rate_limited_responses) > 0
            ), "Rate limiting should have been triggered"

            logger.info(
                f"✓ Rate limiting test passed - {len(rate_limited_responses)} rate limited responses"
            )

        except Exception as e:
            logger.error(f"Error testing rate limiting: {e}")
            raise ServiceTestError(f"Rate limiting test failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.node_path
    async def test_chat_endpoint_node_path_coverage(
        self, http_client, valid_api_headers
    ):
        """
        For each diverse question, send a chat request and extract/log the node path from the queue manager.
        """
        # TODO: Fill in your own diverse questions here
        diverse_questions = [
            "Tell me about yourself",
            "Why did you build this?",
            "Do you have any work experience in the US?",
            "Sing a song about llms"
            # ... add more ...
        ]

        # Expected node paths for each question
        expected_paths = {
            "Tell me about yourself": [
                "relevance_check",
                "query_or_respond",
                "generate_with_context",
                "generate_with_persona",
            ],
            "Why did you build this?": [
                "relevance_check",
                "few_shot_selector",
                "generate_with_persona",
            ],
            "Do you have any work experience in the US?": [
                "relevance_check",
                "query_or_respond",
                "generate_with_context",
                "generate_with_persona",
            ],
            "Sing a song about llms": [
                "relevance_check",
                "few_shot_selector",
                "generate_with_persona",
            ],
            # ... add more expected paths ...
        }

        captured_conversation_flows = []

        # Mock the queue manager to capture conversation flows
        logger.info("Setting up mock for queue manager...")

        # Try different import paths for the mock
        mock_paths = [
            "evaluation_queue_manager.EvaluationQueueManager.enqueue_response",
            "app.EvaluationQueueManager.enqueue_response",
            "graph.graph.EvaluationQueueManager.enqueue_response",
        ]

        mock_created = False
        for mock_path in mock_paths:
            try:
                logger.info(f"Trying mock path: {mock_path}")
                with patch(mock_path) as mock_enqueue:
                    logger.info(f"✅ Mock created successfully with path: {mock_path}")
                    mock_created = True

                    def mock_enqueue_response(message):
                        logger.info(
                            f"🎯 MOCK CALLED! Received message for thread: {message.thread_id}"
                        )
                        logger.info(f"Message type: {type(message)}")
                        logger.info(f"Message attributes: {dir(message)}")

                        # Try to access conversation_flow safely
                        try:
                            if hasattr(message, "conversation_flow"):
                                node_path = [
                                    node.node_name
                                    for node in message.conversation_flow.node_executions
                                ]
                                logger.info(f"Node path extracted: {node_path}")
                            else:
                                logger.error(
                                    f"Message doesn't have conversation_flow attribute. Available: {[attr for attr in dir(message) if not attr.startswith('_')]}"
                                )
                                node_path = []

                            captured_conversation_flows.append(
                                {
                                    "query": getattr(message, "query", "unknown"),
                                    "node_path": node_path,
                                    "thread_id": getattr(
                                        message, "thread_id", "unknown"
                                    ),
                                }
                            )
                            logger.info(
                                f"Captured conversation flow for query: {getattr(message, 'query', 'unknown')}, node_path: {node_path}"
                            )
                        except Exception as e:
                            logger.error(f"Error processing mock message: {e}")
                            logger.error(f"Message content: {message}")

                    mock_enqueue.side_effect = mock_enqueue_response
                    logger.info("Mock side effect set")

                    for question in diverse_questions:
                        logger.info(f"Testing question: {question}")

                        # Use the correct request format that matches ChatRequest model
                        chat_request = {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": question,
                                    "thread_id": f"test-thread-{hash(question) % 10000}",
                                }
                            ]
                        }

                        response = await http_client.post(
                            f"{MAIN_SERVICE_URL}/api/chat",
                            json=chat_request,
                            headers=valid_api_headers,
                        )

                        assert (
                            response.status_code == 200
                        ), f"Chat request failed for: {question}"
                        logger.info(f"✅ Request successful for: {question}")

                        # Wait a bit for the graph to complete and queue the response
                        await asyncio.sleep(2)  # Increased wait time
                        logger.info(f"⏰ Waited 2 seconds after request for: {question}")

                    break  # If we get here, the mock worked

            except Exception as e:
                logger.warning(f"Mock path {mock_path} failed: {e}")
                continue

        if not mock_created:
            logger.error("❌ Failed to create mock with any path")
            pytest.skip("Could not create mock for queue manager")

        logger.info(
            f"Mock context exited. Captured flows: {len(captured_conversation_flows)}"
        )

        # Verify we captured conversation flows
        assert (
            len(captured_conversation_flows) > 0
        ), "No conversation flows were captured from queue manager."

        # Log and verify node paths
        for flow in captured_conversation_flows:
            query = flow["query"]
            actual_path = flow["node_path"]
            expected_path = expected_paths.get(query, [])

            logger.info(f"Query: {query}")
            logger.info(f"  Expected path: {expected_path}")
            logger.info(f"  Actual path: {actual_path}")

            if expected_path:
                assert (
                    actual_path == expected_path
                ), f"Node path mismatch for '{query}'. Expected: {expected_path}, Got: {actual_path}"
            else:
                logger.warning(f"No expected path defined for query: {query}")

        logger.info(
            f"✓ Node path coverage test completed. Tested {len(captured_conversation_flows)} questions."
        )


class TestEvaluationService:
    """Test suite for evaluation service with comprehensive validation"""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, http_client: httpx.AsyncClient):
        """Test evaluation service health endpoint with detailed validation"""
        logger.info("Testing evaluation service health endpoint")

        try:
            response = await http_client.get(f"{EVALUATION_SERVICE_URL}/health")

            # Validate response structure
            assert (
                response.status_code == 200
            ), f"Expected 200, got {response.status_code}"

            health_data = response.json()
            assert "status" in health_data, "Health response missing 'status' field"
            assert (
                health_data["status"] == "healthy"
            ), f"Expected 'healthy', got '{health_data['status']}'"
            assert (
                "queue_size" in health_data
            ), "Health response missing 'queue_size' field"
            assert (
                "timestamp" in health_data
            ), "Health response missing 'timestamp' field"

            logger.info(
                f"✓ Evaluation service health check passed - queue size: {health_data['queue_size']}"
            )

        except httpx.ConnectError as e:
            logger.error(f"Connection error to evaluation service: {e}")
            raise ServiceUnavailableError(f"Evaluation service not available: {e}")
        except Exception as e:
            logger.error(f"Unexpected error testing evaluation service health: {e}")
            raise ServiceTestError(f"Evaluation service health test failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Metrics needs to be implemented")
    async def test_metrics_endpoint(self, http_client: httpx.AsyncClient):
        """Test evaluation service metrics endpoint with comprehensive validation"""
        logger.info("Testing evaluation service metrics endpoint")

        try:
            response = await http_client.get(f"{EVALUATION_SERVICE_URL}/metrics")

            # Validate response structure
            assert (
                response.status_code == 200
            ), f"Expected 200, got {response.status_code}"

            metrics = response.json()
            required_fields = [
                "queue_size",
                "worker_running",
                "timestamp",
                "storage_path",
            ]

            for field in required_fields:
                assert field in metrics, f"Metrics response missing '{field}' field"

            # Validate data types
            assert isinstance(
                metrics["queue_size"], int
            ), "queue_size should be integer"
            assert isinstance(
                metrics["worker_running"], bool
            ), "worker_running should be boolean"
            assert isinstance(metrics["timestamp"], str), "timestamp should be string"

            logger.info(
                f"✓ Metrics endpoint test passed - queue size: {metrics['queue_size']}"
            )

        except Exception as e:
            logger.error(f"Error testing metrics endpoint: {e}")
            raise ServiceTestError(f"Metrics endpoint test failed: {e}")

    @pytest.mark.asyncio
    async def test_evaluate_endpoint_async(
        self, http_client: httpx.AsyncClient, sample_evaluation_request: Dict[str, Any]
    ):
        """Test async evaluation endpoint with comprehensive validation"""
        logger.info("Testing async evaluation endpoint")

        try:
            response = await http_client.post(
                f"{EVALUATION_SERVICE_URL}/evaluate", json=sample_evaluation_request
            )

            # Validate response structure
            assert (
                response.status_code == 200
            ), f"Expected 200, got {response.status_code}"

            result = response.json()
            assert "success" in result, "Response missing 'success' field"
            assert (
                result["success"] is True
            ), f"Expected success=True, got {result['success']}"
            assert (
                result["thread_id"] == sample_evaluation_request["thread_id"]
            ), "Thread ID mismatch"
            assert (
                result["evaluation_result"] is None
            ), "Async evaluation should return null result"
            assert "timestamp" in result, "Response missing 'timestamp' field"

            logger.info("✓ Async evaluation endpoint test passed")

        except Exception as e:
            logger.error(f"Error testing async evaluation endpoint: {e}")
            raise ServiceTestError(f"Async evaluation endpoint test failed: {e}")

    @pytest.mark.asyncio
    async def test_evaluate_endpoint_sync(
        self, http_client: httpx.AsyncClient, sample_evaluation_request: Dict[str, Any]
    ):
        """Test sync evaluation endpoint with error handling for OpenAI dependencies"""
        logger.info("Testing sync evaluation endpoint")

        try:
            response = await http_client.post(
                f"{EVALUATION_SERVICE_URL}/evaluate/sync",
                json=sample_evaluation_request,
            )

            # Validate response structure
            assert (
                response.status_code == 200
            ), f"Expected 200, got {response.status_code}"

            result = response.json()
            assert "thread_id" in result, "Response missing 'thread_id' field"
            assert (
                result["thread_id"] == sample_evaluation_request["thread_id"]
            ), "Thread ID mismatch"
            assert "success" in result, "Response missing 'success' field"
            assert "timestamp" in result, "Response missing 'timestamp' field"

            # Note: sync evaluation might fail if OpenAI API is not configured
            # So we don't assert success, just check the response structure
            logger.info(
                f"✓ Sync evaluation endpoint test passed - success: {result.get('success')}"
            )

        except Exception as e:
            logger.error(f"Error testing sync evaluation endpoint: {e}")
            raise ServiceTestError(f"Sync evaluation endpoint test failed: {e}")


class TestServiceCommunication:
    """Test suite for service-to-service communication with comprehensive error handling"""

    @pytest.mark.asyncio
    async def test_evaluation_client_health_check(self):
        """Test evaluation client health check with proper resource management"""
        logger.info("Testing evaluation client health check")

        try:
            from evaluation_client import get_evaluation_client

            client = await get_evaluation_client()

            try:
                health = await client.health_check()

                # Validate health response
                assert "status" in health, "Health response missing 'status' field"
                assert (
                    health["status"] == "healthy"
                ), f"Expected 'healthy', got '{health['status']}'"
                assert (
                    "timestamp" in health
                ), "Health response missing 'timestamp' field"

                logger.info("✓ Evaluation client health check passed")

            finally:
                await client.close()
                logger.debug("Evaluation client closed")

        except ImportError as e:
            logger.error(f"Failed to import evaluation client: {e}")
            raise ServiceTestError(f"Evaluation client import failed: {e}")
        except Exception as e:
            logger.error(f"Error testing evaluation client health check: {e}")
            raise ServiceTestError(f"Evaluation client health check failed: {e}")

    @pytest.mark.asyncio
    async def test_evaluation_client_metrics(self):
        """Test evaluation client metrics with comprehensive validation"""
        logger.info("Testing evaluation client metrics")

        try:
            from evaluation_client import get_evaluation_client

            client = await get_evaluation_client()

            try:
                metrics = await client.get_metrics()

                # Validate metrics response
                required_fields = ["queue_size", "worker_running", "timestamp"]
                for field in required_fields:
                    assert field in metrics, f"Metrics response missing '{field}' field"

                assert isinstance(
                    metrics["queue_size"], int
                ), "queue_size should be integer"
                assert isinstance(
                    metrics["worker_running"], bool
                ), "worker_running should be boolean"

                logger.info(
                    f"✓ Evaluation client metrics test passed - queue size: {metrics['queue_size']}"
                )

            finally:
                await client.close()
                logger.debug("Evaluation client closed")

        except Exception as e:
            logger.error(f"Error testing evaluation client metrics: {e}")
            raise ServiceTestError(f"Evaluation client metrics test failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="Evaluation client async evaluation needs to be implemented"
    )
    async def test_evaluation_client_async_evaluation(
        self, sample_evaluation_request: Dict[str, Any]
    ):
        """Test evaluation client async evaluation with comprehensive validation"""
        logger.info("Testing evaluation client async evaluation")

        try:
            from evaluation_client import get_evaluation_client

            client = await get_evaluation_client()

            try:
                result = await client.evaluate_conversation_async(
                    thread_id=sample_evaluation_request["thread_id"],
                    query=sample_evaluation_request["query"],
                    response=sample_evaluation_request["response"],
                    conversation_flow=sample_evaluation_request["conversation_flow"],
                )

                # Validate evaluation result
                assert (
                    result.success is True
                ), f"Expected success=True, got {result.success}"
                assert (
                    result.thread_id == sample_evaluation_request["thread_id"]
                ), "Thread ID mismatch"
                assert result.timestamp is not None, "Timestamp should not be null"

                logger.info("✓ Evaluation client async evaluation test passed")

            finally:
                await client.close()
                logger.debug("Evaluation client closed")

        except Exception as e:
            logger.error(f"Error testing evaluation client async evaluation: {e}")
            raise ServiceTestError(
                f"Evaluation client async evaluation test failed: {e}"
            )


@pytest.mark.asyncio
async def test_full_integration_flow(
    http_client: httpx.AsyncClient,
    valid_api_headers: Dict[str, str],
    sample_chat_request: Dict[str, Any],
):
    """Test the complete integration flow from chat to evaluation with comprehensive validation"""
    logger.info("Testing complete integration flow")

    try:
        # Step 1: Make a chat request and validate streaming response
        logger.debug("Step 1: Making chat request")
        chat_response = await http_client.post(
            f"{MAIN_SERVICE_URL}/api/chat",
            headers=valid_api_headers,
            json=sample_chat_request,
        )

        assert chat_response.status_code == 200, "Chat request should succeed"
        assert chat_response.headers["content-type"].startswith(
            "text/event-stream"
        ), "Expected streaming response"

        # Read streaming response
        content = ""
        async for chunk in chat_response.aiter_text():
            content += chunk

        assert len(content) > 0, "Response should not be empty"
        logger.info(
            f"✓ Chat request successful - response length: {len(content)} chars"
        )

        # Step 2: Check evaluation service health and queue status
        logger.debug("Step 2: Checking evaluation service status")
        await asyncio.sleep(2)  # Give time for evaluation to be queued

        eval_health = await http_client.get(f"{EVALUATION_SERVICE_URL}/health")
        assert eval_health.status_code == 200, "Evaluation service should be healthy"

        health_data = eval_health.json()
        assert (
            health_data["status"] == "healthy"
        ), "Evaluation service should be healthy"
        logger.info(
            f"✓ Evaluation service healthy - queue size: {health_data.get('queue_size', 'N/A')}"
        )

        # Step 3: Check evaluation service metrics
        logger.debug("Step 3: Checking evaluation service metrics")
        eval_metrics = await http_client.get(f"{EVALUATION_SERVICE_URL}/metrics")
        assert eval_metrics.status_code == 200, "Metrics endpoint should be accessible"

        metrics_data = eval_metrics.json()
        required_metrics = ["queue_size", "worker_running", "timestamp", "storage_path"]
        for metric in required_metrics:
            assert metric in metrics_data, f"Metrics should include {metric}"

        logger.info(
            f"✓ Evaluation metrics retrieved - worker running: {metrics_data.get('worker_running')}"
        )

        # Step 4: Wait for evaluation processing and check queue status again
        logger.debug("Step 4: Waiting for evaluation processing")
        await asyncio.sleep(5)  # Give more time for evaluation processing

        # Check if queue was processed
        eval_health_after = await http_client.get(f"{EVALUATION_SERVICE_URL}/health")
        health_data_after = eval_health_after.json()

        # The queue should have been processed (size should be 0 or reduced)
        logger.info(
            f"✓ Queue processing check - initial size: {health_data.get('queue_size', 'N/A')}, final size: {health_data_after.get('queue_size', 'N/A')}"
        )

        # Step 5: Validate that the evaluation service is still healthy after processing
        assert (
            health_data_after["status"] == "healthy"
        ), "Evaluation service should remain healthy after processing"

        logger.info("✓ Full integration flow test passed successfully")

    except Exception as e:
        logger.error(f"Error in full integration flow test: {e}")
        raise ServiceTestError(f"Full integration flow test failed: {e}")


# Legacy main function for backward compatibility with proper error handling
async def main():
    """Legacy main function with comprehensive error handling and logging"""
    logger.warning("Using legacy main function. Consider using pytest instead.")

    try:
        # Run basic health checks
        async with httpx.AsyncClient() as client:
            logger.info("Running legacy health checks")

            # Test main service
            try:
                response = await client.get(f"{MAIN_SERVICE_URL}/health")
                logger.info(f"Main service health: {response.status_code}")
            except Exception as e:
                logger.error(f"Main service health check failed: {e}")
                return False

            # Test evaluation service
            try:
                response = await client.get(f"{EVALUATION_SERVICE_URL}/health")
                logger.info(f"Evaluation service health: {response.status_code}")
            except Exception as e:
                logger.error(f"Evaluation service health check failed: {e}")
                return False

        logger.info("✓ Legacy health checks completed successfully")
        return True

    except Exception as e:
        logger.error(f"Legacy main function failed: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(main())
