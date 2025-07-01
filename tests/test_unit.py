#!/usr/bin/env python3
"""
Consolidated unit tests for both ChatRaghu services.

This file contains all unit tests organized by service and component:
- Main service unit tests (API endpoints, validation)
- Evaluation service unit tests (components, queue manager, storage)
- Configuration and utility tests

Test Organization:
- TestMainService: Main backend service endpoint tests
- TestEvaluationService: Evaluation service endpoint tests
- TestEvaluationComponents: Internal evaluation service component tests
- TestConfiguration: Configuration loading and validation tests
"""

import pytest
import httpx
import sys
import os
from typing import Dict, Any

from .conftest import (
    MAIN_SERVICE_URL,
    EVALUATION_SERVICE_URL,
    ServiceTestError,
    ServiceUnavailableError,
    logger,
)

# Add evals-service to path for imports BEFORE any other imports
evals_service_path = os.path.join(os.path.dirname(__file__), "..", "evals-service")
if evals_service_path not in sys.path:
    sys.path.insert(0, evals_service_path)


# ============================================================================
# MAIN SERVICE UNIT TESTS
# ============================================================================


@pytest.mark.unit
class TestMainService:
    """Unit tests for main ChatRaghu backend service endpoints"""

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

            # Check for streaming response headers
            content_type = response.headers.get("content-type", "")
            assert (
                "text/plain" in content_type or "text/event-stream" in content_type
            ), f"Expected streaming content type, got {content_type}"

            # Validate response content structure
            response_text = response.text
            assert response_text, "Response should not be empty"

            # Check for expected response patterns
            assert any(
                keyword in response_text.lower()
                for keyword in ["hello", "hi", "greeting", "rag"]
            ), "Response should contain greeting or relevant content"

            logger.info("✓ Chat endpoint with valid API key passed")

        except httpx.ConnectError as e:
            logger.error(f"Connection error to main service: {e}")
            raise ServiceUnavailableError(f"Main service not available: {e}")
        except httpx.TimeoutException as e:
            logger.error(f"Timeout connecting to main service: {e}")
            raise ServiceTestError(f"Main service timeout: {e}")
        except Exception as e:
            logger.error(f"Unexpected error testing chat endpoint: {e}")
            raise ServiceTestError(f"Chat endpoint test failed: {e}")

    @pytest.mark.asyncio
    async def test_chat_endpoint_without_api_key(
        self, http_client: httpx.AsyncClient, sample_chat_request: Dict[str, Any]
    ):
        """Test chat endpoint without API key - should return 422 (validation error)"""
        logger.info("Testing chat endpoint without API key")

        try:
            response = await http_client.post(
                f"{MAIN_SERVICE_URL}/api/chat",
                json=sample_chat_request,
            )

            assert (
                response.status_code == 422
            ), f"Expected 422 (validation error), got {response.status_code}"

            logger.info("✓ Chat endpoint without API key correctly rejected")

        except httpx.ConnectError as e:
            logger.error(f"Connection error to main service: {e}")
            raise ServiceUnavailableError(f"Main service not available: {e}")
        except Exception as e:
            logger.error(f"Unexpected error testing chat endpoint without API key: {e}")
            raise ServiceTestError(f"Chat endpoint without API key test failed: {e}")

    @pytest.mark.asyncio
    async def test_chat_endpoint_with_invalid_api_key(
        self,
        http_client: httpx.AsyncClient,
        invalid_api_headers: Dict[str, str],
        sample_chat_request: Dict[str, Any],
    ):
        """Test chat endpoint with invalid API key - should return 401"""
        logger.info("Testing chat endpoint with invalid API key")

        try:
            response = await http_client.post(
                f"{MAIN_SERVICE_URL}/api/chat",
                headers=invalid_api_headers,
                json=sample_chat_request,
            )

            assert (
                response.status_code == 401
            ), f"Expected 401, got {response.status_code}"

            # Validate error response structure
            response_data = response.json()
            assert "detail" in response_data, "Error response missing 'detail' field"
            assert (
                "invalid" in response_data["detail"].lower()
                or "unauthorized" in response_data["detail"].lower()
            ), f"Expected error about invalid/unauthorized, got {response_data['detail']}"

            logger.info("✓ Chat endpoint with invalid API key correctly rejected")

        except httpx.ConnectError as e:
            logger.error(f"Connection error to main service: {e}")
            raise ServiceUnavailableError(f"Main service not available: {e}")
        except Exception as e:
            logger.error(
                f"Unexpected error testing chat endpoint with invalid API key: {e}"
            )
            raise ServiceTestError(
                f"Chat endpoint with invalid API key test failed: {e}"
            )


# ============================================================================
# EVALUATION SERVICE UNIT TESTS
# ============================================================================


@pytest.mark.unit
class TestEvaluationService:
    """Unit tests for evaluation service HTTP endpoints"""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, http_client: httpx.AsyncClient):
        """Test evaluation service health endpoint"""
        logger.info("Testing evaluation service health endpoint")

        try:
            response = await http_client.get(f"{EVALUATION_SERVICE_URL}/health")

            # Validate response structure and content
            assert (
                response.status_code == 200
            ), f"Expected 200, got {response.status_code}"

            response_data = response.json()
            assert "status" in response_data, "Health response missing 'status' field"
            assert (
                response_data["status"] == "healthy"
            ), f"Expected 'healthy', got '{response_data['status']}'"

            logger.info("✓ Evaluation service health check passed")

        except httpx.ConnectError as e:
            logger.error(f"Connection error to evaluation service: {e}")
            raise ServiceUnavailableError(f"Evaluation service not available: {e}")
        except httpx.TimeoutException as e:
            logger.error(f"Timeout connecting to evaluation service: {e}")
            raise ServiceTestError(f"Evaluation service timeout: {e}")
        except Exception as e:
            logger.error(f"Unexpected error testing evaluation service health: {e}")
            raise ServiceTestError(f"Evaluation service health test failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Metrics needs to be implemented")
    async def test_metrics_endpoint(self, http_client: httpx.AsyncClient):
        """Test evaluation service metrics endpoint"""
        logger.info("Testing evaluation service metrics endpoint")

        try:
            response = await http_client.get(f"{EVALUATION_SERVICE_URL}/metrics")

            # Validate response structure and content
            assert (
                response.status_code == 200
            ), f"Expected 200, got {response.status_code}"

            # Check for metrics content type
            content_type = response.headers.get("content-type", "")
            assert (
                "text/plain" in content_type
            ), f"Expected text/plain content type, got {content_type}"

            # Validate metrics content
            metrics_text = response.text
            assert metrics_text, "Metrics response should not be empty"

            # Check for expected metrics patterns
            assert any(
                metric in metrics_text.lower()
                for metric in [
                    "http_requests_total",
                    "http_request_duration",
                    "evaluation",
                ]
            ), "Metrics should contain standard HTTP and evaluation metrics"

            logger.info("✓ Evaluation service metrics endpoint passed")

        except httpx.ConnectError as e:
            logger.error(f"Connection error to evaluation service: {e}")
            raise ServiceUnavailableError(f"Evaluation service not available: {e}")
        except httpx.TimeoutException as e:
            logger.error(f"Timeout connecting to evaluation service: {e}")
            raise ServiceTestError(f"Evaluation service timeout: {e}")
        except Exception as e:
            logger.error(f"Unexpected error testing metrics endpoint: {e}")
            raise ServiceTestError(f"Metrics endpoint test failed: {e}")

    @pytest.mark.asyncio
    async def test_evaluate_endpoint_async(
        self, http_client: httpx.AsyncClient, sample_evaluation_request: Dict[str, Any]
    ):
        """Test evaluation service async evaluate endpoint"""
        logger.info("Testing evaluation service async evaluate endpoint")

        try:
            response = await http_client.post(
                f"{EVALUATION_SERVICE_URL}/evaluate",
                json=sample_evaluation_request,
            )

            # Validate response structure and content
            assert (
                response.status_code == 202
            ), f"Expected 202, got {response.status_code}"

            response_data = response.json()
            assert "thread_id" in response_data, "Response missing 'thread_id' field"
            assert "success" in response_data, "Response missing 'success' field"
            assert isinstance(
                response_data["success"], bool
            ), "success field should be a boolean"
            assert "timestamp" in response_data, "Response missing 'timestamp' field"

            logger.info("✓ Evaluation service async evaluate endpoint passed")

        except httpx.ConnectError as e:
            logger.error(f"Connection error to evaluation service: {e}")
            raise ServiceUnavailableError(f"Evaluation service not available: {e}")
        except httpx.TimeoutException as e:
            logger.error(f"Timeout connecting to evaluation service: {e}")
            raise ServiceTestError(f"Evaluation service timeout: {e}")
        except Exception as e:
            logger.error(f"Unexpected error testing async evaluate endpoint: {e}")
            raise ServiceTestError(f"Async evaluate endpoint test failed: {e}")

    @pytest.mark.asyncio
    async def test_evaluation_service_config(self, http_client: httpx.AsyncClient):
        """Test evaluation service config endpoint"""
        logger.info("Testing evaluation service config endpoint")

        try:
            response = await http_client.get(f"{EVALUATION_SERVICE_URL}/config")

            assert (
                response.status_code == 200
            ), f"Config check failed with status {response.status_code}"

            config_data = response.json()
            assert "service" in config_data
            assert "storage" in config_data
            assert "llm" in config_data
            assert "api" in config_data

            # Verify no sensitive data is exposed
            assert "openai_api_key" not in str(config_data)

            logger.info("✓ Evaluation service config check passed")

        except httpx.ConnectError as e:
            logger.warning(f"Evaluation service not available: {e}")
            pytest.skip("Evaluation service not available")
