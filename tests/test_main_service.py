#!/usr/bin/env python3
"""
Unit tests for main ChatRaghu backend service endpoints.
Tests health checks, API key validation, and chat endpoint functionality.
"""

import pytest
import httpx
from typing import Dict, Any
from .conftest import (
    MAIN_SERVICE_URL,
    ServiceTestError,
    ServiceUnavailableError,
    logger,
)


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

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Rate limiting is not to be tested hereafter")
    async def test_chat_endpoint_rate_limiting(
        self,
        http_client: httpx.AsyncClient,
        valid_api_headers: Dict[str, str],
        sample_chat_request: Dict[str, Any],
    ):
        """Test rate limiting on chat endpoint"""
        logger.info("Testing chat endpoint rate limiting")

        try:
            # Send multiple requests rapidly to trigger rate limiting
            responses = []
            for i in range(10):
                response = await http_client.post(
                    f"{MAIN_SERVICE_URL}/api/chat",
                    headers=valid_api_headers,
                    json=sample_chat_request,
                )
                responses.append(response)

            # Check if any request was rate limited (429 status)
            rate_limited_responses = [
                resp for resp in responses if resp.status_code == 429
            ]

            if rate_limited_responses:
                logger.info("✓ Rate limiting detected and working")
                # Validate rate limit response structure
                rate_limit_response = rate_limited_responses[0]
                response_data = rate_limit_response.json()
                assert (
                    "detail" in response_data
                ), "Rate limit response missing 'detail' field"
                assert (
                    "rate limit" in response_data["detail"].lower()
                    or "too many" in response_data["detail"].lower()
                ), f"Expected rate limit error message, got {response_data['detail']}"
            else:
                logger.info("✓ No rate limiting triggered (this is acceptable)")

        except httpx.ConnectError as e:
            logger.error(f"Connection error to main service: {e}")
            raise ServiceUnavailableError(f"Main service not available: {e}")
        except Exception as e:
            logger.error(f"Unexpected error testing rate limiting: {e}")
            raise ServiceTestError(f"Rate limiting test failed: {e}")
