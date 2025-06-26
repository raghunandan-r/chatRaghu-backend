#!/usr/bin/env python3
"""
Unit tests for evaluation service endpoints.
Tests health checks, metrics, and evaluation functionality.

Note: This file contains unit tests for individual evaluation service endpoints.
For integration tests that verify service-to-service communication and full flows,
see test_integration.py.
"""

import pytest
import httpx
from typing import Dict, Any
from .conftest import (
    EVALUATION_SERVICE_URL,
    ServiceTestError,
    ServiceUnavailableError,
    logger,
)


class TestEvaluationService:
    """Test suite for evaluation service"""

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
                response.status_code == 200
            ), f"Expected 200, got {response.status_code}"

            response_data = response.json()
            assert "thread_id" in response_data, "Response missing 'thread_id' field"
            assert "success" in response_data, "Response missing 'success' field"
            assert isinstance(
                response_data["success"], bool
            ), "success field should be a boolean"
            assert "timestamp" in response_data, "Response missing 'timestamp' field"

            # For async evaluation, evaluation_result should be None
            assert (
                "evaluation_result" in response_data
            ), "Response missing 'evaluation_result' field"
            assert (
                response_data["evaluation_result"] is None
            ), "Async evaluation should return None for evaluation_result"

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
    async def test_evaluate_endpoint_sync(
        self, http_client: httpx.AsyncClient, sample_evaluation_request: Dict[str, Any]
    ):
        """Test evaluation service sync evaluate endpoint"""
        logger.info("Testing evaluation service sync evaluate endpoint")

        try:
            response = await http_client.post(
                f"{EVALUATION_SERVICE_URL}/evaluate/sync",
                json=sample_evaluation_request,
            )

            # Validate response structure and content
            assert (
                response.status_code == 200
            ), f"Expected 200, got {response.status_code}"

            response_data = response.json()
            assert (
                "evaluation_result" in response_data
            ), "Response missing 'evaluation_result' field"
            assert "success" in response_data, "Response missing 'success' field"
            assert isinstance(
                response_data["success"], bool
            ), "success field should be a boolean"

            # Validate evaluation result structure if present
            if response_data["evaluation_result"]:
                result = response_data["evaluation_result"]
                assert (
                    "thread_id" in result
                ), "Evaluation result missing 'thread_id' field"
                assert "scores" in result, "Evaluation result missing 'scores' field"

            logger.info("✓ Evaluation service sync evaluate endpoint passed")

        except httpx.ConnectError as e:
            logger.error(f"Connection error to evaluation service: {e}")
            raise ServiceUnavailableError(f"Evaluation service not available: {e}")
        except httpx.TimeoutException as e:
            logger.error(f"Timeout connecting to evaluation service: {e}")
            raise ServiceTestError(f"Evaluation service timeout: {e}")
        except Exception as e:
            logger.error(f"Unexpected error testing sync evaluate endpoint: {e}")
            raise ServiceTestError(f"Sync evaluate endpoint test failed: {e}")
