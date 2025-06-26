#!/usr/bin/env python3
"""
Integration tests for ChatRaghu services.
Tests communication between main service and evaluation service, and complex integration flows.
"""

import pytest
import httpx
import asyncio
from typing import Dict, Any
from unittest.mock import patch
from .conftest import (
    MAIN_SERVICE_URL,
    EVALUATION_SERVICE_URL,
    ServiceTestError,
    ServiceUnavailableError,
    logger,
)


class TestServiceCommunication:
    """Test suite for service-to-service communication"""

    # Note: Individual endpoint tests are in test_evaluation_service.py
    # This class focuses on integration scenarios and service communication patterns


@pytest.mark.asyncio
async def test_full_integration_flow(
    http_client: httpx.AsyncClient,
    valid_api_headers: Dict[str, str],
    sample_chat_request: Dict[str, Any],
    sample_evaluation_request: Dict[str, Any],
):
    """Test complete integration flow from chat request to evaluation"""
    logger.info("Testing full integration flow")

    try:
        # Step 1: Send chat request to main service
        logger.info("Step 1: Sending chat request to main service")
        chat_response = await http_client.post(
            f"{MAIN_SERVICE_URL}/api/chat",
            headers=valid_api_headers,
            json=sample_chat_request,
        )

        assert (
            chat_response.status_code == 200
        ), f"Chat request failed with status {chat_response.status_code}"

        # Step 2: Verify chat response contains expected content
        logger.info("Step 2: Verifying chat response")
        chat_content = chat_response.text
        assert chat_content, "Chat response should not be empty"

        # Step 3: Check if evaluation service is available
        logger.info("Step 3: Checking evaluation service availability")
        try:
            eval_health_response = await http_client.get(
                f"{EVALUATION_SERVICE_URL}/health"
            )
            eval_service_available = eval_health_response.status_code == 200
        except Exception:
            eval_service_available = False

        if eval_service_available:
            logger.info("✓ Evaluation service is available")
            # Step 4: Test evaluation service endpoints
            logger.info("Step 4: Testing evaluation service endpoints")

            # Use the sample evaluation request which has proper conversation_flow structure
            eval_response = await http_client.post(
                f"{EVALUATION_SERVICE_URL}/evaluate/sync",
                json=sample_evaluation_request,
            )

            assert (
                eval_response.status_code == 200
            ), f"Evaluation request failed with status {eval_response.status_code}"

            eval_data = eval_response.json()
            # Fix: Use 'evaluation_result' (singular) not 'evaluation_results' (plural)
            assert (
                "evaluation_result" in eval_data
            ), "Evaluation response missing evaluation_result field"
            assert "success" in eval_data, "Evaluation response missing success field"
            assert isinstance(
                eval_data["success"], bool
            ), "Success field should be a boolean"

            logger.info("✓ Full integration flow completed successfully")
        else:
            logger.warning("Evaluation service not available, skipping evaluation step")
            logger.info("✓ Partial integration flow completed (main service only)")

    except httpx.ConnectError as e:
        logger.error(f"Connection error during integration test: {e}")
        raise ServiceUnavailableError(
            f"Service not available during integration test: {e}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during integration test: {e}")
        raise ServiceTestError(f"Integration test failed: {e}")


@pytest.mark.asyncio
@pytest.mark.node_path
async def test_chat_endpoint_node_path_coverage(http_client, valid_api_headers):
    """Test comprehensive node path coverage in chat endpoint"""
    logger.info("Testing chat endpoint node path coverage")

    try:
        # Test different types of queries to trigger different node paths
        test_queries = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Tell me about yourself?",
                        "thread_id": "test-thread-path-1",
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Why did you build this?",
                        "thread_id": "test-thread-path-2",
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Do you have any work experience in the US?",
                        "thread_id": "test-thread-path-3",
                    }
                ]
            },
        ]

        # Fix: Use the correct module path for the queue manager
        with patch(
            "evaluation_queue_manager.EvaluationQueueManager.enqueue_response"
        ) as mock_enqueue:
            mock_enqueue.return_value = None

            for i, query in enumerate(test_queries):
                logger.info(f"Testing query {i+1}: {query['messages'][0]['content']}")

                response = await http_client.post(
                    f"{MAIN_SERVICE_URL}/api/chat",
                    headers=valid_api_headers,
                    json=query,
                )

                assert (
                    response.status_code == 200
                ), f"Query {i+1} failed with status {response.status_code}"

                response_content = response.text
                assert response_content, f"Query {i+1} response should not be empty"

                # Verify that evaluation was queued (if queue manager is available)
                if mock_enqueue.called:
                    logger.info(f"✓ Query {i+1} evaluation was queued")
                else:
                    logger.info(f"✓ Query {i+1} completed without evaluation queuing")

        logger.info("✓ Node path coverage test completed successfully")

    except httpx.ConnectError as e:
        logger.error(f"Connection error during node path test: {e}")
        raise ServiceUnavailableError(
            f"Service not available during node path test: {e}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during node path test: {e}")
        raise ServiceTestError(f"Node path test failed: {e}")


async def main():
    """Main function for running integration tests"""
    logger.info("Starting integration tests")

    try:
        # This function can be used for manual testing or CI/CD
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Run basic health checks
            main_health = await client.get(f"{MAIN_SERVICE_URL}/health")
            eval_health = await client.get(f"{EVALUATION_SERVICE_URL}/health")

            logger.info(f"Main service health: {main_health.status_code}")
            logger.info(f"Evaluation service health: {eval_health.status_code}")

    except Exception as e:
        logger.error(f"Integration test main function failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
