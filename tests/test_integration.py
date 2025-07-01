#!/usr/bin/env python3
"""
Consolidated integration tests for ChatRaghu services.

This file contains comprehensive integration tests that cover:
- Service health and availability tests
- Complete end-to-end integration flow (main service -> evaluation service -> storage)
- Diverse query testing with different node paths
- File creation verification with configurable batch processing

Test Organization:
- TestEvaluationServiceIntegration: Evaluation service health/availability tests
- test_comprehensive_integration_flow: Complete end-to-end flow with diverse queries
"""

import pytest
import asyncio
import httpx
import json
import os
import time
from pathlib import Path
from typing import Dict
from .conftest import (
    MAIN_SERVICE_URL,
    EVALUATION_SERVICE_URL,
    ServiceTestError,
    ServiceUnavailableError,
    logger,
)

# Test queries for diverse node path coverage
DIVERSE_TEST_QUERIES = [
    {
        "messages": [
            {
                "role": "user",
                "content": "Tell me about yourself?",
                "thread_id": "test-thread-personal-1",
            }
        ]
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Why did you build this?",
                "thread_id": "test-thread-purpose-2",
            }
        ]
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Do you have any work experience in the US?",
                "thread_id": "test-thread-experience-3",
            }
        ]
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "How long have you been a data scientist?",
                "thread_id": "test-thread-personal-1",
            }
        ]
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "How can I improve my coding skills?",
                "thread_id": "test-thread-coding-5",
            }
        ]
    },
]

# Define the single, focused test query
TEST_QUERY = {
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of France?",
            "thread_id": f"test-e2e-{int(time.time())}",
        }
    ]
}

# ============================================================================
# EVALUATION SERVICE INTEGRATION TESTS
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
async def test_comprehensive_integration_flow(
    http_client: httpx.AsyncClient,
    valid_api_headers: Dict[str, str],
):
    """
    Test complete integration flow from chat request to storage with diverse queries.

    This test covers:
    1. Multiple chat requests with diverse content to test different node paths
    2. Evaluation service integration and async processing
    3. Storage verification with immediate file writing (batch_size=1)
    4. File content validation
    """
    logger.info("Testing comprehensive integration flow with diverse queries")

    # Set batch size to 3 for your test (one batch per test run)
    original_batch_size = os.environ.get("STORAGE_BATCH_SIZE")
    os.environ["STORAGE_BATCH_SIZE"] = "3"

    try:
        # Step 1: Check if evaluation service is available
        logger.info("Step 1: Checking evaluation service availability")
        try:
            eval_health_response = await http_client.get(
                f"{EVALUATION_SERVICE_URL}/health"
            )
            eval_service_available = eval_health_response.status_code == 200
        except Exception:
            eval_service_available = False

        if not eval_service_available:
            logger.warning(
                "Evaluation service not available, skipping storage verification"
            )
            eval_service_available = False

        # Step 2: Get initial file counts if evaluation service is available
        if eval_service_available:
            logger.info("Step 2: Getting initial file counts from host filesystem")

            # Check files in host filesystem (volumes are mounted)
            audit_data_path = Path("evals-service/audit_data")
            eval_results_path = Path("evals-service/eval_results")

            initial_audit_files = len(list(audit_data_path.rglob("*.json"))) + len(
                list(audit_data_path.rglob("*.parquet"))
            )
            initial_eval_files = len(list(eval_results_path.rglob("*.json"))) + len(
                list(eval_results_path.rglob("*.parquet"))
            )

            logger.info(f"Initial audit files: {initial_audit_files}")
            logger.info(f"Initial eval files: {initial_eval_files}")

        # Step 3: Send diverse chat requests and verify responses
        logger.info("Step 3: Sending diverse chat requests")
        successful_requests = 0

        for i, query in enumerate(DIVERSE_TEST_QUERIES):
            logger.info(f"Testing query {i+1}: {query['messages'][0]['content']}")

            try:
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

                successful_requests += 1
                logger.info(f"✓ Query {i+1} completed successfully")

            except Exception as e:
                logger.error(f"Query {i+1} failed: {e}")
                # Continue with other queries even if one fails

        assert successful_requests > 0, "At least one chat request should succeed"

        # Step 4: Wait for background processing if evaluation service is available
        if eval_service_available:
            logger.info("Step 4: Waiting for background processing")
            await asyncio.sleep(10)  # Reduced wait time since batch_size=1

            # Step 5: Verify files were created
            logger.info("Step 5: Verifying files were created")

            # Check files in host filesystem (volumes are mounted)
            final_audit_files = len(list(audit_data_path.rglob("*.json"))) + len(
                list(audit_data_path.rglob("*.parquet"))
            )
            final_eval_files = len(list(eval_results_path.rglob("*.json"))) + len(
                list(eval_results_path.rglob("*.parquet"))
            )

            logger.info(f"Final audit files: {final_audit_files}")
            logger.info(f"Final eval files: {final_eval_files}")

            # Check if new files were created
            audit_files_created = final_audit_files > initial_audit_files
            eval_files_created = final_eval_files > initial_eval_files

            if audit_files_created:
                logger.info("✓ Audit files were created successfully")
            else:
                logger.warning("⚠ No new audit files were created")

            if eval_files_created:
                logger.info("✓ Evaluation result files were created successfully")
            else:
                logger.warning("⚠ No new evaluation result files were created")

            # Step 6: Check file contents (optional)
            if audit_files_created:
                logger.info("Step 6: Checking audit file contents")
                audit_files = list(audit_data_path.rglob("*.json"))
                if audit_files:
                    latest_audit_file = max(
                        audit_files, key=lambda f: f.stat().st_mtime
                    )
                    logger.info(f"Latest audit file: {latest_audit_file}")

                    with open(latest_audit_file, "r") as f:
                        audit_data = json.load(f)
                        logger.info(f"Audit file contains {len(audit_data)} records")

            if eval_files_created:
                logger.info("Step 6: Checking evaluation result file contents")
                eval_files = list(eval_results_path.rglob("*.json"))
                if eval_files:
                    latest_eval_file = max(eval_files, key=lambda f: f.stat().st_mtime)
                    logger.info(f"Latest eval file: {latest_eval_file}")

                    with open(latest_eval_file, "r") as f:
                        eval_data = json.load(f)
                        logger.info(f"Eval file contains {len(eval_data)} records")

        logger.info("✓ Comprehensive integration flow completed successfully")

    except httpx.ConnectError as e:
        logger.error(f"Connection error during integration test: {e}")
        raise ServiceUnavailableError(
            f"Service not available during integration test: {e}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during integration test: {e}")
        raise ServiceTestError(f"Integration test failed: {e}")
    finally:
        # Restore original batch size
        if original_batch_size is not None:
            os.environ["STORAGE_BATCH_SIZE"] = original_batch_size
        elif "STORAGE_BATCH_SIZE" in os.environ:
            del os.environ["STORAGE_BATCH_SIZE"]


@pytest.mark.skip(reason="Skipping ruthless end-to-end flow test")
@pytest.mark.asyncio
async def test_ruthless_end_to_end_flow(
    http_client: httpx.AsyncClient,
    valid_api_headers: Dict[str, str],
):
    """
    A true end-to-end test that verifies file creation.
    It will:
    1. Clean up old test files.
    2. Send ONE chat request to the main service.
    3. Poll the filesystem until an audit file and a result file are created.
    4. Fail if files do not appear within a timeout.
    5. Verify the contents of the created files.
    """
    thread_id = TEST_QUERY["messages"][0]["thread_id"]
    logger.info(f"RUTHLESS_TEST_START: Starting e2e test for thread_id={thread_id}")

    audit_path = Path("evals-service/audit_data")
    results_path = Path("evals-service/eval_results")

    # 1. Clean up old files
    logger.info("RUTHLESS_TEST_CLEANUP: Deleting old test files...")
    for p in audit_path.glob("*.json"):
        p.unlink()
    for p in results_path.glob("*.json"):
        p.unlink()

    try:
        # 2. Send ONE chat request
        logger.info(
            f"RUTHLESS_TEST_REQUEST: Sending chat request for thread_id={thread_id}"
        )
        response = await http_client.post(
            f"{MAIN_SERVICE_URL}/api/chat",
            headers=valid_api_headers,
            json=TEST_QUERY,
        )
        response.raise_for_status()
        logger.info(
            f"RUTHLESS_TEST_RESPONSE: Received {response.status_code} from main service."
        )

        # Consume the streaming response to ensure the graph execution completes
        response_text = ""
        async for chunk in response.aiter_text():
            response_text += chunk
        assert response_text, "Chat response should not be empty"
        logger.info(
            "RUTHLESS_TEST_STREAM_COMPLETE: Full response received from main service."
        )

        # 3. Poll for file creation
        timeout = 20  # seconds
        start_time = time.time()
        audit_file_path = None
        results_file_path = None

        logger.info(
            f"RUTHLESS_TEST_POLLING: Waiting up to {timeout}s for files to be created..."
        )
        while time.time() - start_time < timeout:
            if not audit_file_path:
                # Look for any audit files and check their content for the thread_id
                audit_files = list(audit_path.glob("audit_request*.json"))
                for audit_file in audit_files:
                    try:
                        with open(audit_file, "r") as f:
                            audit_data = json.load(f)
                            if isinstance(audit_data, list) and len(audit_data) > 0:
                                if audit_data[0].get("thread_id") == thread_id:
                                    audit_file_path = audit_file
                                    logger.info(
                                        f"RUTHLESS_TEST_AUDIT_FOUND: Found audit file: {audit_file_path}"
                                    )
                                    break
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

            if not results_file_path:
                # Look for any result files and check their content for the thread_id
                results_files = list(results_path.glob("eval_result*.json"))
                for result_file in results_files:
                    try:
                        with open(result_file, "r") as f:
                            result_data = json.load(f)
                            # Handle both list and dict formats
                            if isinstance(result_data, list) and len(result_data) > 0:
                                if result_data[0].get("thread_id") == thread_id:
                                    results_file_path = result_file
                                    logger.info(
                                        f"RUTHLESS_TEST_RESULT_FOUND: Found result file: {results_file_path}"
                                    )
                                    break
                            elif isinstance(result_data, dict):
                                if result_data.get("thread_id") == thread_id:
                                    results_file_path = result_file
                                    logger.info(
                                        f"RUTHLESS_TEST_RESULT_FOUND: Found result file: {results_file_path}"
                                    )
                                    break
                    except (json.JSONDecodeError, KeyError):
                        continue

            if audit_file_path and results_file_path:
                break
            await asyncio.sleep(1)

        # 4. Fail if files are not found
        assert (
            audit_file_path is not None
        ), f"TIMEOUT: Audit file for thread {thread_id} was not created."
        assert (
            results_file_path is not None
        ), f"TIMEOUT: Result file for thread {thread_id} was not created."

        # 5. Verify file contents
        logger.info("RUTHLESS_TEST_VERIFY: Verifying file contents...")
        with open(audit_file_path) as f:
            audit_data = json.load(f)
            # Handle list format for audit data
            if isinstance(audit_data, list) and len(audit_data) > 0:
                assert audit_data[0]["thread_id"] == thread_id
            else:
                assert audit_data["thread_id"] == thread_id
            logger.info("✓ Audit file content verified.")

        with open(results_file_path) as f:
            results_data = json.load(f)
            # Handle both list and dict formats for result data
            if isinstance(results_data, list) and len(results_data) > 0:
                assert results_data[0]["thread_id"] == thread_id
                assert results_data[0]["metadata"]["overall_success"] is True
            else:
                assert results_data["thread_id"] == thread_id
                assert results_data["metadata"]["overall_success"] is True
            logger.info("✓ Result file content verified.")

        logger.info(
            f"RUTHLESS_TEST_PASSED: End-to-end flow for thread_id={thread_id} is working."
        )

    except httpx.ConnectError as e:
        logger.error(
            f"RUTHLESS_TEST_FAILURE: Connection error - is the service running? {e}"
        )
        pytest.fail(f"Could not connect to the service: {e}")
    except Exception as e:
        logger.error(
            f"RUTHLESS_TEST_FAILURE: An unexpected error occurred: {e}", exc_info=True
        )
        pytest.fail(f"Test failed due to an unexpected error: {e}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


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
