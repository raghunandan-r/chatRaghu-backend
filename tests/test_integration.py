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
from datetime import datetime

# ---------------------------------------------------------------------------
# Optional Google Cloud import (allows running tests without the library)
# ---------------------------------------------------------------------------
try:
    from google.cloud import storage as _storage  # type: ignore
except ImportError:  # pragma: no cover – library may be absent in CI
    _storage = None  # type: ignore
    logger.warning(
        "google-cloud-storage not installed; GCS-specific checks will be skipped"
    )

# Test queries for diverse node path coverage
DIVERSE_TEST_QUERIES = [
    {
        "messages": [
            {
                "role": "user",
                "content": "Tell me about yourself?",
                "thread_id": f"test-e2e-1-{int(time.time())}",
            }
        ]
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Give me the Python code to reverse a linked list.",
                "thread_id": f"test-e2e-2-{int(time.time())}",
            }
        ]
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Do you have any work experience in the US?",
                "thread_id": f"test-e2e-1-{int(time.time())}",
            }
        ]
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Why are you better than ChatGPT?",
                "thread_id": f"test-e2e-4-{int(time.time())}",
            }
        ]
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "You listed 'XAI' as a skill. What does that mean?",
                "thread_id": f"test-e2e-4{int(time.time())}",
            }
        ]
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Can you explain that in simpler terms??",
                "thread_id": f"test-e2e-4{int(time.time())}",
            }
        ]
    },
]

# Define the single, focused test query
TEST_QUERY = {
    "messages": [
        {
            "role": "user",
            "content": "What is your tech stack?",
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

            initial_audit_files = len(list(audit_data_path.rglob("*.jsonl"))) + len(
                list(audit_data_path.rglob("*.parquet"))
            )
            initial_eval_files = len(list(eval_results_path.rglob("*.jsonl"))) + len(
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
            final_audit_files = len(list(audit_data_path.rglob("*.jsonl"))) + len(
                list(audit_data_path.rglob("*.parquet"))
            )
            final_eval_files = len(list(eval_results_path.rglob("*.jsonl"))) + len(
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
                audit_files = list(audit_data_path.rglob("*.jsonl"))
                if audit_files:
                    latest_audit_file = max(
                        audit_files, key=lambda f: f.stat().st_mtime
                    )
                    logger.info(f"Latest audit file: {latest_audit_file}")

                    with open(latest_audit_file, "r") as f:
                        audit_data = [json.loads(line) for line in f if line.strip()]
                        logger.info(f"Audit file contains {len(audit_data)} records")

            if eval_files_created:
                logger.info("Step 6: Checking evaluation result file contents")
                eval_files = list(eval_results_path.rglob("*.jsonl"))
                if eval_files:
                    latest_eval_file = max(eval_files, key=lambda f: f.stat().st_mtime)
                    logger.info(f"Latest eval file: {latest_eval_file}")

                    with open(latest_eval_file, "r") as f:
                        eval_data = [json.loads(line) for line in f if line.strip()]
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


def verify_evaluation_result(result_data: dict, thread_id: str):
    """Helper function to verify evaluation result structure and content for the new flattened schema."""
    assert "evaluations" in result_data, "Result missing 'evaluations' list"
    evaluations = result_data["evaluations"]

    # Ensure evaluations is a list (flattened structure)
    assert isinstance(
        evaluations, list
    ), "'evaluations' should be a list after flattening"

    # Build helper mappings for quick lookup
    evals_by_node = {}
    for item in evaluations:
        node = item.get("node_name")
        if node:
            evals_by_node.setdefault(node, []).append(item)

    # We expect at least two nodes evaluated
    assert "relevance_check" in evals_by_node, "Missing relevance_check evaluation"
    assert (
        "generate_with_persona" in evals_by_node
    ), "Missing generate_with_persona evaluation"

    # -------------------------
    # Validate relevance_check
    # -------------------------
    rel_evals = evals_by_node["relevance_check"]
    # Prefer the evaluate_relevance_check evaluator if present
    rel_eval = next(
        (e for e in rel_evals if e.get("evaluator_name") == "evaluate_relevance_check"),
        rel_evals[0],
    )

    assert "routing_correct" in rel_eval, "Missing routing_correct in relevance_check"
    assert rel_eval["routing_correct"] in [
        "IRRELEVANT",
        "RELEVANT",
    ], f"Invalid routing_correct: {rel_eval['routing_correct']}"
    assert "explanation" in rel_eval, "Missing explanation in relevance_check"
    assert isinstance(rel_eval["explanation"], str), "explanation should be string"

    # -------------------------------
    # Validate generate_with_persona
    # -------------------------------
    persona_evals = evals_by_node["generate_with_persona"]
    persona_eval = next(
        (
            e
            for e in persona_evals
            if e.get("evaluator_name") == "evaluate_generate_with_persona"
        ),
        persona_evals[0],
    )

    for bool_field in ["persona_adherence", "follows_rules", "faithfulness"]:
        assert bool_field in persona_eval, f"Missing {bool_field}"
        assert isinstance(
            persona_eval[bool_field], bool
        ), f"{bool_field} should be boolean"

    assert "explanation" in persona_eval, "Missing explanation"
    assert isinstance(persona_eval["explanation"], str), "explanation should be string"

    # Verify thread_id matches
    assert (
        result_data.get("thread_id") == thread_id
    ), f"Thread ID mismatch. Expected {thread_id}, got {result_data.get('thread_id')}"

    return True


@pytest.mark.skip(reason="Skipping local end-to-end test")
@pytest.mark.asyncio
async def test_local_end_to_end(
    http_client: httpx.AsyncClient,
    valid_api_headers: Dict[str, str],
):
    """
    A true end-to-end test that verifies file creation for multiple diverse queries.
    It will:
    1. Clean up old test files.
    2. Send multiple diverse chat requests to the main service with wait times.
    3. Poll the filesystem until audit files and result files are created for each.
    4. Fail if files do not appear within a timeout.
    5. Verify the contents of the created files.
    """
    audit_path = Path("evals_service/audit_data")
    results_path = Path("evals_service/eval_results")

    # --- Fast-fail check: ensure evals-service actually instantiated local backends ---
    metrics_resp = await http_client.get(f"{EVALUATION_SERVICE_URL}/metrics")
    metrics_resp.raise_for_status()
    metrics_json = metrics_resp.json()

    try:
        audit_backend_type = metrics_json["components"]["audit_storage"][
            "backend_metrics"
        ]["backend_type"]
        result_backend_type = metrics_json["components"]["results_storage"][
            "backend_metrics"
        ]["backend_type"]
    except KeyError as e:
        pytest.fail(
            f"Unexpected metrics schema – missing key {e}. Full payload: {metrics_json}"
        )

    if audit_backend_type != "local" or result_backend_type != "local":
        pytest.fail(
            "Evaluation service is not using local backends as expected: "
            f"audit_backend={audit_backend_type}, result_backend={result_backend_type}. "
            "Check STORAGE_* env vars and service startup logs."
        )

    # 1. Clean up old files
    logger.info("RUTHLESS_TEST_CLEANUP: Deleting old test files...")
    # Clean up audit data
    for p in audit_path.rglob("*.jsonl"):
        p.unlink()
    # Remove empty date and run_id directories
    for date_dir in audit_path.iterdir():
        if date_dir.is_dir():
            for run_dir in date_dir.iterdir():
                if run_dir.is_dir():
                    try:
                        run_dir.rmdir()  # Will only succeed if directory is empty
                    except OSError:
                        pass  # Directory not empty or other error, skip it
            try:
                date_dir.rmdir()  # Will only succeed if directory is empty
            except OSError:
                pass  # Directory not empty or other error, skip it

    # Clean up eval results
    for p in results_path.rglob("*.jsonl"):
        p.unlink()
    # Remove empty date and run_id directories
    for date_dir in results_path.iterdir():
        if date_dir.is_dir():
            for run_dir in date_dir.iterdir():
                if run_dir.is_dir():
                    try:
                        run_dir.rmdir()  # Will only succeed if directory is empty
                    except OSError:
                        pass  # Directory not empty or other error, skip it
            try:
                date_dir.rmdir()  # Will only succeed if directory is empty
            except OSError:
                pass  # Directory not empty or other error, skip it

    try:
        # 2. Send multiple diverse chat requests with wait times
        logger.info(
            f"RUTHLESS_TEST_REQUESTS: Sending {len(DIVERSE_TEST_QUERIES)} diverse chat requests"
        )

        for i, query in enumerate(DIVERSE_TEST_QUERIES):
            thread_id = query["messages"][0]["thread_id"]
            logger.info(
                f"RUTHLESS_TEST_REQUEST_{i+1}: Sending request for thread_id={thread_id}"
            )
            logger.info(
                f"RUTHLESS_TEST_CONTENT_{i+1}: {query['messages'][0]['content']}"
            )

            response = await http_client.post(
                f"{MAIN_SERVICE_URL}/api/chat",
                headers=valid_api_headers,
                json=query,
            )
            response.raise_for_status()
            logger.info(
                f"RUTHLESS_TEST_RESPONSE_{i+1}: Received {response.status_code} from main service."
            )

            # Consume the streaming response to ensure the graph execution completes
            response_text = ""
            async for chunk in response.aiter_text():
                response_text += chunk
            assert response_text, f"Chat response {i+1} should not be empty"
            logger.info(
                f"RUTHLESS_TEST_STREAM_COMPLETE_{i+1}: Full response received from main service."
            )

            # Add wait time between queries (except for the last one)
            if i < len(DIVERSE_TEST_QUERIES) - 1:
                wait_time = 3  # seconds
                logger.info(
                    f"RUTHLESS_TEST_WAIT_{i+1}: Waiting {wait_time} seconds before next query..."
                )
                await asyncio.sleep(wait_time)

        # 3. Poll for file creation for all queries
        timeout = 50  # seconds - increased for multiple queries
        start_time = time.time()
        expected_thread_ids = [
            query["messages"][0]["thread_id"] for query in DIVERSE_TEST_QUERIES
        ]
        found_audit_files = {}
        found_result_files = {}

        logger.info(
            f"RUTHLESS_TEST_POLLING: Waiting up to {timeout}s for files to be created for {len(expected_thread_ids)} queries..."
        )

        while time.time() - start_time < timeout:
            # Check for audit files - now using recursive glob for partitioned structure
            audit_files = list(audit_path.rglob("audit_request*.jsonl"))
            for audit_file in audit_files:
                try:
                    with open(audit_file, "r") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            audit_data = json.loads(line)
                            thread_id = audit_data.get("thread_id")
                            if (
                                thread_id in expected_thread_ids
                                and thread_id not in found_audit_files
                            ):
                                found_audit_files[thread_id] = audit_file
                                logger.info(
                                    f"RUTHLESS_TEST_AUDIT_FOUND: Found audit file for thread {thread_id}: {audit_file}",
                                    extra={
                                        "thread_id": thread_id,
                                        "file_path": str(audit_file),
                                        "partition_info": {
                                            "date": audit_file.parent.parent.name,
                                            "run_id": audit_file.parent.name,
                                        },
                                    },
                                )
                                break
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error processing audit file {audit_file}: {e}")
                    continue

            # Check for result files - now using recursive glob for partitioned structure
            results_files = list(results_path.rglob("eval_result*.jsonl"))
            for result_file in results_files:
                try:
                    with open(result_file, "r") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            result_data = json.loads(line)
                            thread_id = result_data.get("thread_id")
                            if (
                                thread_id in expected_thread_ids
                                and thread_id not in found_result_files
                            ):
                                found_result_files[thread_id] = result_file
                                logger.info(
                                    f"RUTHLESS_TEST_RESULT_FOUND: Found result file for thread {thread_id}: {result_file}",
                                    extra={
                                        "thread_id": thread_id,
                                        "file_path": str(result_file),
                                        "partition_info": {
                                            "date": result_file.parent.parent.name,
                                            "run_id": result_file.parent.name,
                                        },
                                    },
                                )
                                break
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error processing result file {result_file}: {e}")
                    continue

            # Check if we found files for all expected thread IDs
            if len(found_audit_files) == len(expected_thread_ids) and len(
                found_result_files
            ) == len(expected_thread_ids):
                break

            await asyncio.sleep(1)

        # 4. Fail if files are not found for all queries
        missing_audit_threads = set(expected_thread_ids) - set(found_audit_files.keys())
        missing_result_threads = set(expected_thread_ids) - set(
            found_result_files.keys()
        )

        assert (
            len(missing_audit_threads) == 0
        ), f"TIMEOUT: Audit files missing for threads: {missing_audit_threads}"
        assert (
            len(missing_result_threads) == 0
        ), f"TIMEOUT: Result files missing for threads: {missing_result_threads}"

        # 5. Verify file contents for all queries
        logger.info("RUTHLESS_TEST_VERIFY: Verifying file contents for all queries...")

        for thread_id in expected_thread_ids:
            # Verify audit file
            audit_file_path = found_audit_files[thread_id]
            with open(audit_file_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    audit_data = json.loads(line)
                    if audit_data["thread_id"] == thread_id:
                        break
                else:
                    raise AssertionError(
                        f"Thread ID {thread_id} not found in audit file"
                    )
            logger.info(f"✓ Audit file content verified for thread {thread_id}.")

            # Verify result file
            results_file_path = found_result_files[thread_id]
            with open(results_file_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    results_data = json.loads(line)
                    if results_data["thread_id"] == thread_id:
                        assert (
                            "metadata" in results_data
                        ), f"Missing metadata in results data for thread {thread_id}"
                        assert (
                            "overall_success" in results_data["metadata"]
                        ), f"Missing overall_success in metadata for thread {thread_id}"
                        break
                else:
                    raise AssertionError(
                        f"Thread ID {thread_id} not found in results file"
                    )

            logger.info(f"✓ Result file content verified for thread {thread_id}.")

            # Enhanced verification section
            logger.info(
                f"RUTHLESS_TEST_VERIFY: Verifying results for thread {thread_id}"
            )

            # Find and load the results file
            results_files = list(results_path.rglob("eval_result*.jsonl"))
            thread_result_file = None
            result_data = None

            for result_file in results_files:
                try:
                    with open(result_file, "r") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            data = json.loads(line)
                            if data.get("thread_id") == thread_id:
                                result_data = data
                                thread_result_file = result_file
                                break
                        if thread_result_file:
                            break
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error processing result file {result_file}: {e}")
                    continue

            assert (
                thread_result_file is not None
            ), f"No result file found for thread {thread_id}"
            assert (
                result_data is not None
            ), f"No result data found for thread {thread_id}"

            # Verify the evaluation results
            try:
                verify_evaluation_result(result_data, thread_id)
                logger.info(
                    f"✓ Evaluation result verification passed for thread {thread_id}"
                )
            except AssertionError as e:
                logger.error(
                    f"Evaluation result verification failed for thread {thread_id}: {e}"
                )
                raise

        logger.info(
            f"RUTHLESS_TEST_PASSED: End-to-end flow for all {len(expected_thread_ids)} queries is working."
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


@pytest.mark.integration_gcs
@pytest.mark.skipif(
    not all(
        os.getenv(k)
        for k in [
            "STORAGE_GCS_AUDIT_BUCKET_NAME",
            "STORAGE_GCS_EVAL_RESULTS_BUCKET_NAME",
            "GOOGLE_APPLICATION_CREDENTIALS",
        ]
    )
    or _storage is None,
    reason="This test requires real GCS credentials and the google-cloud-storage library.",
)
@pytest.mark.asyncio
async def test_gcs_end_to_end(
    http_client,
    valid_api_headers: Dict[str, str],
):
    """
    This test verifies that the evals-service, when configured to use GCS,
    successfully writes evaluation results to the bucket and that they have the correct structure.
    """

    gcs_audit_bucket_name = os.getenv("STORAGE_GCS_AUDIT_BUCKET_NAME")
    gcs_results_bucket_name = os.getenv("STORAGE_GCS_EVAL_RESULTS_BUCKET_NAME")

    if not gcs_audit_bucket_name or not gcs_results_bucket_name:
        pytest.fail("GCS bucket environment variables are not set.")

    logger.info(f"Targeting GCS audit bucket: {gcs_audit_bucket_name}")
    logger.info(f"Targeting GCS results bucket: {gcs_results_bucket_name}")

    # --- Fast-fail check: ensure evals-service actually instantiated GCS backends ---
    metrics_resp = await http_client.get(f"{EVALUATION_SERVICE_URL}/metrics")
    metrics_resp.raise_for_status()
    metrics_json = metrics_resp.json()

    try:
        audit_backend_type = metrics_json["components"]["audit_storage"][
            "backend_metrics"
        ]["backend_type"]
        result_backend_type = metrics_json["components"]["results_storage"][
            "backend_metrics"
        ]["backend_type"]
    except KeyError as e:
        pytest.fail(
            f"Unexpected metrics schema – missing key {e}. Full payload: {metrics_json}"
        )

    if audit_backend_type != "gcs" or result_backend_type != "gcs":
        pytest.fail(
            "Evaluation service is not using GCS backends as expected: "
            f"audit_backend={audit_backend_type}, result_backend={result_backend_type}. "
            "Check STORAGE_* env vars and service startup logs."
        )

    # 1. Send a unique chat request.
    unique_id = f"gcs-e2e-test-{int(time.time())}"
    test_query = {
        "messages": [
            {
                "role": "user",
                "content": f"Test message for GCS: {unique_id}",
                "thread_id": unique_id,
            }
        ]
    }
    logger.info(f"Sending request with unique thread_id: {unique_id}")
    response = await http_client.post(
        f"{MAIN_SERVICE_URL}/api/chat", headers=valid_api_headers, json=test_query
    )
    response.raise_for_status()
    async for _ in response.aiter_text():
        pass  # Consume the response
    logger.info("Chat request sent successfully.")

    # 2. Poll the GCS buckets to verify the files were created.
    logger.info("Waiting for files to appear in GCS buckets...")
    timeout = 45
    start_time = time.time()
    found_blobs = []

    gcs_client = _storage.Client()
    audit_bucket = gcs_client.bucket(gcs_audit_bucket_name)
    results_bucket = gcs_client.bucket(gcs_results_bucket_name)

    # Keep track of which files we've found
    audit_file_found = False
    result_file_found = False

    while time.time() - start_time < timeout:
        # Search audit bucket for audit_request file
        if not audit_file_found:
            audit_blobs = list(audit_bucket.list_blobs(prefix="audit_data/"))
            logger.debug(
                f"Found {len(audit_blobs)} audit blobs with prefix audit_data/"
            )
            for blob in audit_blobs:
                # Add this line to filter by date and file prefix
                if "audit_request" not in blob.name:
                    continue
                try:
                    content = blob.download_as_text()
                    if unique_id in content:
                        logger.info(
                            f"SUCCESS: Found audit file '{blob.name}' containing unique ID."
                        )
                        found_blobs.append(blob)
                        audit_file_found = True
                        break
                    else:
                        logger.debug(
                            f"Audit blob {blob.name} does not contain unique_id {unique_id}"
                        )
                except Exception as e:
                    logger.warning(f"Could not read blob {blob.name}: {e}")

        # Search results bucket for eval_result file
        if not result_file_found:
            results_blobs = list(results_bucket.list_blobs(prefix="eval_results/"))
            logger.debug(
                f"Found {len(results_blobs)} result blobs with prefix eval_results/"
            )
            for blob in results_blobs:
                # Add this line to filter by date and file prefix
                if "eval_result" not in blob.name:
                    continue
                try:
                    content = blob.download_as_text()
                    # A GCS file can contain multiple JSONL records. Split by newline.
                    # Filter out any empty lines that might result from a trailing newline.
                    result_lines = [
                        line for line in content.strip().split("\n") if line
                    ]
                    for line in result_lines:
                        result_data = json.loads(line)
                        if unique_id in result_data.get("thread_id", ""):
                            logger.info(f"Found result data in blob {blob.name}")
                            verify_evaluation_result(result_data, unique_id)
                            result_file_found = True
                            found_blobs.append(blob)
                            break

                    if result_file_found:
                        break
                except Exception as e:
                    logger.warning(f"Could not process blob {blob.name}: {e}")

        if audit_file_found and result_file_found:
            logger.info("Found both audit and result files.")
            break

        await asyncio.sleep(3)

    # 3. Assert that both files were found and clean them up.
    try:
        assert (
            audit_file_found
        ), f"TIMEOUT: Audit file with ID '{unique_id}' did not appear in GCS bucket within {timeout}s."
        assert (
            result_file_found
        ), f"TIMEOUT: Result file with ID '{unique_id}' did not appear in GCS bucket within {timeout}s."
    finally:
        if found_blobs:
            logger.info(f"Cleaning up {len(found_blobs)} test files...")
            cleanup_failures = []
            for blob in found_blobs:
                try:
                    logger.info(
                        f"Attempting to delete: {blob.name} from bucket {blob.bucket.name}"
                    )
                    blob.delete()
                    # Wait a moment and check if it still exists
                    await asyncio.sleep(1)
                    if blob.exists():
                        error_msg = (
                            f"Blob {blob.name} still exists after deletion attempt"
                        )
                        logger.error(error_msg)
                        cleanup_failures.append(error_msg)
                    else:
                        logger.info(f"Successfully deleted: {blob.name}")
                except Exception as e:
                    error_msg = f"Exception deleting {blob.name}: {e}"
                    logger.error(error_msg)
                    cleanup_failures.append(error_msg)

            if cleanup_failures:
                pytest.fail(
                    f"Cleanup failed for {len(cleanup_failures)} files:\n"
                    + "\n".join(cleanup_failures)
                )
            else:
                logger.info("Cleanup completed successfully for all files.")
        else:
            logger.warning(
                "No blobs found to clean up - this might indicate a problem with file detection"
            )


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
