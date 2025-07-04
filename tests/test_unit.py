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
import time
from typing import Dict, Any
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from .conftest import (
    MAIN_SERVICE_URL,
    ServiceTestError,
    ServiceUnavailableError,
    logger,
)

# Add evals-service to path for imports BEFORE any other imports
evals_service_path = os.path.join(os.path.dirname(__file__), "..", "evals-service")
if evals_service_path not in sys.path:
    sys.path.insert(0, evals_service_path)

# Try to import the app, skip if modules are not available
try:
    from app import app

    EVALS_SERVICE_AVAILABLE = True
except ImportError:
    EVALS_SERVICE_AVAILABLE = False


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
# UNIT TESTS FOR EVALUATION SERVICE
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(
    not EVALS_SERVICE_AVAILABLE, reason="Evaluation service dependencies not installed"
)
class TestEvaluationService:
    """Unit tests for the FastAPI evaluation service application."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Setup a test client with mocked state for the FastAPI app."""
        # Mock the components that the app relies on
        mock_evaluator = AsyncMock()
        mock_queue_manager = AsyncMock()
        mock_audit_storage = AsyncMock()
        mock_results_storage = AsyncMock()

        # Mock the health_check methods to return healthy by default
        mock_evaluator.health_check.return_value = {"evaluator_healthy": True}
        mock_queue_manager.health_check.return_value = {"queue_manager_healthy": True}
        mock_audit_storage.health_check.return_value = {"storage_manager_healthy": True}
        mock_results_storage.health_check.return_value = {
            "storage_manager_healthy": True
        }

        # Mock the get_metrics methods
        mock_evaluator.get_metrics.return_value = {"evals": 10}
        mock_queue_manager.get_metrics.return_value = {"queue_size": 5}
        mock_audit_storage.get_metrics.return_value = {"files_written": 2}
        mock_results_storage.get_metrics.return_value = {"files_written": 2}

        # Use monkeypatch to replace the real components in the app's state
        # with our mocks before the app starts up.
        monkeypatch.setattr(app.state, "evaluator", mock_evaluator)
        monkeypatch.setattr(app.state, "queue_manager", mock_queue_manager)
        monkeypatch.setattr(app.state, "audit_storage", mock_audit_storage)
        monkeypatch.setattr(app.state, "results_storage", mock_results_storage)

        with TestClient(app) as test_client:
            yield test_client

    def test_health_endpoint_healthy(self, client: TestClient):
        """Test the /health endpoint when all components are healthy."""
        logger.info("Testing /health endpoint with all components healthy")

        response = client.get("/health")

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "healthy"
        assert response_data["overall_healthy"] is True
        assert response_data["components"]["evaluator"]["evaluator_healthy"] is True
        assert (
            response_data["components"]["queue_manager"]["queue_manager_healthy"]
            is True
        )
        assert (
            response_data["components"]["audit_storage"]["storage_manager_healthy"]
            is True
        )
        assert (
            response_data["components"]["results_storage"]["storage_manager_healthy"]
            is True
        )

        logger.info("✓ /health endpoint returned 'healthy' as expected")

    def test_health_endpoint_unhealthy(self, client: TestClient, monkeypatch):
        """Test the /health endpoint when one component is unhealthy."""
        logger.info("Testing /health endpoint with one component unhealthy")

        # Make one of the components report as unhealthy
        unhealthy_mock = AsyncMock()
        unhealthy_mock.health_check.return_value = {
            "storage_manager_healthy": False,
            "error": "Disk full",
        }
        monkeypatch.setattr(app.state, "audit_storage", unhealthy_mock)

        response = client.get("/health")

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "unhealthy"
        assert response_data["overall_healthy"] is False
        assert (
            response_data["components"]["audit_storage"]["storage_manager_healthy"]
            is False
        )

        logger.info("✓ /health endpoint returned 'unhealthy' as expected")

    def test_metrics_endpoint(self, client: TestClient):
        """Test that the /metrics endpoint returns a valid structure."""
        logger.info("Testing /metrics endpoint")

        response = client.get("/metrics")

        assert response.status_code == 200
        response_data = response.json()

        # Check for the expected structure
        assert "service" in response_data
        assert "components" in response_data
        assert "configuration" in response_data
        assert response_data["components"]["queue_manager"]["queue_size"] == 5

        logger.info("✓ /metrics endpoint returned the correct structure")

    def test_evaluate_endpoint_async(
        self, client: TestClient, sample_evaluation_request: Dict[str, Any]
    ):
        """Test the async /evaluate endpoint."""
        logger.info("Testing async /evaluate endpoint")

        response = client.post("/evaluate", json=sample_evaluation_request)

        assert response.status_code == 202
        response_data = response.json()
        assert response_data["success"] is True
        assert (
            response_data["message"]
            == "Evaluation request accepted and is being processed."
        )
        assert response_data["thread_id"] == sample_evaluation_request["thread_id"]

        # Check that the request was passed to the queue manager mocks
        app.state.queue_manager.enqueue_audit.assert_called_once()
        app.state.queue_manager.enqueue_evaluation.assert_called_once()

        logger.info("✓ /evaluate endpoint correctly processed the request")

    def test_config_endpoint(self, client: TestClient):
        """Test the /config endpoint."""
        logger.info("Testing /config endpoint")
        response = client.get("/config")

        assert response.status_code == 200
        config_data = response.json()
        assert "service" in config_data
        assert "storage" in config_data
        assert "llm" in config_data
        assert "api" in config_data
        assert "openai_api_key" not in str(config_data)
        logger.info("✓ /config endpoint returned successfully and hid sensitive data")


# ============================================================================
# STORAGE BACKEND UNIT TESTS
# ============================================================================


@pytest.mark.unit
class TestStorageBackends:
    """Unit tests for storage backends and storage manager functionality"""

    # Test fixtures
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary directory for testing local storage"""
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_storage_data(self):
        """Sample data for storage testing"""
        return [
            {
                "run_id": "test-run-123",
                "thread_id": "test-thread-456",
                "message": "Test message 1",
                "timestamp": "2024-01-01T00:00:00Z",
            },
            {
                "run_id": "test-run-123",
                "thread_id": "test-thread-789",
                "message": "Test message 2",
                "timestamp": "2024-01-01T00:01:00Z",
            },
        ]

    @pytest.fixture
    def mock_gcs_backend(self):
        """Mock GCS backend for testing"""
        from unittest.mock import Mock, AsyncMock

        backend = Mock()
        backend.write_batch = AsyncMock(return_value=True)
        backend.health_check = AsyncMock(return_value=True)
        backend.get_metrics = AsyncMock(
            return_value={"backend_type": "gcs", "write_count": 0}
        )
        return backend

    # Helper fixture to mock google.cloud.storage when not installed
    @pytest.fixture(autouse=False)
    def mock_google_cloud_storage(self):
        """Inject a dummy google.cloud.storage module into sys.modules so that
        tests can import GCSStorageBackend without the real library."""
        import types
        import sys
        from unittest.mock import Mock

        # Create dummy modules
        google_mod = types.ModuleType("google")
        cloud_mod = types.ModuleType("google.cloud")
        storage_mod = types.ModuleType("google.cloud.storage")
        api_core_mod = types.ModuleType("google.api_core")
        exceptions_mod = types.ModuleType("google.api_core.exceptions")

        # Define dummy GoogleAPICallError
        class _GoogleAPICallError(Exception):
            pass

        exceptions_mod.GoogleAPICallError = _GoogleAPICallError

        # Add minimal attributes used in storage.GCSStorageBackend
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_blob.upload_from_string = Mock()
        mock_bucket.blob.return_value = mock_blob
        mock_client.bucket.return_value = mock_bucket
        mock_client.lookup_bucket = Mock(return_value=mock_bucket)
        storage_mod.Client = Mock(return_value=mock_client)

        # Insert into sys.modules
        sys.modules.setdefault("google", google_mod)
        sys.modules.setdefault("google.cloud", cloud_mod)
        sys.modules.setdefault("google.cloud.storage", storage_mod)
        sys.modules.setdefault("google.api_core", api_core_mod)
        sys.modules.setdefault("google.api_core.exceptions", exceptions_mod)

        yield {
            "client": mock_client,
            "bucket": mock_bucket,
            "blob": mock_blob,
        }

        # Clean up mocked modules
        for mod_name in [
            "google.api_core.exceptions",
            "google.api_core",
            "google.cloud.storage",
            "google.cloud",
            "google",
        ]:
            if mod_name in sys.modules:
                del sys.modules[mod_name]

    # Local Storage Backend Tests
    @pytest.mark.asyncio
    async def test_local_storage_write_batch_success(
        self, temp_storage_path, sample_storage_data
    ):
        """Test successful batch write to local storage"""
        try:
            from storage import LocalStorageBackend
            import json
            from pathlib import Path

            backend = LocalStorageBackend(temp_storage_path)
            filename = "test_batch.json"

            success = await backend.write_batch(sample_storage_data, filename)

            assert success is True
            assert backend._write_count == 1
            assert backend._error_count == 0

            # Verify file was created and content is correct
            file_path = Path(temp_storage_path) / filename
            assert file_path.exists()

            with open(file_path, "r") as f:
                saved_data = json.load(f)

            assert saved_data == sample_storage_data
            logger.info("✓ Local storage batch write test passed")

        except ImportError as e:
            logger.warning(f"Storage module not available: {e}")
            pytest.skip("Storage module not available for testing")

    @pytest.mark.asyncio
    async def test_local_storage_health_check(self, temp_storage_path):
        """Test local storage health check"""
        try:
            from storage import LocalStorageBackend

            backend = LocalStorageBackend(temp_storage_path)
            health = await backend.health_check()
            assert health is True
            logger.info("✓ Local storage health check test passed")

        except ImportError as e:
            logger.warning(f"Storage module not available: {e}")
            pytest.skip("Storage module not available for testing")

    @pytest.mark.asyncio
    async def test_local_storage_get_metrics(
        self, temp_storage_path, sample_storage_data
    ):
        """Test getting local storage metrics"""
        try:
            from storage import LocalStorageBackend

            backend = LocalStorageBackend(temp_storage_path)

            # Write some test data first
            await backend.write_batch(sample_storage_data, "test1.json")
            await backend.write_batch(sample_storage_data, "test2.json")

            metrics = await backend.get_metrics()

            assert metrics["backend_type"] == "local"
            assert metrics["base_path"] == str(backend.base_path)
            assert metrics["total_files"] == 2
            assert metrics["write_count"] == 2
            assert metrics["error_count"] == 0
            assert "total_size_bytes" in metrics
            logger.info("✓ Local storage metrics test passed")

        except ImportError as e:
            logger.warning(f"Storage module not available: {e}")
            pytest.skip("Storage module not available for testing")

    # GCS Storage Backend Tests
    @pytest.mark.asyncio
    async def test_gcs_storage_init_success(self, mock_google_cloud_storage):
        """Test GCS backend initialization"""
        from storage import GCSStorageBackend

        backend = GCSStorageBackend("test-bucket")
        assert backend.bucket_name == "test-bucket"
        assert backend.storage_client is not None
        assert backend._write_count == 0
        assert backend._error_count == 0
        logger.info("✓ GCS backend initialization test passed")

    @pytest.mark.asyncio
    async def test_gcs_storage_write_batch_success(
        self, mock_google_cloud_storage, sample_storage_data
    ):
        """Test successful GCS batch write"""
        from storage import GCSStorageBackend

        backend = GCSStorageBackend("test-bucket")
        success = await backend.write_batch(sample_storage_data, "test.json")

        assert success is True
        assert backend._write_count == 1
        assert backend._error_count == 0
        mock_google_cloud_storage["blob"].upload_from_string.assert_called_once()
        logger.info("✓ GCS batch write test passed")

    @pytest.mark.asyncio
    async def test_gcs_storage_health_check(self, mock_google_cloud_storage):
        """Test GCS storage health check"""
        from storage import GCSStorageBackend

        backend = GCSStorageBackend("test-bucket")
        health = await backend.health_check()

        assert health is True
        logger.info("✓ GCS health check test passed")

    # Storage Factory Tests
    @pytest.mark.asyncio
    async def test_storage_backend_factory_local_default(self):
        """Test storage factory creates local backend by default"""
        try:
            from unittest.mock import patch
            from storage import create_storage_backend, LocalStorageBackend

            with patch.dict(os.environ, {}, clear=True):
                backend = create_storage_backend()
                assert isinstance(backend, LocalStorageBackend)
                logger.info("✓ Storage factory local default test passed")

        except ImportError as e:
            logger.warning(f"Storage module not available: {e}")
            pytest.skip("Storage module not available for testing")

    @pytest.mark.asyncio
    async def test_storage_backend_factory_gcs_configured(
        self, mock_google_cloud_storage
    ):
        """Test storage factory creates GCS backend when properly configured"""
        import importlib
        import sys
        from unittest.mock import patch

        with patch.dict(
            os.environ,
            {
                "STORAGE_STORAGE_BACKEND": "gcs",
                "STORAGE_GCS_BUCKET_NAME": "test-bucket",
            },
            clear=False,
        ):
            # Reload config and storage modules to pick up new env vars
            if "config" in sys.modules:
                importlib.reload(sys.modules["config"])
            if "storage" in sys.modules:
                importlib.reload(sys.modules["storage"])

            from storage import (
                create_storage_backend,
                GCSStorageBackend,
            )  # Import after reload

            backend = create_storage_backend()
            assert isinstance(backend, GCSStorageBackend)
            assert backend.bucket_name == "test-bucket"
            logger.info("✓ Storage factory GCS configuration test passed")

    @pytest.mark.asyncio
    async def test_storage_backend_factory_gcs_missing_bucket(
        self, mock_google_cloud_storage
    ):
        """Test storage factory falls back to local when GCS bucket is missing"""
        import importlib
        import sys
        from unittest.mock import patch

        with patch.dict(
            os.environ,
            {
                "STORAGE_STORAGE_BACKEND": "gcs"
                # Missing STORAGE_GCS_BUCKET_NAME
            },
            clear=False,
        ):
            # Reload config and storage modules to pick up new env vars
            if "config" in sys.modules:
                importlib.reload(sys.modules["config"])
            if "storage" in sys.modules:
                importlib.reload(sys.modules["storage"])

            from storage import create_storage_backend, LocalStorageBackend

            backend = create_storage_backend()
            assert isinstance(backend, LocalStorageBackend)
            logger.info(
                "✓ Storage factory fallback to local when bucket missing test passed"
            )

    # Integration test for real GCS (optional - requires credentials)
    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("STORAGE_GCS_BUCKET_NAME")
        or not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        reason="GCS credentials not available",
    )
    @pytest.mark.asyncio
    async def test_real_gcs_integration(self):
        """Test GCS integration with real credentials (when available)"""
        try:
            from storage import GCSStorageBackend

            bucket_name = os.getenv("STORAGE_GCS_BUCKET_NAME")
            backend = GCSStorageBackend(bucket_name)

            if backend.storage_client:
                # Test health check
                health = await backend.health_check()
                assert (
                    health is True
                ), "GCS health check should pass with valid credentials"

                # Test write operation
                test_data = [
                    {
                        "run_id": f"test-integration-{int(time.time())}",
                        "message": "Integration test data from unit tests",
                        "timestamp": "2024-01-01T00:00:00Z",
                    }
                ]

                success = await backend.write_batch(
                    test_data, "unit_test_integration.json"
                )
                assert (
                    success is True
                ), "GCS write should succeed with valid credentials"

                logger.info("✓ Real GCS integration test passed")
            else:
                pytest.skip("GCS client not available")

        except ImportError as e:
            logger.warning(f"Storage module not available: {e}")
            pytest.skip("Storage module not available for testing")


# ============================================================================
# CONFIGURATION UNIT TESTS
# ============================================================================
