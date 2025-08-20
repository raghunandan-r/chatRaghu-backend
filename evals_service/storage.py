import asyncio
import os
from datetime import datetime, timezone
from typing import List, Protocol, Optional, Dict, Any
from pathlib import Path
from utils.logger import logger
from config import config
import json


class StorageBackend(Protocol):
    """Protocol for storage backends"""

    async def write_batch(self, data: List[Dict[str, Any]], filename: str) -> bool:
        """Write a batch of data to storage"""
        ...

    async def health_check(self) -> bool:
        """Check if the storage backend is healthy"""
        ...

    async def get_metrics(self) -> Dict[str, Any]:
        """Get storage backend metrics"""
        ...


class LocalStorageBackend:
    """Local file system storage backend"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._write_count = 0
        self._error_count = 0

    async def write_batch(self, data: List[Dict[str, Any]], filename: str) -> bool:
        """Write batch data to JSONL file"""
        try:
            # Ensure filename has .jsonl extension
            if not filename.endswith(".jsonl"):
                filename = f"{filename}.jsonl"

            json_path = self.base_path / filename

            # Ensure directory exists
            json_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to JSONL with one JSON object per line
            with open(json_path, "w", encoding="utf-8") as f:
                for record in data:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

            self._write_count += 1
            logger.info(
                f"Successfully wrote {len(data)} records to {filename}",
                extra={
                    "json_path": str(json_path),
                    "record_count": len(data),
                    "json_size_bytes": json_path.stat().st_size
                    if json_path.exists()
                    else 0,
                },
            )
            return True

        except Exception as e:
            self._error_count += 1
            logger.error(
                "Failed to write batch to local storage",
                extra={
                    "error": str(e),
                    "file_name": filename,
                    "record_count": len(data),
                },
            )
            return False

    async def health_check(self) -> bool:
        """Check if local storage is accessible"""
        try:
            # Test write access
            test_file = self.base_path / ".health_check"
            test_file.write_text("health_check")
            test_file.unlink()  # Clean up
            return True
        except Exception as e:
            logger.error("Local storage health check failed", extra={"error": str(e)})
            return False

    async def get_metrics(self) -> Dict[str, Any]:
        """Get local storage metrics"""
        try:
            # Update to look for .jsonl files instead of .json
            json_files = list(self.base_path.rglob("*.jsonl"))
            total_json_files = len(json_files)
            total_json_size = sum(f.stat().st_size for f in json_files)

            return {
                "backend_type": "local",
                "base_path": str(self.base_path),
                "total_files": total_json_files,
                "total_size_bytes": total_json_size,
                "write_count": self._write_count,
                "error_count": self._error_count,
                "available_space_bytes": self._get_available_space(),
            }
        except Exception as e:
            logger.error("Failed to get local storage metrics", extra={"error": str(e)})
            return {"backend_type": "local", "error": str(e)}

    def _get_available_space(self) -> int:
        """Get available disk space"""
        try:
            return (
                os.statvfs(self.base_path).f_frsize
                * os.statvfs(self.base_path).f_bavail
            )
        except Exception:
            return 0


class GCSStorageBackend:
    """Google Cloud Storage backend"""

    def __init__(self, bucket_name: str):
        try:
            from google.cloud import storage
            from google.api_core.exceptions import GoogleAPICallError

            self.storage_client = storage.Client()
            self.bucket_name = bucket_name
            self.bucket = self.storage_client.bucket(self.bucket_name)
            self.GoogleAPICallError = GoogleAPICallError
        except ImportError:
            logger.error(
                "GCS client libraries not found. Please `pip install google-cloud-storage`"
            )
            self.storage_client = None

        self._write_count = 0
        self._error_count = 0

    async def write_batch(self, data: List[Dict[str, Any]], filename: str) -> bool:
        """Write batch data to a GCS blob in JSONL format."""
        if not self.storage_client:
            self._error_count += 1
            return False

        try:
            blob = self.bucket.blob(filename)

            # Create a single string in JSONL format (one JSON object per line)
            jsonl_data = "\n".join(
                json.dumps(record, ensure_ascii=False, default=str) for record in data
            )

            # Use a non-blocking call for the upload
            await asyncio.to_thread(
                blob.upload_from_string,
                data=jsonl_data,
                content_type="application/x-ndjson",
            )

            self._write_count += 1
            logger.info(
                f"Successfully wrote {len(data)} records to GCS",
                extra={
                    "gcs_path": f"gs://{self.bucket_name}/{filename}",
                    "record_count": len(data),
                },
            )
            return True

        except self.GoogleAPICallError as e:
            self._error_count += 1
            logger.error(
                "Failed to write batch to GCS",
                extra={"error": str(e), "file_name": filename},
            )
            return False

    async def health_check(self) -> bool:
        """Check GCS connectivity by checking bucket existence."""
        if not self.storage_client:
            return False
        try:
            # Check if bucket exists and we have permissions
            return (
                await asyncio.to_thread(
                    self.storage_client.lookup_bucket, self.bucket_name
                )
                is not None
            )
        except self.GoogleAPICallError as e:
            logger.error("GCS health check failed", extra={"error": str(e)})
            return False

    async def get_metrics(self) -> Dict[str, Any]:
        """Get GCS storage metrics."""
        return {
            "backend_type": "gcs",
            "bucket_name": self.bucket_name,
            "write_count": self._write_count,
            "error_count": self._error_count,
        }


class StorageManager:
    """
    Manages batching items from an asyncio queue and writing them to storage.
    Supports multiple storage backends and provides health monitoring.
    """

    def __init__(
        self,
        queue: asyncio.Queue,
        storage_backend: StorageBackend,
        file_prefix: str,
        path_prefix: Optional[str] = None,
        batch_size: int = None,
        write_timeout: float = None,
    ):
        self.queue = queue
        self.storage_backend = storage_backend
        self.file_prefix = file_prefix
        self.path_prefix = path_prefix
        self.batch_size = batch_size or config.storage.batch_size
        self.write_timeout = write_timeout or config.storage.write_timeout_seconds
        self._worker_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._processed_count = 0
        self._error_count = 0

        logger.info(
            f"Initialized StorageManager for {file_prefix}",
            extra={
                "batch_size": self.batch_size,
                "write_timeout": self.write_timeout,
                "backend_type": type(storage_backend).__name__,
            },
        )

    async def _storage_worker(self):
        """Background worker to store data in batches."""
        logger.info(f"Starting storage worker for {self.file_prefix}")
        self._is_running = True

        while self._is_running:
            try:
                items_to_write = await self._collect_batch()

                if items_to_write:
                    success = await self._write_batch(items_to_write)
                    if success:
                        self._processed_count += len(items_to_write)
                    else:
                        self._error_count += len(items_to_write)

            except asyncio.CancelledError:
                logger.info(f"Storage worker for {self.file_prefix} is shutting down.")
                break
            except Exception as e:
                self._error_count += 1
                logger.error(
                    "Error in storage worker",
                    extra={
                        "error": str(e),
                        "prefix": self.file_prefix,
                        "queue_size": self.queue.qsize(),
                    },
                )
                await asyncio.sleep(5)  # Back off on error

        self._is_running = False

    async def _collect_batch(self) -> List[Dict[str, Any]]:
        """Collects a batch of items from the queue with a timeout."""
        batch = []
        first_item = None

        def _serialize_item(item):
            """Convert item to dictionary for storage"""
            if hasattr(item, 'to_dict'):
                return item.to_dict()
            elif hasattr(item, 'model_dump'):  # Legacy Pydantic fallback
                return item.model_dump(mode="json")
            elif isinstance(item, dict):
                return item
            else:
                # Log warning for unexpected type
                logger.warning(f"Unexpected item type for storage: {type(item)}")
                return str(item)  # Fallback to string representation



        try:
            # Wait for the first item with timeout
            first_item = await asyncio.wait_for(
                self.queue.get(), timeout=self.write_timeout
            )
            batch.append(_serialize_item(first_item))

            # Collect subsequent items until batch is full or queue is empty
            while len(batch) < self.batch_size:
                try:
                    item = self.queue.get_nowait()
                    batch.append(_serialize_item(item))
                except asyncio.QueueEmpty:
                    break

        except asyncio.TimeoutError:
            pass  # No items arrived within the timeout
        except asyncio.QueueEmpty:
            pass  # The queue is empty
        except asyncio.CancelledError:
            # If we were cancelled after getting the first item, mark it as done
            if first_item is not None:
                self.queue.task_done()
            raise  # Re-raise the cancellation

        return batch

    async def _write_batch(self, items: List[Dict[str, Any]]) -> bool:
        """Write a batch of items to storage with a flattened path for BigQuery compatibility."""
        if not items:
            return True

        # Extract run_id from the first item, handling different data structures
        first_item = items[0]
        run_id = "unknown_run_id"  # Default fallback

        # Try to extract run_id from different possible structures
        if "run_id" in first_item:
            # Direct run_id field (for EvaluationResult objects)
            run_id = first_item["run_id"]
        elif "conversation_flow" in first_item and first_item["conversation_flow"]:
            # Nested in conversation_flow (for EvaluationRequest objects)
            conversation_flow = first_item["conversation_flow"]
            if isinstance(conversation_flow, dict) and "run_id" in conversation_flow:
                run_id = conversation_flow["run_id"]

        # Ensure we have a valid run_id, fallback if None or empty
        if not run_id or run_id is None:
            run_id = "unknown_run_id"

        # Get the current date and timestamp
        utc_now = datetime.now(timezone.utc)
        timestamp = utc_now.strftime("%Y%m%d_%H%M%S_%f")

        # --- New Flattened Path Logic ---
        # Embed date and run_id into the filename for BigQuery compatibility.
        filename = f"{self.file_prefix}_{run_id}_{timestamp}.jsonl"

        # Prepend the path prefix if it exists to create a top-level folder
        if self.path_prefix:
            full_path = f"{self.path_prefix}/{filename}"
        else:
            full_path = filename
        # --- End New Logic ---

        return await self.storage_backend.write_batch(items, full_path)

    async def start(self):
        """Starts the background storage worker."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._storage_worker())
            logger.info(f"Started storage worker for {self.file_prefix}")

    async def stop(self):
        """Stops the background storage worker."""
        self._is_running = False
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
            logger.info(f"Stopped storage worker for {self.file_prefix}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on storage system."""
        try:
            backend_healthy = await self.storage_backend.health_check()
            worker_healthy = (
                self._worker_task is not None and not self._worker_task.done()
            )

            return {
                "storage_manager_healthy": backend_healthy and worker_healthy,
                "backend_healthy": backend_healthy,
                "worker_healthy": worker_healthy,
                "queue_size": self.queue.qsize(),
                "processed_count": self._processed_count,
                "error_count": self._error_count,
                "is_running": self._is_running,
            }
        except Exception as e:
            logger.error("Storage manager health check failed", extra={"error": str(e)})
            return {"storage_manager_healthy": False, "error": str(e)}

    async def get_metrics(self) -> Dict[str, Any]:
        """Get storage manager metrics."""
        try:
            backend_metrics = await self.storage_backend.get_metrics()

            return {
                "file_prefix": self.file_prefix,
                "batch_size": self.batch_size,
                "write_timeout": self.write_timeout,
                "queue_size": self.queue.qsize(),
                "processed_count": self._processed_count,
                "error_count": self._error_count,
                "is_running": self._is_running,
                "backend_metrics": backend_metrics,
            }
        except Exception as e:
            logger.error(
                "Failed to get storage manager metrics", extra={"error": str(e)}
            )
            return {"file_prefix": self.file_prefix, "error": str(e)}


def create_storage_backend() -> StorageBackend:
    """Factory function to create appropriate storage backend based on configuration."""
    storage_config = config.storage

    if storage_config.storage_backend == "gcs" and storage_config.gcs_bucket_name:
        return GCSStorageBackend(bucket_name=storage_config.gcs_bucket_name)
    else:
        # Default to local storage
        return LocalStorageBackend(storage_config.audit_data_path)
