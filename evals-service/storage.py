import asyncio
import os
import pandas as pd
from datetime import datetime
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
        """Write batch data to both local parquet and JSON files"""
        try:
            # Create base filename without extension
            base_filename = filename.replace(".parquet", "").replace(".json", "")
            parquet_path = self.base_path / f"{base_filename}.parquet"
            json_path = self.base_path / f"{base_filename}.json"

            # Ensure directory exists
            parquet_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to parquet with compression
            df = pd.DataFrame(data)
            df.to_parquet(parquet_path, compression="snappy", index=False)

            # Write to JSON with pretty formatting
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            self._write_count += 1
            logger.info(
                f"Successfully wrote {len(data)} records to {base_filename}",
                extra={
                    "parquet_path": str(parquet_path),
                    "json_path": str(json_path),
                    "record_count": len(data),
                    "parquet_size_bytes": parquet_path.stat().st_size
                    if parquet_path.exists()
                    else 0,
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
            # Count both parquet and JSON files
            parquet_files = list(self.base_path.rglob("*.parquet"))
            json_files = list(self.base_path.rglob("*.json"))

            total_parquet_files = len(parquet_files)
            total_json_files = len(json_files)
            total_parquet_size = sum(f.stat().st_size for f in parquet_files)
            total_json_size = sum(f.stat().st_size for f in json_files)

            return {
                "backend_type": "local",
                "base_path": str(self.base_path),
                "total_parquet_files": total_parquet_files,
                "total_json_files": total_json_files,
                "total_parquet_size_bytes": total_parquet_size,
                "total_json_size_bytes": total_json_size,
                "total_size_bytes": total_parquet_size + total_json_size,
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


class S3StorageBackend:
    """S3 storage backend (placeholder for future implementation)"""

    def __init__(self, bucket_name: str, region: str, access_key: str, secret_key: str):
        self.bucket_name = bucket_name
        self.region = region
        self.access_key = access_key
        self.secret_key = secret_key
        self._write_count = 0
        self._error_count = 0
        # TODO: Initialize boto3 client when S3 is needed

    async def write_batch(self, data: List[Dict[str, Any]], filename: str) -> bool:
        """Write batch data to S3 (placeholder)"""
        logger.warning("S3 storage not yet implemented", extra={"filename": filename})
        return False

    async def health_check(self) -> bool:
        """Check S3 connectivity (placeholder)"""
        logger.warning("S3 health check not yet implemented")
        return False

    async def get_metrics(self) -> Dict[str, Any]:
        """Get S3 metrics (placeholder)"""
        return {
            "backend_type": "s3",
            "bucket_name": self.bucket_name,
            "region": self.region,
            "write_count": self._write_count,
            "error_count": self._error_count,
            "status": "not_implemented",
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
        batch_size: int = None,
        write_timeout: float = None,
    ):
        self.queue = queue
        self.storage_backend = storage_backend
        self.file_prefix = file_prefix
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
        try:
            # Wait for the first item with timeout
            first_item = await asyncio.wait_for(
                self.queue.get(), timeout=self.write_timeout
            )
            batch.append(
                first_item.model_dump(mode="json")
                if hasattr(first_item, "model_dump")
                else first_item
            )

            # Collect subsequent items until batch is full or queue is empty
            while len(batch) < self.batch_size:
                try:
                    item = self.queue.get_nowait()
                    batch.append(
                        item.model_dump(mode="json")
                        if hasattr(item, "model_dump")
                        else item
                    )
                except asyncio.QueueEmpty:
                    break

        except asyncio.TimeoutError:
            pass  # No items arrived within the timeout
        except asyncio.QueueEmpty:
            pass  # The queue is empty

        return batch

    async def _write_batch(self, items: List[Dict[str, Any]]) -> bool:
        """Write a batch of items to storage in both Parquet and JSON formats."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.file_prefix}_{timestamp}"

        return await self.storage_backend.write_batch(items, filename)

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

    if storage_config.storage_backend == "s3" and storage_config.s3_bucket_name:
        return S3StorageBackend(
            bucket_name=storage_config.s3_bucket_name,
            region=storage_config.s3_region,
            access_key=storage_config.s3_access_key_id,
            secret_key=storage_config.s3_secret_access_key,
        )
    else:
        # Default to local storage
        return LocalStorageBackend(storage_config.audit_data_path)
