"""
Configuration management for the GraphEngine refactor.

This module provides a single source of truth for all configuration values,
reading from environment variables with sensible defaults.
"""

from dataclasses import dataclass
import os


@dataclass
class GraphConfig:
    """Configuration for the graph engine and related components."""
    
    # LLM settings
    default_model: str = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    default_temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    
    # Retry settings
    llm_retry_count: int = int(os.getenv("LLM_RETRY", "3"))
    llm_retry_base_delay: float = float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0"))
    llm_retry_max_delay: float = float(os.getenv("LLM_RETRY_MAX_DELAY", "60.0"))
    
    # Queue settings
    queue_retry_count: int = int(os.getenv("QUEUE_RETRY", "3"))
    queue_retry_base_delay: float = float(os.getenv("QUEUE_RETRY_BASE_DELAY", "1.0"))
    queue_retry_max_delay: float = float(os.getenv("QUEUE_RETRY_MAX_DELAY", "30.0"))
    
    # Streaming settings
    stream_timeout: int = int(os.getenv("STREAM_TIMEOUT", "60"))
    chunk_timeout: int = int(os.getenv("CHUNK_TIMEOUT", "30"))
    keep_alive_interval: int = int(os.getenv("KEEP_ALIVE_INTERVAL", "15"))
