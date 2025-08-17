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
    default_model: str = "gpt-4o-mini" #"gpt-5-nano-2025-08-07" #
    default_temperature: float = 1.0
    
    # Retry settings
    llm_retry_count: int = 3
    llm_retry_base_delay: float = 1.0
    llm_retry_max_delay: float = 60.0
    
    # Queue settings
    queue_retry_count: int = 3
    queue_retry_base_delay: float = 1.0
    queue_retry_max_delay: float = 30.0
    
    # Streaming settings
    stream_timeout: int = 60
    chunk_timeout: int = 30
    keep_alive_interval: int = 15
