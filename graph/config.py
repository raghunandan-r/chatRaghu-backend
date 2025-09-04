"""
Configuration management for the GraphEngine refactor.

This module provides a single source of truth for all configuration values,
reading from environment variables with sensible defaults.
"""

from dataclasses import dataclass
from graph.utils import get_graph_version


@dataclass
class GraphConfigDefault:
    """Configuration for the graph engine and related components."""

    graph_type: str = "resume"
    # LLM settings
    default_model: str = "gpt-4o-mini"  # "gpt-5-nano-2025-08-07" #
    default_temperature: float = 0.1  # 1.0

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

    # Graph version
    graph_version: str = get_graph_version("resume")


@dataclass
class GraphConfigImmi:
    """Configuration for the graph engine and related components."""

    graph_type: str = "immi"
    # LLM settings
    default_model: str = "gpt-4.1-mini-2025-04-14"  # "gpt-5-nano-2025-08-07" #
    default_temperature: float = 0.1

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

    # Graph version
    graph_version: str = get_graph_version("immi")
