from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional, List


class StorageConfig(BaseSettings):
    """Configuration for storage backends"""

    # Local storage paths
    audit_data_path: str = Field(
        default="./audit_data", description="Path for audit data storage"
    )
    eval_results_path: str = Field(
        default="./eval_results", description="Path for evaluation results storage"
    )

    # GCS configuration for separate buckets
    gcs_audit_bucket_name: Optional[str] = Field(
        default=None, description="GCS bucket for audit data storage"
    )
    gcs_eval_results_bucket_name: Optional[str] = Field(
        default=None, description="GCS bucket for evaluation results storage"
    )

    # Generic GCS bucket name (for legacy compatibility / simple setups)
    gcs_bucket_name: Optional[str] = Field(
        default=None,
        description="Generic GCS bucket name (if using a single bucket for all data).",
    )

    # Storage backend selection
    storage_backend: str = Field(
        default="local", description="Storage backend: local or gcs"
    )

    # Batch processing configuration
    batch_size: int = Field(
        default=15, description="Number of items to batch before writing"
    )
    write_timeout_seconds: float = Field(
        default=5.0, description="Timeout for collecting batch items"
    )

    model_config = SettingsConfigDict(env_prefix="STORAGE_")


class LLMConfig(BaseSettings):
    """Configuration for LLM services"""

    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(
        default="gpt-5-nano",
        description="OpenAI model to use for evaluation",
    )
    openai_max_retries: int = Field(
        default=3, description="Maximum retries for OpenAI API calls"
    )
    openai_timeout_seconds: int = Field(
        default=30, description="Timeout for OpenAI API calls"
    )

    # Rate limiting
    requests_per_minute: int = Field(
        default=60, description="Rate limit for LLM requests per minute"
    )
    tokens_per_minute: int = Field(
        default=100000, description="Rate limit for LLM tokens per minute"
    )

    model_config = SettingsConfigDict(env_prefix="LLM_")


class ServiceConfig(BaseSettings):
    """Main service configuration"""

    # Service identification
    service_name: str = Field(
        default="chatraghu-evals", description="Service name for monitoring"
    )
    service_version: str = Field(default="1.0.0", description="Service version")
    environment: str = Field(
        default="development",
        description="Environment: development, staging, production",
    )

    # Health check configuration
    health_check_interval_seconds: int = Field(
        default=30, description="Health check interval"
    )
    health_check_timeout_seconds: int = Field(
        default=5, description="Health check timeout"
    )

    # Queue configuration
    max_queue_size: int = Field(
        default=10000, description="Maximum queue size before backpressure"
    )
    queue_worker_count: int = Field(default=1, description="Number of queue workers")

    # Monitoring and observability
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    enable_tracing: bool = Field(default=True, description="Enable distributed tracing")
    log_level: str = Field(default="INFO", description="Logging level")

    # Error handling
    max_retry_attempts: int = Field(
        default=3, description="Maximum retry attempts for failed operations"
    )
    retry_backoff_factor: float = Field(
        default=2.0, description="Exponential backoff factor"
    )

    model_config = SettingsConfigDict(env_prefix="EVAL_")


class Config(BaseSettings):
    """Main configuration class that combines all sub-configurations"""

    service: ServiceConfig = Field(default_factory=ServiceConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # API configuration
    api_host: str = Field(default="::", description="API host")
    api_port: int = Field(default=8001, description="API port")
    api_workers: int = Field(default=1, description="Number of API workers")

    # CORS configuration
    cors_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    cors_credentials: bool = Field(default=True, description="Allow CORS credentials")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields from environment
    )


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Return the global config instance (for backward compatibility)"""
    return config
