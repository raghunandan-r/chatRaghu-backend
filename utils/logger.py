import logging
import os
from datetime import datetime
from typing import Any
from pythonjsonlogger import jsonlogger
from fastapi import Request
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration


def initialize_sentry():
    """Initializes Sentry SDK if in production environment."""
    if os.getenv("ENVIRONMENT") == "prod":
        sentry_logging = LoggingIntegration(
            level=logging.INFO,  # Capture info and above as breadcrumbs
            event_level=logging.ERROR,  # Send errors as events
        )
        sentry_sdk.init(
            dsn=os.getenv("SENTRY_DSN"),
            environment="production",
            traces_sample_rate=1.0,
            profiles_sample_rate=1.0,
            integrations=[
                FastApiIntegration(),
                sentry_logging,
            ],
            send_default_pii=True,
            _experiments={
                "continuous_profiling_auto_start": True,
            },
        )


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record.update(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "environment": os.getenv("ENVIRONMENT", "development"),
            }
        )


# Set up the logger for your application
logger = logging.getLogger("chatraghu")
logger.setLevel(logging.INFO)

# Prevent duplicate logs by not adding handlers if they already exist
if not logger.handlers:
    # Console handler with color formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter for console
    formatter = CustomJsonFormatter("%(timestamp)s %(level)s %(name)s %(message)s")
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)


# Add a function to capture errors in Sentry with additional context
def log_error(message: str, error: Exception = None, **kwargs):
    """Log an error and capture it in Sentry with additional context"""
    logger.error(message, extra=kwargs)
    if error:
        sentry_sdk.capture_exception(error)
    else:
        sentry_sdk.capture_message(message, level="error")


async def log_request_info(request: Request) -> dict:
    return {
        "client_ip": request.client.host,
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "path_params": dict(request.path_params),
        "thread_id": request.headers.get("x-thread-id"),
    }
