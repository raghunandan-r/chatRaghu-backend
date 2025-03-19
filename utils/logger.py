import logging
import sys
import json
from datetime import datetime
from typing import Any
from pythonjsonlogger import jsonlogger
from fastapi import Request
import sentry_sdk

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record: dict[str, Any], record: logging.LogRecord, message_dict: dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record.update({
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "environment": "production" if not sys.gettrace() else "development"
        })

# Set up the logger for your application
logger = logging.getLogger("chatraghu")
logger.setLevel(logging.INFO)

# Console handler with color formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter for console
formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(name)s %(message)s')
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)

# Add a function to capture errors in Sentry
def log_error(message: str, error: Exception = None, **kwargs):
    """Log an error and capture it in Sentry"""
    logger.error(message, **kwargs)
    if error:
        sentry_sdk.capture_exception(error)

async def log_request_info(request: Request) -> dict:
    return {
        "client_ip": request.client.host,
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "path_params": dict(request.path_params),
        "thread_id": request.headers.get("x-thread-id"),
    } 