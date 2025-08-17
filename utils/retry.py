"""
Async retry decorator with exponential backoff.

This module provides a reusable decorator for implementing retry logic
with configurable backoff strategies.
"""

import asyncio
import logging
from functools import wraps
from typing import Callable, Any


def async_retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator that adds retry logic with exponential backoff to async functions.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        exceptions: Tuple of exceptions to catch and retry on (default: (Exception,))
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    logging.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
            
            logging.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            raise last_exception
            
        return wrapper
    return decorator
