"""
Timer decorator module for deos2ac project.
Provides function execution time tracking using the project's logger.
"""

import time
import functools
from typing import Callable, Any, Optional

from utils.logger import get_logger

logger = get_logger("timer")

def time_it(level: str = "info", message: Optional[str] = None):
    """
    Decorator to log the execution time of a function.
    
    Args:
        level: Log level to use ('debug', 'info', 'success', 'warning', 'error', 'critical')
        message: Optional custom message prefix. If None, a default message is used.
    
    Returns:
        The decorated function.
    
    Example:
        @time_it()
        def my_function():
            # do something
            
        @time_it(level="debug", message="Custom timing message for")  
        def another_function():
            # do something
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            elapsed_time = time.time() - start_time
            
            # Format time string based on duration
            if elapsed_time < 0.001:
                time_str = f"{elapsed_time * 1000000:.2f} μs"
            elif elapsed_time < 1:
                time_str = f"{elapsed_time * 1000:.2f} ms"
            else:
                time_str = f"{elapsed_time:.4f} s"
            
            # Use the provided message or create a default one
            msg = message or "Execution time of"
            log_message = f"{msg} '{func.__name__}': {time_str}"
            
            # Use the appropriate log level
            log_level = level.lower()
            if log_level == "debug":
                logger.debug(log_message)
            elif log_level == "info":
                logger.info(log_message)
            elif log_level == "success":
                logger.success(log_message)
            elif log_level == "warning":
                logger.warning(log_message)
            elif log_level == "error":
                logger.error(log_message)
            elif log_level == "critical":
                logger.critical(log_message)
            else:
                # Default to info if an invalid level is provided
                logger.info(log_message)
            
            return result
        return wrapper
    return decorator