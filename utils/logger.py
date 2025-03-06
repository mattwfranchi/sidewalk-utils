"""
Custom logger for the autoglancing project.
Provides standardized logging with timestamp, log level, and source file information.
"""

import logging
import os
import sys
import datetime
from typing import Optional

# Define color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

# Create custom log level for success messages
SUCCESS_LEVEL_NUM = 25  # between INFO and WARNING
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors and consistent formatting"""
    
    def __init__(self):
        fmt = "[%(asctime)s] %(levelname)8s - [%(filename)s:%(lineno)d] - %(message)s"
        super().__init__(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
        
        self.FORMATS = {
            logging.DEBUG: Colors.BLUE + self._fmt + Colors.RESET,
            logging.INFO: Colors.WHITE + self._fmt + Colors.RESET,
            SUCCESS_LEVEL_NUM: Colors.GREEN + self._fmt + Colors.RESET,
            logging.WARNING: Colors.YELLOW + self._fmt + Colors.RESET,
            logging.ERROR: Colors.RED + self._fmt + Colors.RESET,
            logging.CRITICAL: Colors.BOLD + Colors.RED + self._fmt + Colors.RESET
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

class Logger(logging.Logger):
    """Custom logger class for autoglancing project"""
    
    def __init__(self, name, level=logging.INFO):
        super().__init__(name, level)
        self.setup_logger()
        
    def success(self, msg, *args, **kwargs):
        """Log a success message (level 25)"""
        if self.isEnabledFor(SUCCESS_LEVEL_NUM):
            self._log(SUCCESS_LEVEL_NUM, msg, args, **kwargs)
    
    def setup_logger(self):
        """Configure logger with console handler and custom formatter"""
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter())
        self.addHandler(console_handler)

def get_logger(name: Optional[str] = None) -> Logger:
    """
    Get a logger instance with the given name.
    If name is None, the calling module's name will be used.
    
    Args:
        name: Logger name (usually __name__ from the calling module)
        
    Returns:
        Logger: Configured logger instance
    """
    if name is None:
        # If no name provided, try to get the caller's module name
        import inspect
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        name = os.path.basename(module.__file__).split('.')[0] if module else "autoglancing"
    
    # Clear any existing logger to avoid duplicate handlers
    if name in logging.Logger.manager.loggerDict:
        for handler in logging.getLogger(name).handlers[:]:
            logging.getLogger(name).removeHandler(handler)
    
    # Register our custom logger class
    logging.setLoggerClass(Logger)
    
    # Get and return logger
    return logging.getLogger(name)