import logging
import os
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler

class LoggerConfig:
    """
    Centralized configuration for the logging system.
    
    This class contains all constants and settings used by the logging system,
    making it easy to modify logging behavior in one place.
    """
    
    # Directory and file settings
    LOG_DIR = "logs"
    LOG_FILE_PATTERN = "{date}.log"
    DATE_FORMAT = "%Y-%m-%d"
    
    # Default log levels
    DEFAULT_CONSOLE_LEVEL = logging.INFO
    DEFAULT_FILE_LEVEL = logging.DEBUG
    
    # Log message formats
    CONSOLE_FORMAT = "%(name)s - %(levelname)s - %(message)s"
    FILE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # File rotation settings
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5

def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    console_level: Optional[int] = None,
    file_level: Optional[int] = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True
) -> logging.Logger:
    """
    Configure and return a logger with console and file handlers.
    
    This function creates a complete logging system that outputs to both console
    and rotating log files. It automatically creates the log directory and
    prevents duplicate handlers when called multiple times.
    
    Args:
        name (str): Logger name, typically __name__ of the calling module
        log_dir (Optional[str]): Directory for log files. Defaults to "logs"
        console_level (Optional[int]): Console logging level. Defaults to INFO
        file_level (Optional[int]): File logging level. Defaults to DEBUG
        enable_file_logging (bool): Enable file logging. Defaults to True
        enable_console_logging (bool): Enable console logging. Defaults to True
        
    Returns:
        logging.Logger: Configured logger instance
        
    Raises:
        OSError: If log directory cannot be created
        ValueError: If invalid log levels are provided
        
    Example:
        >>> # Basic usage
        >>> logger = setup_logger(__name__)
        >>> logger.info("Application started successfully")
        
        >>> # Custom configuration
        >>> logger = setup_logger(
        ...     "my_module",
        ...     log_dir="/custom/logs",
        ...     console_level=logging.WARNING,
        ...     file_level=logging.DEBUG
        ... )
        >>> logger.debug("Detailed debug information")
        >>> logger.warning("This will appear in console")
        
    Note:
        - Log files are named by date (e.g., "2023-12-25.log")
        - Files rotate when they reach 10MB, keeping 5 backup copies
        - Console logs are simplified for readability
        - File logs include timestamps for detailed analysis
    """
    try:
        # Get or create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # Capture all log levels
        
        # Set default values
        log_dir = log_dir or LoggerConfig.LOG_DIR
        console_level = console_level or LoggerConfig.DEFAULT_CONSOLE_LEVEL
        file_level = file_level or LoggerConfig.DEFAULT_FILE_LEVEL
        
        # Validate log levels
        _validate_log_levels(console_level, file_level)
        
        # Clear existing handlers to prevent duplicates
        _clear_existing_handlers(logger)
        
        # Create log directory if needed
        if enable_file_logging:
            _ensure_log_directory_exists(log_dir)
        
        # Add console handler if enabled
        if enable_console_logging:
            console_handler = _create_console_handler(console_level)
            logger.addHandler(console_handler)
        
        # Add file handler if enabled
        if enable_file_logging:
            file_handler = _create_file_handler(file_level, log_dir)
            logger.addHandler(file_handler)
        
        # Prevent duplicate logs in root logger
        logger.propagate = False
        
        return logger
        
    except Exception as e:
        # Fallback to basic logging if setup fails
        _fallback_logger_setup(name, e)
        return logging.getLogger(name)

def _validate_log_levels(console_level: int, file_level: int) -> None:
    """
    Validate that provided log levels are within acceptable range.
    
    Args:
        console_level (int): Console logging level
        file_level (int): File logging level
        
    Raises:
        ValueError: If any log level is invalid
    """
    valid_levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
    
    for level in [console_level, file_level]:
        if level not in valid_levels:
            raise ValueError(f"Invalid log level: {level}. Must be one of {valid_levels}")

def _clear_existing_handlers(logger: logging.Logger) -> None:
    """
    Remove all existing handlers from a logger to prevent duplicates.
    
    Args:
        logger: Logger instance to clear handlers from
    """
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

def _ensure_log_directory_exists(log_dir: str) -> None:
    """
    Create log directory if it doesn't exist.
    
    Args:
        log_dir (str): Path to the log directory
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

def _create_console_handler(level: int) -> logging.StreamHandler:
    """
    Create and configure a console handler.
    
    Args:
        level (int): Logging level for this handler
        
    Returns:
        logging.StreamHandler: Configured console handler
    """
    handler = logging.StreamHandler()
    handler.setLevel(level)
    
    formatter = logging.Formatter(
        LoggerConfig.CONSOLE_FORMAT,
        datefmt=LoggerConfig.DATETIME_FORMAT
    )
    handler.setFormatter(formatter)
    
    return handler

def _create_file_handler(level: int, log_dir: str) -> RotatingFileHandler:
    """
    Create and configure a rotating file handler.
    
    Args:
        level (int): Logging level for this handler
        log_dir (str): Directory for log files
        
    Returns:
        RotatingFileHandler: Configured file handler with rotation
    """
    # Generate filename with current date
    current_date = datetime.now().strftime(LoggerConfig.DATE_FORMAT)
    filename = LoggerConfig.LOG_FILE_PATTERN.format(date=current_date)
    filepath = os.path.join(log_dir, filename)
    
    # Create rotating file handler
    handler = RotatingFileHandler(
        filepath,
        maxBytes=LoggerConfig.MAX_FILE_SIZE,
        backupCount=LoggerConfig.BACKUP_COUNT,
        encoding='utf-8'
    )
    handler.setLevel(level)
    
    # Add detailed formatter with timestamps
    formatter = logging.Formatter(
        LoggerConfig.FILE_FORMAT,
        datefmt=LoggerConfig.DATETIME_FORMAT
    )
    handler.setFormatter(formatter)
    
    return handler

def _fallback_logger_setup(name: str, error: Exception) -> None:
    """
    Setup basic logging as fallback if advanced setup fails.
    
    Args:
        name (str): Logger name
        error (Exception): Original error that caused fallback
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.error(f"Failed to setup advanced logging: {error}")
    logger.info("Using fallback logging configuration")