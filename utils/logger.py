"""
Logging utility - Centralized logging configuration
"""
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def setup_logger(name, log_file=None, level=None):
    """
    Set up logger with console and file handlers
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level (default from .env)
    
    Returns:
        Configured logger
    """
    if level is None:
        level = os.getenv('LOG_LEVEL', 'INFO')
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
