"""
Centralized Logging Configuration for Creator AI

Usage:
    from shared.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline started")
    logger.warning("GPU memory low")
    logger.error("Model failed to load")
"""
import logging
import sys


_INITIALIZED = False


def _init_logging():
    """Initialize the root logger once."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    
    root = logging.getLogger("creator_ai")
    root.setLevel(logging.DEBUG)
    
    # Console handler with clean format
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    
    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    _init_logging()
    # Prefix with creator_ai for namespace isolation
    if not name.startswith("creator_ai"):
        name = f"creator_ai.{name}"
    return logging.getLogger(name)
