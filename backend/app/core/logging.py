"""
Logging configuration for NeuroMotion backend.

Uses structlog for structured, JSON-formatted logs that are
easy to parse and analyze in log aggregation systems.
"""

import logging
import os
import sys
from typing import Any, Dict

import structlog
from structlog.stdlib import LoggerFactory


def configure_logging():
    """
    Configure structured logging for the application.

    Features:
    - JSON format for production
    - Pretty format for development
    - Request ID correlation
    - Log level from configuration
    - Timestamps in ISO format
    """
    log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Configure structlog
    structlog.configure(
        processors=[
            # Add context (request ID, user, etc.)
            structlog.contextvars.merge_contextvars,
            # Add log level
            structlog.processors.add_log_level,
            # Add timestamp
            structlog.processors.TimeStamper(fmt="iso"),
            # Add exception details
            structlog.dev.set_exc_info,
            # Pretty-print for development
            structlog.dev.ConsoleRenderer() if log_level == logging.DEBUG else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str = None) -> structlog.BoundLogger:
    """
    Get a configured logger instance.

    Args:
        name: Optional logger name (usually __name__ of the calling module)

    Returns:
        Configured structlog logger instance.

    Usage:
        logger = get_logger(__name__)
        logger.info("User logged in", user_id=user.id)
    """
    return structlog.get_logger(name)
