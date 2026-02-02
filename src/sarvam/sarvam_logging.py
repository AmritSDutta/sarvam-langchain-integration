"""Logging configuration for Sarvam LangChain integration."""

import logging

# Create package logger
logger = logging.getLogger("sarvam")

# Add NullHandler to prevent logging if not configured by application
# This is the best practice for libraries - let the application control logging
logger.addHandler(logging.NullHandler())

# Export logger for use in other modules
__all__ = ["logger"]
