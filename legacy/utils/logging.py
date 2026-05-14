"""
Comprehensive logging utilities for KaggleSlayer.

This module provides:
1. Standard Python logging setup (setup_logging, get_logger, LoggerMixin)
2. Verbose printing controls for clean console output (verbose_print, suppress_feature_logs)
3. sklearn warning suppression for clean model training output
"""

import logging
import logging.handlers
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


# =============================================================================
# Standard Python Logging
# =============================================================================

def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None,
                 log_format: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        log_format: Optional custom format string

    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Create logger
    logger = logging.getLogger("kaggle_slayer")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name (will be prefixed with 'kaggle_slayer.')

    Returns:
        Logger instance
    """
    return logging.getLogger(f"kaggle_slayer.{name}")


class LoggerMixin:
    """Mixin class to add logging capability to any class.

    Usage:
        class MyClass(LoggerMixin):
            def my_method(self):
                self.log_info("This is an info message")
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__name__)

    def log_info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def log_error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def log_debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)


# =============================================================================
# Verbose Printing Controls (for clean console output)
# =============================================================================

# Global verbosity flag
_VERBOSE = True


def set_verbosity(verbose: bool) -> None:
    """Set global verbosity level.

    Args:
        verbose: If True, verbose_print() will output. If False, it will be silent.
    """
    global _VERBOSE
    _VERBOSE = verbose


def get_verbosity() -> bool:
    """Get current verbosity level.

    Returns:
        Current verbosity state
    """
    return _VERBOSE


def verbose_print(*args, **kwargs) -> None:
    """Print only if verbosity is enabled.

    This is used to suppress repetitive feature engineering logs during CV
    while still allowing important messages to be printed.

    Usage:
        verbose_print("This will only print if verbosity is True")
    """
    if _VERBOSE:
        print(*args, **kwargs)


@contextmanager
def suppress_feature_logs():
    """Suppress feature engineering logs by temporarily setting verbosity to False.

    This is used during cross-validation to prevent repetitive feature engineering
    logs from cluttering the output (e.g., logs appearing 40 times for 8 models × 5 folds).

    Usage:
        with suppress_feature_logs():
            # Feature engineering logs will be suppressed here
            model.fit(X, y)
    """
    global _VERBOSE
    original_verbose = _VERBOSE
    try:
        _VERBOSE = False
        yield
    finally:
        _VERBOSE = original_verbose


@contextmanager
def suppress_sklearn_warnings():
    """Suppress sklearn warnings and CV error tracebacks during model training.

    This hides verbose error messages from failed CV folds while still allowing
    the training to complete successfully. Useful when some hyperparameter combinations
    cause errors that are gracefully handled.

    Usage:
        with suppress_sklearn_warnings():
            # sklearn warnings and CV errors will be suppressed
            optimizer.optimize(X, y)
    """
    # Save original stdout and stderr
    original_stderr = sys.stderr
    original_stdout = sys.stdout

    try:
        # Open devnull once for reuse
        devnull = open(os.devnull, 'w')

        # Suppress warnings
        warnings.filterwarnings('ignore')

        # Redirect stderr to devnull to hide CV error tracebacks
        sys.stderr = devnull

        yield

    finally:
        # Restore stderr and stdout
        if sys.stderr != original_stderr:
            sys.stderr.close()
        sys.stderr = original_stderr

        if sys.stdout != original_stdout:
            sys.stdout.close()
        sys.stdout = original_stdout