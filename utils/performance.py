"""
Performance monitoring and profiling utilities.
"""

import time
import functools
from typing import Callable, Dict, Any
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class PerformanceTimer:
    """Context manager and decorator for timing operations."""

    def __init__(self, name: str = "Operation", log_level: str = "INFO"):
        """
        Initialize performance timer.

        Args:
            name: Name of the operation being timed
            log_level: Logging level for timing output
        """
        self.name = name
        self.log_level = log_level
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log results."""
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time

        log_func = getattr(logger, self.log_level.lower(), logger.info)
        log_func(f"{self.name} completed in {self.elapsed:.2f}s")

    def __call__(self, func: Callable) -> Callable:
        """Decorator usage."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with PerformanceTimer(f"{func.__name__}", self.log_level):
                return func(*args, **kwargs)
        return wrapper


@contextmanager
def timer(name: str = "Operation"):
    """
    Simple context manager for timing code blocks.

    Example:
        with timer("Data loading"):
            df = load_data()
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{name} took {elapsed:.2f}s")


def timed(func: Callable) -> Callable:
    """
    Decorator to time function execution.

    Example:
        @timed
        def expensive_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


class PerformanceProfiler:
    """Track performance metrics across multiple operations."""

    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = {}

    def record(self, operation: str, duration: float, metadata: Dict[str, Any] = None):
        """
        Record a performance metric.

        Args:
            operation: Name of the operation
            duration: Duration in seconds
            metadata: Optional metadata about the operation
        """
        if operation not in self.metrics:
            self.metrics[operation] = {
                'count': 0,
                'total_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0,
                'calls': []
            }

        metric = self.metrics[operation]
        metric['count'] += 1
        metric['total_time'] += duration
        metric['min_time'] = min(metric['min_time'], duration)
        metric['max_time'] = max(metric['max_time'], duration)
        metric['calls'].append({
            'duration': duration,
            'metadata': metadata or {}
        })

    def get_summary(self, operation: str = None) -> Dict[str, Any]:
        """
        Get performance summary.

        Args:
            operation: Specific operation to summarize, or None for all

        Returns:
            Dictionary with performance statistics
        """
        if operation:
            if operation not in self.metrics:
                return {}

            metric = self.metrics[operation]
            return {
                'operation': operation,
                'count': metric['count'],
                'total_time': metric['total_time'],
                'avg_time': metric['total_time'] / metric['count'],
                'min_time': metric['min_time'],
                'max_time': metric['max_time']
            }

        # Return summary for all operations
        return {
            op: self.get_summary(op)
            for op in self.metrics
        }

    def print_summary(self):
        """Print a formatted performance summary."""
        if not self.metrics:
            print("No performance metrics recorded.")
            return

        print("\n" + "="*70)
        print("PERFORMANCE SUMMARY")
        print("="*70)

        for operation, summary in self.get_summary().items():
            print(f"\n{operation}:")
            print(f"  Calls: {summary['count']}")
            print(f"  Total Time: {summary['total_time']:.2f}s")
            print(f"  Avg Time: {summary['avg_time']:.2f}s")
            print(f"  Min Time: {summary['min_time']:.2f}s")
            print(f"  Max Time: {summary['max_time']:.2f}s")

        print("\n" + "="*70)

    @contextmanager
    def profile(self, operation: str, metadata: Dict[str, Any] = None):
        """
        Context manager to profile an operation.

        Example:
            profiler = PerformanceProfiler()
            with profiler.profile("data_loading"):
                load_data()
        """
        start = time.time()
        yield
        duration = time.time() - start
        self.record(operation, duration, metadata)


# Global profiler instance
_global_profiler = None


def get_profiler() -> PerformanceProfiler:
    """Get global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profile(operation: str = None):
    """
    Decorator to profile function execution.

    Example:
        @profile("model_training")
        def train_model():
            ...
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            profiler.record(op_name, duration)
            return result
        return wrapper
    return decorator
