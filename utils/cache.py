"""
Caching utilities for expensive operations.
"""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Any, Optional, Callable
from functools import wraps
import time


class DiskCache:
    """Simple disk-based cache for expensive operations."""

    def __init__(self, cache_dir: Path = None, ttl: int = 3600):
        """
        Initialize disk cache.

        Args:
            cache_dir: Directory to store cache files
            ttl: Time-to-live in seconds (default 1 hour)
        """
        if cache_dir is None:
            cache_dir = Path.cwd() / ".cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl

    def _get_cache_key(self, key: str) -> str:
        """Generate a hash-based cache key."""
        return hashlib.md5(key.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        cache_key = self._get_cache_key(key)
        return self.cache_dir / f"{cache_key}.pkl"

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache."""
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        # Check if cache is expired
        if time.time() - cache_path.stat().st_mtime > self.ttl:
            cache_path.unlink()
            return None

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        """Store value in cache."""
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            print(f"Warning: Failed to cache value: {e}")

    def clear(self) -> None:
        """Clear all cached values."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

    def cache_result(self, key_prefix: str = ""):
        """
        Decorator to cache function results.

        Args:
            key_prefix: Optional prefix for cache key
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key from function name and arguments
                key_parts = [key_prefix, func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = "|".join(key_parts)

                # Try to get from cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value

                # Compute and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result)
                return result

            return wrapper
        return decorator


class LRUCache:
    """Simple in-memory LRU cache."""

    def __init__(self, max_size: int = 100):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of items to store
        """
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.current_time = 0

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache."""
        if key in self.cache:
            self.current_time += 1
            self.access_times[key] = self.current_time
            return self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Store value in cache."""
        # Remove oldest item if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        self.cache[key] = value
        self.current_time += 1
        self.access_times[key] = self.current_time

    def clear(self) -> None:
        """Clear all cached values."""
        self.cache.clear()
        self.access_times.clear()
        self.current_time = 0


# Global cache instances
_disk_cache = None
_memory_cache = None


def get_disk_cache(cache_dir: Path = None, ttl: int = 3600) -> DiskCache:
    """Get global disk cache instance."""
    global _disk_cache
    if _disk_cache is None:
        _disk_cache = DiskCache(cache_dir, ttl)
    return _disk_cache


def get_memory_cache(max_size: int = 100) -> LRUCache:
    """Get global memory cache instance."""
    global _memory_cache
    if _memory_cache is None:
        _memory_cache = LRUCache(max_size)
    return _memory_cache


def cached(use_disk: bool = False, ttl: int = 3600, max_size: int = 100):
    """
    Decorator to cache function results.

    Args:
        use_disk: Use disk cache instead of memory cache
        ttl: Time-to-live for disk cache (seconds)
        max_size: Maximum size for memory cache
    """
    if use_disk:
        cache = get_disk_cache(ttl=ttl)
        return cache.cache_result()
    else:
        cache = get_memory_cache(max_size=max_size)

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = "|".join(key_parts)

                # Try to get from cache
                cached_value = cache.get(cache_key)
                if cached_value is not None:
                    return cached_value

                # Compute and cache result
                result = func(*args, **kwargs)
                cache.set(cache_key, result)
                return result

            return wrapper
        return decorator
