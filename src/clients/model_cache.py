"""
Logic for implementing embedding cache
"""

import hashlib
import json
import logging
import os
from typing import Any, List, Optional, Union, Dict
from pathlib import Path
import diskcache
import platformdirs


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCache:
    """
    High-performance disk cache for model API responses (embeddings, completions, etc.).

    By default, cache is stored in platform-specific cache directories:
    - Linux: ~/.cache/nlp-dataset/
    - macOS: ~/Library/Caches/nlp-dataset/
    - Windows: %LOCALAPPDATA%\\nlp-dataset\\Cache\\

    Override via:
    - Environment variable: NLP_DATASET_CACHE_DIR
    - Constructor parameter: cache_dir (absolute or relative path)

    Keys are generated based on input text, service, model, and other parameters.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        size_limit: int = 8 * 10**10,  # 8GB default
        eviction_policy: str = "least-recently-used",
    ):
        """
        Initialize ModelCache with automatic platform-specific cache directory.

        Args:
            cache_dir: Optional custom cache directory. If None, uses platform-specific
                      cache directory. Can be overridden via NLP_DATASET_CACHE_DIR env var.
            size_limit: Maximum cache size in bytes (default: 16GB)
            eviction_policy: Cache eviction strategy (default: "least-recently-used")
        """
        # Determine cache directory with priority: explicit param > env var > platform default
        if cache_dir is not None:
            # User explicitly provided a cache_dir
            if not Path(cache_dir).is_absolute():
                # Relative path: make it relative to cwd
                self.cache_dir = Path.cwd() / cache_dir
            else:
                self.cache_dir = Path(cache_dir)
        else:
            # Check for environment variable override
            env_cache_dir = os.environ.get("NLP_DATASET_CACHE_DIR")
            if env_cache_dir:
                self.cache_dir = Path(env_cache_dir)
            else:
                # Use platform-specific cache directory
                self.cache_dir = Path(
                    platformdirs.user_cache_dir("nlp-dataset", "nlp-dataset")
                )

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache = diskcache.Cache(
            str(self.cache_dir), size_limit=size_limit, eviction_policy=eviction_policy
        )
        self.size_limit = size_limit
        self.eviction_policy = eviction_policy
        logger.info("Initialized ModelCache at %s", self.cache_dir)
        logger.info(f"Cache size limit: {size_limit / 10**10:.1f} GB")  # pylint: disable=w1203
        logger.info("Cache entries will persist permanently (no expiration)")

    def _generate_key(
        self,
        user_input: Union[str, List[Dict[str, str]]],
        model_name: str,
    ) -> str:
        """
        Generate a unique cache key based on user input and model parameters.

        Args:
            user_input: Input text(s) or messages sent to the model
            model_name: Name of the model used

        Returns:
            Unique cache key string (format: model_name:hash)
        """
        if isinstance(user_input, dict):
            user_input_str = json.dumps(user_input, sort_keys=True)
        else:
            user_input_str = user_input

        # Create deterministic key components
        key_data = {"input": user_input_str, "model_name": model_name}

        # Generate hash
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()

        # Create readable prefix
        prefix = model_name
        return f"{prefix}:{key_hash[:32]}"

    def get(
        self, user_input: Union[str, List[Dict[str, str]]], model_name: str
    ) -> Optional[Any]:
        """
        Retrieve cached model response if it exists.

        Args:
            user_input: Input text(s) or messages that were sent to the model
            model_name: Name of the model used

        Returns:
            Cached model response or None if not found
        """
        key = self._generate_key(user_input, model_name)
        with self.cache as cache:
            return cache.get(key)

    def set(
        self,
        user_input: Union[str, List[Dict[str, str]]],
        model_name: str,
        response_result: Any,
    ) -> None:
        """
        Store model response in cache permanently (no expiration).

        Args:
            user_input: Input text(s) or messages that were sent to the model
            model_name: Name of the model used
            response_result: The model response to cache
        """
        key = self._generate_key(user_input, model_name)
        with self.cache as cache:
            cache.set(key, response_result)  # No expiration

    def clear(self) -> None:
        """Clear all cached entries"""
        self.cache.clear()
        logger.info("Cache cleared")

    def expire_old_entries(self) -> int:
        """Remove expired entries and return count of removed items"""
        initial_size = len(self.cache)
        self.cache.expire()
        removed = initial_size - len(self.cache)
        logger.info("Expired %s old cache entries", removed)
        return removed

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary containing cache statistics:
            - cache_dir: Path to cache directory
            - size_bytes: Current cache size in bytes
            - size_mb: Current cache size in MB
            - size_gb: Current cache size in GB
            - entry_count: Number of cached entries
            - size_limit_bytes: Maximum cache size in bytes
            - size_limit_gb: Maximum cache size in GB
            - eviction_policy: Current eviction policy
            - usage_percent: Percentage of cache size limit used
        """
        volume = self.cache.volume()
        size_limit = self.size_limit

        return {
            "cache_dir": str(self.cache_dir),
            "size_bytes": volume,
            "size_mb": volume / 10**6,
            "size_gb": volume / 10**9,
            "entry_count": len(self.cache),
            "size_limit_bytes": size_limit,
            "size_limit_gb": size_limit / 10**9,
            "eviction_policy": self.eviction_policy,
            "usage_percent": (volume / size_limit * 100) if size_limit > 0 else 0,
        }

    def print_stats(self) -> None:
        """Print human-readable cache statistics to logger."""
        stats = self.get_cache_stats()
        logger.info("=" * 50)
        logger.info("ModelCache Statistics")
        logger.info("=" * 50)
        logger.info("Cache Directory: %s", stats["cache_dir"])
        logger.info("Entry Count: %d", stats["entry_count"])
        logger.info("Cache Size: %.2f GB (%.2f MB)", stats["size_gb"], stats["size_mb"])
        logger.info("Size Limit: %.2f GB", stats["size_limit_gb"])
        logger.info("Usage: %.1f%%", stats["usage_percent"])
        logger.info("Eviction Policy: %s", stats["eviction_policy"])
        logger.info("=" * 50)

    def get_cache_size(self) -> int:
        """
        Get current cache size in bytes.

        Returns:
            Current cache size in bytes
        """
        return self.cache.volume()

    def get_entry_count(self) -> int:
        """
        Get number of entries in cache.

        Returns:
            Number of cached entries
        """
        return len(self.cache)

    def close(self) -> None:
        """Close the cache"""
        self.cache.close()
