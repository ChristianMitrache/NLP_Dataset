"""
Logic for implementing embedding cache
"""

import hashlib
import json
import logging
from typing import Any, List, Optional, Union, Dict
from pathlib import Path
import diskcache


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCache:
    """
    High-performance disk cache for embedding API responses.
    Keys are generated based on input text, service, model, and other parameters.
    """

    def __init__(
        self,
        cache_dir: str = "model_cache",
        size_limit: int = 32**10,  # 32GB default
        eviction_policy: str = "least-recently-used",
    ):
        script_dir = Path(__file__).parent

        # If cache_dir is relative, make it relative to the script directory
        if not Path(cache_dir).is_absolute():
            self.cache_dir = script_dir / cache_dir
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(exist_ok=True)

        self.cache = diskcache.Cache(
            str(self.cache_dir), size_limit=size_limit, eviction_policy=eviction_policy
        )

        logger.info("Initialized EmbeddingCache at %s", self.cache_dir)
        logger.info(f"Cache size limit: {size_limit / 10**9:.1f} GB")
        logger.info("Cache entries will persist permanently (no expiration)")

    def _generate_key(
        self,
        user_input: Union[str, List[Dict[str, str]]],
        model_name: str,
    ) -> str:
        """
        Generate a unique cache key based on text and embedding parameters.

        Args:
            text: Input text(s) to embed
            params: Embedding parameters (service, model, etc.)

        Returns:
            Unique cache key string
        """
        if isinstance(user_input, dict):
            user_input_str = json.dumps(dict, sort_keys=True)
        else:
            user_input_str = user_input

        # Create deterministic key components
        key_data = {"input": user_input_str, "model_name": model_name}

        # Generate hash
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()

        # Create readable prefix
        prefix = model_name
        return f"{prefix}:{key_hash[:16]}"

    def get(
        self, user_input: Union[str, List[Dict[str, str]]], model_name: str
    ) -> Optional[Any]:
        """
        Retrieve cached embedding if it exists.

        Args:
            text: Input text(s) that was embedded
            params: Embedding parameters used

        Returns:
            Cached embedding result or None if not found
        """
        key = self._generate_key(user_input, model_name)
        return self.cache.get(key)

    def set(
        self,
        user_input: Union[str, List[Dict[str, str]]],
        model_name: str,
        response_result: Any,
    ) -> None:
        """
        Store embedding result in cache permanently (no expiration).

        Args:
            text: Input text(s) that was embedded
            params: Embedding parameters used
            embedding_result: The embedding result to cache
        """
        key = self._generate_key(user_input, model_name)
        self.cache.set(key, response_result)  # No expiration

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

    def close(self) -> None:
        """Close the cache"""
        self.cache.close()
