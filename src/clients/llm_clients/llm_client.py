"""
Wrapper around pydantic ai enabling additional features
"""

from typing import Optional, List, Dict, Union
import asyncio
from pydantic import BaseModel
from openai import AsyncOpenAI, APIError
from src.clients.model_cache import ModelCache


class FormattingException(Exception):
    """Exception raised when response format validation fails."""


class LLMCompletions:
    """
    Convenience Wrapper for various functionality.
    - Keeping with the openai api (which can be used for vllm servers, TGI and openai)
    - caching model outputs
    - Certain model output constrained operations.
    """

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_cache: Optional[ModelCache] = None,
        max_api_retries: int = 3,
        max_output_retries: int = 3,
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.max_api_retries = max_api_retries
        self.cache = model_cache
        self.token_usage = {}
        self.model_name = model_name
        self.max_output_retries = max_output_retries

    async def submit_request(
        self,
        messages: List[Dict[str, str]],
        sample_size: int = 1,
        response_format: Optional[type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> List[Union[str, type[BaseModel]]]:
        """
        Submit a request to the OpenAI API with caching support.
        Cache key is generated based on model_name and messages only.

        Args:
            messages: List of message dicts [{"role": "user/assistant/system", "content": "..."}]
            sample_size: Number of completions to generate
            response_format: Optional Pydantic model for structured output
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments to pass to the API

        Returns:
            List of generated responses
        """
        # Check cache if available (cache key only depends on model_name and messages)
        cached_response = None
        if self.cache:
            cached_response = self.cache.get(
                user_input=messages, model_name=self.model_name
            )
        cached_size = len(cached_response) if cached_response else 0

        # If cache has enough samples, return the requested amount
        if cached_size >= sample_size:
            return cached_response[:sample_size]

        # Partial cache hit: generate the difference
        remaining_samples = sample_size - cached_size
        new_responses = await self._submit_no_cache_request(
            messages=messages,
            sample_size=remaining_samples,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Combine cached and new responses
        combined_responses = (
            cached_response + new_responses if cached_response else new_responses
        )

        # Update cache with the combined results
        if self.cache:
            self.cache.set(
                user_input=messages,
                model_name=self.model_name,
                response_result=combined_responses,
            )

        return combined_responses

    async def _submit_no_cache_request(
        self,
        messages: List[Dict[str, str]],
        sample_size: int = 1,
        response_format: Optional[type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> List[Union[str, type[BaseModel]]]:
        """
        Submit a request to the OpenAI API without caching.

        Args:
            messages: List of message dicts [{"role": "user/assistant/system", "content": "..."}]
            sample_size: Number of completions to generate
            response_format: Optional Pydantic model for structured output
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments to pass to the API

        Returns:
            List of generated responses
        """
        # Prepare API call parameters
        api_params = {
            "model": self.model_name,
            "messages": messages,
            "n": sample_size,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            **kwargs,
        }
        # Retry logic for API calls
        for attempt in range(self.max_api_retries):
            try:
                if response_format:
                    completion = await self.client.chat.completions.parse(**api_params)
                    result = [choice.message.parsed for choice in completion.choices]
                else:
                    completion = await self.client.chat.completions.create(**api_params)
                    result = [choice.message.content for choice in completion.choices]

                if completion.usage:
                    self._update_token_usage(completion)

            except APIError:
                # Retry with exponential backoff for all other errors
                if attempt == self.max_api_retries - 1:
                    raise
                await asyncio.sleep(min(2 ** (attempt + 2), 60))
                continue
            break  # Success, exit retry loop

        return result

    def _update_token_usage(self, completion):
        """
        Updating token
        """
        model_key = self.model_name or "default"
        if model_key not in self.token_usage:
            self.token_usage[model_key] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        self.token_usage[model_key]["prompt_tokens"] += completion.usage.prompt_tokens
        self.token_usage[model_key]["completion_tokens"] += (
            completion.usage.completion_tokens
        )
        self.token_usage[model_key]["total_tokens"] += completion.usage.total_tokens

    def get_token_usage(self) -> Dict[str, Dict[str, int]]:
        """Get token usage statistics."""
        return self.token_usage

    def reset_token_usage(self):
        """Reset token usage statistics."""
        self.token_usage = {}
