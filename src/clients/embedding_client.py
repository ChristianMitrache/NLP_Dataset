"""
Logic for Embedding documents using an external Embedding Server.
Note this is compatible with both OpenAI embeddings as well as
embedding servers hosted using TEI (From huggingface).
"""

from typing import List, Optional
import logging
import asyncio
from itertools import chain
from tqdm.asyncio import tqdm_asyncio

from openai import AsyncOpenAI, OpenAIError, RateLimitError
from src.config import settings
from src.schemas.embedding_schemas import EmbeddingResponse
from src.clients.model_cache import ModelCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class EmbeddingClient:
    """
    Class for managing embedding batches of text in asynchronous manner.
    """

    def __init__(
        self,
        max_concurrency: int,
        model: Optional[str] = None,
        model_cache: Optional[ModelCache] = None,
        batch_size: int = 100,
        embedding_endpoint: Optional[str] = settings.EMBEDDING_ENDPOINT,
        api_key: Optional[str] = settings.EMBEDDING_API_KEY,
        max_retries: int = 3,
    ):
        if model_cache and not isinstance(model_cache, ModelCache):
            raise ValueError("model_cache parameter must be of type ModelCache")

        if not api_key:
            logger.warning(
                "No API Key provided - defaulting to OPENAI_API_KEY env variable..."
            )
        if not embedding_endpoint:
            logger.warning(
                "No embedding endpoint provided - defaulting to openai endpoint..."
            )
        self.cache = model_cache
        self.max_retries = max_retries
        self.client = AsyncOpenAI(api_key=api_key, base_url=embedding_endpoint)
        self.model = model
        self.max_concurrent = max_concurrency
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrency)

    def _post_process_requests(
        self, texts: list[str], response
    ) -> list[EmbeddingResponse]:
        """
        Post processing and adding each resulting EmbeddingResponse object to cache.
        """
        if response is None:
            return [EmbeddingResponse(text, None) for text in texts]

        return_embeddings = [None] * len(texts)

        for item in response.data:
            text = texts[item.index]

            emb_response = EmbeddingResponse(
                text=text,
                embedding=item.embedding,
            )
            if self.cache:
                self.cache.set(
                    user_input=text,
                    model_name=self.model,
                    response_result=emb_response,
                )
            return_embeddings[item.index] = emb_response

        return return_embeddings

    async def _make_request(self, texts: List[str]) -> List[EmbeddingResponse]:
        for attempt in range(self.max_retries):
            async with self.semaphore:
                try:
                    response = await self.client.embeddings.create(
                        model=self.model,
                        input=texts,
                        encoding_format="float",
                    )

                    return self._post_process_requests(texts, response)

                except RateLimitError as e:
                    if attempt < self.max_retries - 1:
                        retry_after = int(e.response.headers.get("Retry-After", 5))
                        logger.warning(
                            "Rate limited. Waiting %s seconds...", retry_after
                        )
                        await asyncio.sleep(retry_after)
                        continue

                except OpenAIError as e:
                    if attempt < self.max_retries - 1:
                        logger.warning("OpenAI API error: %s", e)

        logger.error(
            "Max Retries exceeded on request \
                    returing Empty EmbeddingResponse Objects"
        )
        return self._post_process_requests(texts, None)

    async def embed_batches(self, texts: List[str], show_progress: bool = True):
        """
        Function to batch submit embeddings.

        1. Split up input into cached and non-cached embeddings
        2. Send off non-cached embeddings to asynchronous api calls.
        3. Collect outputs from both in same order and return.
        """
        if self.cache is None:
            return await self._embed_batches_non_cached(
                texts, show_progress=show_progress
            )

        # If cache is supplied, keep track of cache misses and only send
        # texts with cache misses to the embedding endpoint
        cached_output = {}
        non_cached_input = []
        non_cached_indices = {}
        logger.info("Checking Embedding Cache...")
        for i, text in enumerate(texts):
            cache_result = self.cache.get(text, self.model)
            if cache_result:
                cached_output[i] = cache_result
            else:
                non_cached_indices[i] = len(non_cached_input)
                non_cached_input.append(text)

        embedding_outputs = await self._embed_batches_non_cached(
            non_cached_input, show_progress=show_progress
        )

        return [
            cached_output[i]
            if i in cached_output
            else embedding_outputs[non_cached_indices[i]]
            for i in range(len(texts))
        ]

    async def _embed_batches_non_cached(
        self, texts: List[str], show_progress: bool = True
    ) -> List[EmbeddingResponse]:
        """
        Function to batch submit embeddings to a hosted client.
        """
        if not texts:
            return []

        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        tasks = [self._make_request(batch_texts) for batch_texts in batches]

        if show_progress:
            results = await tqdm_asyncio.gather(
                *tasks,
                desc="Embedding Batches...",
            )
        else:
            results = await asyncio.gather(*tasks)

        return list(chain(*results))
