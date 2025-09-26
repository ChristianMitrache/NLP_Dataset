"""
Logic for Embedding documents using an external Embedding Server
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
import asyncio
from openai import AsyncOpenAI, OpenAIError, RateLimitError
from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResponse:
    """Container for embedding response data"""

    text: str
    embedding: List[float]
    index: int
    model: str
    usage: Dict[str, int]


class EmbeddingClient:
    """
    Class for managing embedding batches of text in asynchronous manner.
    """

    def __init__(
        self,
        max_concurrency: int,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        embedding_endpoint: Optional[str] = settings.EMBEDDING_API_KEY,
        api_key: Optional[str] = settings.EMBEDDING_API_KEY,
        max_retries: int = 2,
    ):
        if not api_key:
            logger.warning(
                "No API Key provided - defaulting to OPENAI_API_KEY env variable..."
            )
        if not embedding_endpoint:
            logger.warning(
                "No embedding endpoint provided - defaulting to openai endpoint..."
            )

        self.max_retries = max_retries
        self.client = AsyncOpenAI(api_key=api_key, base_url=embedding_endpoint)
        self.model = model
        self.max_concurrent = max_concurrency
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def _make_request(
        self, texts: List[str], start_index: int
    ) -> List[EmbeddingResponse]:
        for attempt in range(self.max_retries):
            async with self.semaphore:
                try:
                    response = await self.client.embeddings.create(
                        model=self.model,
                        input=texts,
                        encoding_format="float",
                    )

                    if response.status == 200:
                        return [
                            EmbeddingResponse(
                                text=texts[item.index],
                                embedding=item.embedding,
                                index=start_index + item.index,
                                model=response.model,
                                usage=dict(response.usage),
                            )
                            for item in response.data
                        ]
                except RateLimitError as e:
                    retry_after = int(e.response.headers.get("Retry-After", 5))
                    logger.warning("Rate limited. Waiting %s seconds...", retry_after)
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(retry_after)
                    continue

                except OpenAIError as e:
                    logger.error("OpenAI API error: %s", e)
                    if attempt == self.max_retries - 1:
                        raise

    async def embed_batches(
        self, texts: List[str], show_progress: bool = True
    ) -> List[EmbeddingResponse]:
        """
        Function to batch submit embeddings to a hosted client.
        """
        if not texts:
            return []

        batches = [
            (texts[i : i + self.batch_size], i)
            for i in range(0, len(texts), self.batch_size)
        ]

        tasks = [
            self._make_request(batch_texts, start_index)
            for batch_texts, start_index in batches
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Batch %s failed: %s", i, result)
                raise result
            all_responses.extend(result)

        if show_progress:
            total_tokens = sum(
                r.usage.get("total_tokens", 0) if not r is None else 0
                for r in all_responses
            )
            logger.info("Completed! Total tokens used: %s", total_tokens)

        return all_responses
