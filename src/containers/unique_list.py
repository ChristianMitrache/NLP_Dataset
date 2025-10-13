"""
Holds logic for single categorical topic dataset.
"""

from typing import Optional, List, Dict
from collections.abc import Callable
import inspect
import logging
import faiss
import numpy as np
from pydantic import BaseModel
from src.clients.llm_client import LLMCompletions
from src.clients.embedding_client import EmbeddingClient
from src.utilities.text_cleaning import clean_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class BooleanResponse(BaseModel):
    answer: bool


class UniqueList:
    """
    This implements the main functionality of SingleCategoryList.

    This class is implemented for <10k unique data-points in mind.
    """

    async def __init__(
        self,
        items: List[str],
        get_equality_prompt: Callable[[str, str], List[Dict]],
        llm_endpoint: LLMCompletions,
        embedding_endpoint: Optional[EmbeddingClient],
        deduplicate_on_init: bool = True,
        llm_temperature: Optional[float] = None,
        top_k=10,
        top_percentage=0.05,
    ):
        # Validate get_first_contained_in_second_prompt
        if not callable(get_equality_prompt):
            raise ValueError("get_first_contained_in_second_prompt must be a callable")

        sig = inspect.signature(get_equality_prompt)
        params = list(sig.parameters.values())
        if len(params) != 2:
            raise ValueError(
                f"get_first_contained_in_second_prompt must accept exactly 2 parameters (str, str), "
                f"but got {len(params)} parameters"
            )

        # Endpoints and utilities to endpoints
        self.embedding_endpoint = embedding_endpoint
        self.llm_endpoint = llm_endpoint
        self.get_equality_prompt = get_equality_prompt
        self.user_top_k = top_k
        self.top_percentage = top_percentage
        self.llm_temp = llm_temperature
        self.top_k = None

        # Internal Stored Data
        self.items = items
        self.standard_vector_index = None
        self.standard_items = []
        self.standard_to_items = {}  # Standardized item --> items that correspond to this
        await self.create_vector_index()
        if deduplicate_on_init:
            await self.standardize_items()

    async def create_vector_index(self) -> np.ndarray:
        """
        Create Vector Index and return vectors
        """
        # Accumulating and cleaning standard terms:
        for item in self.items:
            standardized_term = clean_string(item)
            if standardized_term in self.standard_to_items:
                self.standard_to_items[standardized_term].append(item)
            else:
                self.standard_to_items[standardized_term] = [item]

        self.standard_items = list(self.standard_to_items.keys())
        self.standard_vector_index = faiss.IndexFlatL2(
            self.embedding_endpoint.get_embedding_dim()
        )
        logger.info("Embedding Standardized texts...")
        vectors = np.array(self.embedding_endpoint.embed_batches(self.standard_items))
        self.standard_vector_index.add(
            len(self.standard_items),
            vectors,
        )
        return vectors

    def update_top_k(self) -> None:
        """
        Update true top_k parameter based on item size.
        """
        self.top_k = max(
            min(self.user_top_k, len(self.standard_items)),
            int(self.top_percentage * len(self.standard_items)),
        )

    async def standardize_items(self) -> None:
        """
        Iterates over items provided by user
        """
        self.update_top_k()
        _, indices = self.standard_vector_index.search(  # pylint: disable=no-value-for-parameter
            np.array(self.embedding_endpoint.embed_batches(self.standard_items)),
            self.top_k,
        )

        eq_prompts = [
            self.get_equality_prompt(
                self.standard_items[i], self.standard_items[int(indices[i, j])]
            )
            for i in range(indices.shape[0])
            for j in range(indices.shape[1])
        ]
        eq_results = np.array(
            self.llm_endpoint.submit_requests(
                eq_prompts, response_format=BooleanResponse
            )
        )
