"""
Core Schemas storing text and embeddings.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EmbeddingResponse:
    """Container for embedding response data"""

    text: str
    embedding: Optional[List[float]]
