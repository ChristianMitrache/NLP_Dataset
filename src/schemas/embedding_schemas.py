"""
Core Schemas storing text and embeddings.
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class EmbeddingResponse:
    """Container for embedding response data"""

    text: str
    embedding: List[float]
    usage: Dict[str, int]
