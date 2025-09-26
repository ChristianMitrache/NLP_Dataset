"""
File For Fixtures
"""
from unittest.mock import AsyncMock
import pytest
from src.clients.embedding_client import EmbeddingClient 

@pytest.fixture
def mock_embedding_client():
    """
    Provides an EmbeddingClient with a mocked AsyncOpenAI client.
    """
    client = EmbeddingClient(batch_size=2, max_concurrency=5, api_key="fake-key")

    # Fake response object structure
    fake_response = AsyncMock()
    fake_response.status = 200
    fake_response.model = "text-embedding-3-small"
    fake_response.usage = {"total_tokens": 5}
    fake_response.data = [
        AsyncMock(index=i, embedding=[0.1, 0.2, 0.3]) for i in range(2)
    ]

    # Patch the embeddings.create method
    client.client.embeddings.create = AsyncMock(return_value=fake_response)

    return client
