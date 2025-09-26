"""
Tests related to pydantic settings and env variables.
"""

from src.config import Settings


def test_settings_empty_string_logic(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_API_KEY", "my-llm-key")
    settings = Settings()
    assert settings.LLM_API_KEY is None
    assert settings.EMBEDDING_API_KEY == "my-llm-key"


def test_settings_from_env(monkeypatch):
    # Set environment variables
    monkeypatch.setenv("LLM_API_KEY", "my-key")
    monkeypatch.setenv("EMBEDDING_API_KEY", "my-embedding_key")
    monkeypatch.setenv("OPENAI_API_KEY", "my-key")

    settings = Settings()

    assert settings.LLM_API_KEY == "my-key"
    assert settings.OPENAI_API_KEY == "my-key"
    assert settings.LLM_ENDPOINT is None
