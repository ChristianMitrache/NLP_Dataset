"""
Config for Env Variables
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent

print(PROJECT_ROOT)


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.

    The .env file should be located at the project root directory.
    """

    LLM_API_KEY: Optional[str] = None
    EMBEDDING_API_KEY: Optional[str] = None
    LLM_ENDPOINT: Optional[str] = None
    EMBEDDING_ENDPOINT: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    NLP_DATASET_CACHE_DIR: Optional[str] = None

    @field_validator("*", mode="before")
    @classmethod
    def empty_to_none(cls, v):
        """
        Convert empty strings in env variables to None.
        """
        return v or None

    model_config = SettingsConfigDict(
        env_file=os.path.join(PROJECT_ROOT, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
