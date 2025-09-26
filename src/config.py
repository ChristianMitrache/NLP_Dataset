"""
Config for Env Variables
"""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    LLM_API_KEY: Optional[str] = None
    EMBEDDING_API_KEY: Optional[str] = None
    LLM_ENDPOINT: Optional[str] = None
    EMBEDDING_ENDPOINT: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None

    @field_validator("*", mode="before")
    @classmethod
    def empty_to_none(cls, v):
        """
        Convert empty strings in env variables to None.
        """
        return v or None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
