"""Centralized configuration management using Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Active LLM Provider
    llm_provider: Literal["gemini", "openai", "ollama"] = Field(
        default="gemini",
        description="Active LLM provider for inference and embeddings",
    )

    # Google Gemini Settings
    gemini_api_key: str | None = Field(
        default=None,
        description="API key for Google Gemini provider",
    )
    gemini_router_model: str = Field(
        default="gemini-3.5-flash-lite",
        description="Model identifier for deterministic routing and guardrails",
    )
    gemini_synthesizer_model: str = Field(
        default="gemini-3.5-flash-lite",
        description="Model identifier for customer response synthesis",
    )
    gemini_embedding_model: str = Field(
        default="models/gemini-embedding-001",
        description="Model identifier for vector embeddings",
    )

    # OpenAI Settings
    openai_api_key: str | None = Field(
        default=None,
        description="API key for OpenAI provider",
    )
    openai_router_model: str = Field(
        default="gpt-4o-mini",
        description="Model identifier for deterministic routing and guardrails",
    )
    openai_synthesizer_model: str = Field(
        default="gpt-4o-mini",
        description="Model identifier for customer response synthesis",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model identifier for vector embeddings",
    )

    # Ollama Settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama service",
    )
    ollama_router_model: str = Field(
        default="qwen2.5:7b",
        description="Model identifier for deterministic routing and guardrails",
    )
    ollama_synthesizer_model: str = Field(
        default="qwen2.5:7b",
        description="Model identifier for customer response synthesis",
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Model identifier for vector embeddings",
    )

    # Defense-in-Depth & Semantic Cache Settings
    enable_model_fallback: bool = Field(
        default=True,
        description="Enable automatic model failover across providers",
    )
    fallback_providers: list[str] = Field(
        default_factory=lambda: ["openai", "ollama"],
        description="Ordered list of secondary fallback providers",
    )
    semantic_cache_threshold: float = Field(
        default=0.92,
        description="Cosine similarity threshold for Semantic Cache hits",
    )
    rate_limit_max_requests: int = Field(
        default=15,
        description="Maximum requests allowed within the rate limit window",
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        description="Time window in seconds for sliding-window rate limiting",
    )
    max_input_chars: int = Field(
        default=2000,
        description="Maximum allowed character length for incoming user messages",
    )

    # LangSmith Tracing Settings
    langchain_tracing_v2: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "LANGCHAIN_TRACING_V2",
            "langchain_tracing_v2",
            "LANGCHAIN_TRACING",
            "langchain_tracing",
            "LANGSMITH_TRACING",
            "langsmith_tracing",
        ),
        description="Enable LangSmith distributed tracing",
    )
    langchain_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LANGCHAIN_API_KEY",
            "langchain_api_key",
            "LANGSMITH_API_KEY",
            "langsmith_api_key",
        ),
        description="API key for LangSmith tracing",
    )
    langchain_project: str = Field(
        default="emporio-da-musica-agent",
        validation_alias=AliasChoices(
            "LANGCHAIN_PROJECT",
            "langchain_project",
            "LANGSMITH_PROJECT",
            "langsmith_project",
        ),
        description="Project name in LangSmith dashboard",
    )
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        validation_alias=AliasChoices(
            "LANGCHAIN_ENDPOINT",
            "langchain_endpoint",
            "LANGSMITH_ENDPOINT",
            "langsmith_endpoint",
        ),
        description="LangSmith API endpoint",
    )

    # Observability & Paths
    data_dir: Path = Field(
        default=Path("data"),
        description="Directory containing operational CSVs and store policy PDF",
    )
    log_level: str = Field(
        default="INFO",
        description="Application logging verbosity level",
    )

    @model_validator(mode="after")
    def setup_langsmith_tracing(self) -> Settings:
        """Propagate LangSmith environment variables if tracing is enabled or API key is set."""
        import os

        # Prevent LangSmith EnvAddressError collision between project and agent address
        os.environ.pop("LANGSMITH_AGENT_ID", None)
        os.environ.pop("LANGSMITH_AGENT_ENVIRONMENT", None)

        if self.langchain_api_key or self.langchain_tracing_v2:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_TRACING"] = "true"
            if self.langchain_api_key:
                os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
                os.environ["LANGSMITH_API_KEY"] = self.langchain_api_key
            if self.langchain_project:
                os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
                os.environ["LANGSMITH_PROJECT"] = self.langchain_project
            if self.langchain_endpoint:
                os.environ["LANGCHAIN_ENDPOINT"] = self.langchain_endpoint
                os.environ["LANGSMITH_ENDPOINT"] = self.langchain_endpoint
        return self

    @field_validator("semantic_cache_threshold")
    @classmethod
    def validate_cache_threshold(cls, value: float) -> float:
        """Ensure semantic cache threshold is within [0.0, 1.0]."""
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"semantic_cache_threshold must be between 0.0 and 1.0, got {value}")
        return value

    @field_validator("rate_limit_max_requests", "rate_limit_window_seconds", "max_input_chars")
    @classmethod
    def validate_positive_integers(cls, value: int) -> int:
        """Ensure limits and windows are strictly positive integers."""
        if value <= 0:
            raise ValueError(f"Value must be strictly positive (> 0), got {value}")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
