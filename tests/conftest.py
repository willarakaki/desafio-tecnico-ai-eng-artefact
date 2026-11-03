"""Pytest configuration and hermetic test environment fixtures."""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest
from src.config import Settings, get_settings


@pytest.fixture(autouse=True, scope="session")
def isolate_test_environment() -> Generator[None, None, None]:
    """Ensure automated tests run hermetically offline without hitting live API rate limits."""
    saved_gemini = os.environ.get("GEMINI_API_KEY")
    saved_openai = os.environ.get("OPENAI_API_KEY")

    # Remove active keys from environment
    os.environ.pop("GEMINI_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)

    # Disable .env file loading in Settings during test runs
    Settings.model_config["env_file"] = None
    get_settings.cache_clear()

    yield

    Settings.model_config["env_file"] = ".env"
    get_settings.cache_clear()

    if saved_gemini is not None:
        os.environ["GEMINI_API_KEY"] = saved_gemini
    if saved_openai is not None:
        os.environ["OPENAI_API_KEY"] = saved_openai
