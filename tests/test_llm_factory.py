"""Unit tests for centralized configuration and LLMFactory provider decoupling."""

from __future__ import annotations

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import ValidationError
from src.agent.llm_factory import LLMFactory
from src.config import Settings, get_settings


def test_default_settings_initialization() -> None:
    """Validate that default settings match challenge specifications."""
    settings = Settings(
        gemini_api_key="dummy_gemini_key",
        _env_file=None,  # Do not read .env to test pristine defaults
    )
    assert settings.llm_provider == "gemini"
    assert settings.gemini_router_model == "gemini-2.5-flash"
    assert settings.gemini_synthesizer_model == "gemini-2.5-flash"
    assert settings.gemini_embedding_model == "models/gemini-embedding-001"
    assert settings.openai_router_model == "gpt-4o-mini"
    assert settings.openai_synthesizer_model == "gpt-4o-mini"
    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_router_model == "qwen2.5:7b"
    assert settings.ollama_synthesizer_model == "qwen2.5:7b"
    assert settings.ollama_embedding_model == "nomic-embed-text"
    assert settings.semantic_cache_threshold == 0.92
    assert settings.rate_limit_max_requests == 15
    assert settings.rate_limit_window_seconds == 60
    assert settings.max_input_chars == 2000
    assert settings.log_level == "INFO"


def test_settings_validation_thresholds() -> None:
    """Validate bounds on numeric thresholds."""
    with pytest.raises(ValidationError):
        Settings(semantic_cache_threshold=1.5, _env_file=None)

    with pytest.raises(ValidationError):
        Settings(semantic_cache_threshold=-0.1, _env_file=None)

    with pytest.raises(ValidationError):
        Settings(rate_limit_max_requests=0, _env_file=None)

    with pytest.raises(ValidationError):
        Settings(rate_limit_window_seconds=-10, _env_file=None)

    with pytest.raises(ValidationError):
        Settings(max_input_chars=0, _env_file=None)


def test_get_settings_cached_singleton() -> None:
    """Ensure get_settings returns a cached singleton instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_gemini_provider_instantiation() -> None:
    """Validate LLMFactory instantiation for Google Gemini models."""
    settings = Settings(
        llm_provider="gemini",
        gemini_api_key="test-gemini-key",
        _env_file=None,
    )
    factory = LLMFactory(settings=settings)

    router_llm = factory.get_router_llm()
    assert isinstance(router_llm, ChatGoogleGenerativeAI)
    assert router_llm.model in ("gemini-2.5-flash", "models/gemini-2.5-flash")
    assert router_llm.temperature == 0.0

    synth_llm = factory.get_synthesizer_llm()
    assert isinstance(synth_llm, ChatGoogleGenerativeAI)
    assert synth_llm.model in ("gemini-2.5-flash", "models/gemini-2.5-flash")
    assert synth_llm.temperature == 0.2

    embeddings = factory.get_embeddings()
    assert isinstance(embeddings, GoogleGenerativeAIEmbeddings)
    assert embeddings.model == "models/gemini-embedding-001"


def test_gemini_missing_api_key_raises() -> None:
    """Validate that missing Gemini API key raises descriptive ValueError."""
    settings = Settings(
        llm_provider="gemini",
        gemini_api_key=None,
        _env_file=None,
    )
    factory = LLMFactory(settings=settings)

    with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
        factory.get_router_llm()

    with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
        factory.get_synthesizer_llm()

    with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
        factory.get_embeddings()


def test_openai_provider_instantiation() -> None:
    """Validate LLMFactory instantiation for OpenAI models."""
    settings = Settings(
        llm_provider="openai",
        openai_api_key="test-openai-key",
        _env_file=None,
    )
    factory = LLMFactory(settings=settings)

    router_llm = factory.get_router_llm()
    assert isinstance(router_llm, ChatOpenAI)
    assert router_llm.model_name == "gpt-4o-mini"
    assert router_llm.temperature == 0.0

    synth_llm = factory.get_synthesizer_llm()
    assert isinstance(synth_llm, ChatOpenAI)
    assert synth_llm.model_name == "gpt-4o-mini"
    assert synth_llm.temperature == 0.2

    embeddings = factory.get_embeddings()
    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.model == "text-embedding-3-small"


def test_openai_missing_api_key_raises() -> None:
    """Validate that missing OpenAI API key raises descriptive ValueError."""
    settings = Settings(
        llm_provider="openai",
        openai_api_key=None,
        _env_file=None,
    )
    factory = LLMFactory(settings=settings)

    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        factory.get_router_llm()

    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        factory.get_synthesizer_llm()

    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        factory.get_embeddings()


def test_ollama_provider_instantiation() -> None:
    """Validate LLMFactory instantiation for local Ollama open-weights models."""
    settings = Settings(
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_router_model="qwen2.5:7b",
        ollama_synthesizer_model="qwen2.5:7b",
        ollama_embedding_model="nomic-embed-text",
        _env_file=None,
    )
    factory = LLMFactory(settings=settings)

    router_llm = factory.get_router_llm()
    assert isinstance(router_llm, ChatOllama)
    assert router_llm.model == "qwen2.5:7b"
    assert router_llm.base_url == "http://localhost:11434"
    assert router_llm.temperature == 0.0

    synth_llm = factory.get_synthesizer_llm()
    assert isinstance(synth_llm, ChatOllama)
    assert synth_llm.model == "qwen2.5:7b"
    assert synth_llm.temperature == 0.2

    embeddings = factory.get_embeddings()
    assert isinstance(embeddings, OllamaEmbeddings)
    assert embeddings.model == "nomic-embed-text"
    assert embeddings.base_url == "http://localhost:11434"


def test_temperature_override() -> None:
    """Ensure explicit temperature overrides factory defaults."""
    settings = Settings(
        llm_provider="gemini",
        gemini_api_key="test-key",
        _env_file=None,
    )
    factory = LLMFactory(settings=settings)

    custom_router = factory.get_router_llm(temperature=0.7)
    assert custom_router.temperature == 0.7

    custom_synth = factory.get_synthesizer_llm(temperature=0.0)
    assert custom_synth.temperature == 0.0


def test_unsupported_provider_raises() -> None:
    """Validate error handling when an unsupported provider is configured."""
    settings = Settings(
        llm_provider="gemini",
        gemini_api_key="test-key",
        _env_file=None,
    )
    # Force invalid provider attribute
    object.__setattr__(settings, "llm_provider", "unsupported_provider")
    factory = LLMFactory(settings=settings)

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        factory.get_router_llm()

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        factory.get_synthesizer_llm()

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        factory.get_embeddings()


def test_classmethod_and_instance_parity() -> None:
    """Verify classmethod calls behave identically to instance calls with injected settings."""
    settings = Settings(
        llm_provider="openai",
        openai_api_key="test-key",
        _env_file=None,
    )
    instance_factory = LLMFactory(settings=settings)

    inst_llm = instance_factory.get_router_llm()
    cls_llm = LLMFactory.get_router_llm(settings=settings)

    assert type(inst_llm) is type(cls_llm)
    assert inst_llm.model_name == cls_llm.model_name
    assert inst_llm.temperature == cls_llm.temperature
