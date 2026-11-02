"""Decoupled LLM Factory supporting Google Gemini, OpenAI, and Ollama providers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import Settings, get_settings


class _ClassOrInstanceMethod:
    """Descriptor enabling a method to be called either on the class or an instance."""

    def __init__(self, func: Callable[..., Any]) -> None:
        self.func = func

    def __get__(self, instance: Any, owner: Any = None) -> Callable[..., Any]:
        target = instance if instance is not None else owner

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.func(target, *args, **kwargs)

        return wrapper


class LLMFactory:
    """Factory for instantiating router LLMs, synthesizer LLMs, and embedding models.

    Centralizes provider abstraction (Gemini, OpenAI, Ollama) and enforces
    temperature presets (0.0 for deterministic routing, 0.2 for empathetic synthesis).
    Supports dependency injection via instance initialization or direct classmethod invocation.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings: Settings = settings or get_settings()

    @staticmethod
    def _resolve_settings(target: Any, override: Settings | None = None) -> Settings:
        """Resolve active settings between explicit override, instance settings, or singleton."""
        if override is not None:
            return override
        if isinstance(target, LLMFactory):
            return target.settings
        return get_settings()

    @_ClassOrInstanceMethod
    def get_router_llm(
        self: Any,
        temperature: float = 0.0,
        settings: Settings | None = None,
    ) -> BaseChatModel:
        """Return a low-temperature model optimized for deterministic routing and guardrails.

        Args:
            temperature: Sampling temperature (default 0.0 for deterministic outputs).
            settings: Optional explicit Settings instance to override active settings.

        Returns:
            An instantiated LangChain BaseChatModel for the active provider.
        """
        active_settings = LLMFactory._resolve_settings(self, settings)
        provider = active_settings.llm_provider

        if provider == "gemini":
            if not active_settings.gemini_api_key:
                raise ValueError(
                    "GEMINI_API_KEY is required when LLM_PROVIDER is 'gemini'. "
                    "Please set GEMINI_API_KEY in your environment or .env file."
                )
            return ChatGoogleGenerativeAI(
                model=active_settings.gemini_router_model,
                google_api_key=active_settings.gemini_api_key,
                temperature=temperature,
            )

        if provider == "openai":
            if not active_settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'. "
                    "Please set OPENAI_API_KEY in your environment or .env file."
                )
            return ChatOpenAI(
                model=active_settings.openai_router_model,
                api_key=active_settings.openai_api_key,
                temperature=temperature,
            )

        if provider == "ollama":
            return ChatOllama(
                base_url=active_settings.ollama_base_url,
                model=active_settings.ollama_router_model,
                temperature=temperature,
            )

        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            "Supported providers are 'gemini', 'openai', and 'ollama'."
        )

    @_ClassOrInstanceMethod
    def get_synthesizer_llm(
        self: Any,
        temperature: float = 0.2,
        settings: Settings | None = None,
    ) -> BaseChatModel:
        """Return a consultative model aligned with the Empório da Música store persona.

        Args:
            temperature: Sampling temperature (default 0.2 for empathetic consultative responses).
            settings: Optional explicit Settings instance to override active settings.

        Returns:
            An instantiated LangChain BaseChatModel for the active provider.
        """
        active_settings = LLMFactory._resolve_settings(self, settings)
        provider = active_settings.llm_provider

        if provider == "gemini":
            if not active_settings.gemini_api_key:
                raise ValueError(
                    "GEMINI_API_KEY is required when LLM_PROVIDER is 'gemini'. "
                    "Please set GEMINI_API_KEY in your environment or .env file."
                )
            return ChatGoogleGenerativeAI(
                model=active_settings.gemini_synthesizer_model,
                google_api_key=active_settings.gemini_api_key,
                temperature=temperature,
            )

        if provider == "openai":
            if not active_settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'. "
                    "Please set OPENAI_API_KEY in your environment or .env file."
                )
            return ChatOpenAI(
                model=active_settings.openai_synthesizer_model,
                api_key=active_settings.openai_api_key,
                temperature=temperature,
            )

        if provider == "ollama":
            return ChatOllama(
                base_url=active_settings.ollama_base_url,
                model=active_settings.ollama_synthesizer_model,
                temperature=temperature,
            )

        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            "Supported providers are 'gemini', 'openai', and 'ollama'."
        )

    @_ClassOrInstanceMethod
    def get_embeddings(
        self: Any,
        settings: Settings | None = None,
    ) -> Embeddings:
        """Return a dense vector embeddings model matching the active provider.

        Args:
            settings: Optional explicit Settings instance to override active settings.

        Returns:
            An instantiated LangChain Embeddings model for the active provider.
        """
        active_settings = LLMFactory._resolve_settings(self, settings)
        provider = active_settings.llm_provider

        if provider == "gemini":
            if not active_settings.gemini_api_key:
                raise ValueError(
                    "GEMINI_API_KEY is required when LLM_PROVIDER is 'gemini'. "
                    "Please set GEMINI_API_KEY in your environment or .env file."
                )
            return GoogleGenerativeAIEmbeddings(
                model=active_settings.gemini_embedding_model,
                google_api_key=active_settings.gemini_api_key,
            )

        if provider == "openai":
            if not active_settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'. "
                    "Please set OPENAI_API_KEY in your environment or .env file."
                )
            return OpenAIEmbeddings(
                model=active_settings.openai_embedding_model,
                api_key=active_settings.openai_api_key,
            )

        if provider == "ollama":
            return OllamaEmbeddings(
                base_url=active_settings.ollama_base_url,
                model=active_settings.ollama_embedding_model,
            )

        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            "Supported providers are 'gemini', 'openai', and 'ollama'."
        )
