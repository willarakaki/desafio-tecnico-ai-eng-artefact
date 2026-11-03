"""Decoupled LLM Factory supporting Gemini, OpenAI, and Ollama with Fallback resiliency."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


def extract_model_identifiers(model: Any, default_provider: str = "gemini") -> tuple[str, str]:
    """Extract (provider, model_name) from a LangChain BaseChatModel, Runnable, or mock."""
    name = (
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or getattr(model, "name", None)
    )
    provider = getattr(model, "provider", None)

    if provider and name:
        return str(provider), str(name)

    cls_name = type(model).__name__.lower()
    if not provider:
        if "google" in cls_name or "gemini" in cls_name:
            provider = "gemini"
        elif "openai" in cls_name:
            provider = "openai"
        elif "ollama" in cls_name:
            provider = "ollama"
        else:
            provider = default_provider

    if not name:
        name = str(type(model).__name__)

    return str(provider), str(name)


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
    """Factory for router LLMs, synthesizer LLMs, embedding models, and fallback chains.

    Centralizes provider abstraction (Gemini, OpenAI, Ollama) and enforces
    temperature presets (0.0 for deterministic routing, 0.2 for empathetic synthesis).
    Supports automatic provider failover (.with_fallbacks) across configured providers.
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
    def get_model_for_provider(
        self: Any,
        provider: str,
        model_type: str = "synthesizer",
        temperature: float = 0.2,
        settings: Settings | None = None,
    ) -> BaseChatModel:
        """Instantiate a chat model for an explicit provider identifier.

        Args:
            provider: Provider name ('gemini', 'openai', 'ollama').
            model_type: Model operational profile ('router' or 'synthesizer').
            temperature: Sampling temperature.
            settings: Optional explicit Settings.
        """
        active_settings = LLMFactory._resolve_settings(self, settings)
        p = provider.strip().lower()

        if p == "gemini":
            if not active_settings.gemini_api_key:
                raise ValueError(
                    "GEMINI_API_KEY is required when LLM_PROVIDER is 'gemini'. "
                    "Please set GEMINI_API_KEY in your environment or .env file."
                )
            m_name = (
                active_settings.gemini_router_model
                if model_type == "router"
                else active_settings.gemini_synthesizer_model
            )
            return ChatGoogleGenerativeAI(
                model=m_name,
                google_api_key=active_settings.gemini_api_key,
                temperature=temperature,
            )

        if p == "openai":
            if not active_settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'. "
                    "Please set OPENAI_API_KEY in your environment or .env file."
                )
            m_name = (
                active_settings.openai_router_model
                if model_type == "router"
                else active_settings.openai_synthesizer_model
            )
            return ChatOpenAI(
                model=m_name,
                api_key=active_settings.openai_api_key,
                temperature=temperature,
            )

        if p == "ollama":
            m_name = (
                active_settings.ollama_router_model
                if model_type == "router"
                else active_settings.ollama_synthesizer_model
            )
            return ChatOllama(
                base_url=active_settings.ollama_base_url,
                model=m_name,
                temperature=temperature,
            )

        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            "Supported providers are 'gemini', 'openai', and 'ollama'."
        )

    @_ClassOrInstanceMethod
    def get_fallback_llms(
        self: Any,
        model_type: str = "synthesizer",
        temperature: float = 0.2,
        settings: Settings | None = None,
    ) -> list[BaseChatModel]:
        """Discover and instantiate available secondary fallback LLM models.

        Inspects settings.fallback_providers, filtering out the active primary
        provider and any provider lacking required credentials.

        Returns:
            List of successfully instantiated fallback BaseChatModel instances.
        """
        active_settings = LLMFactory._resolve_settings(self, settings)
        if not getattr(active_settings, "enable_model_fallback", True):
            return []

        primary_provider = active_settings.llm_provider
        fallback_candidates = getattr(active_settings, "fallback_providers", ["openai", "ollama"])

        fallbacks: list[BaseChatModel] = []
        for provider in fallback_candidates:
            if provider == primary_provider:
                continue
            try:
                model = LLMFactory.get_model_for_provider(
                    provider=provider,
                    model_type=model_type,
                    temperature=temperature,
                    settings=active_settings,
                )
                fallbacks.append(model)
            except Exception as exc:
                logger.debug(
                    "Skipping fallback provider '%s' due to missing credentials or error: %s",
                    provider,
                    exc,
                )
        return fallbacks

    @staticmethod
    def _bind_tools_safe(model: Any, tools: list[Any] | None = None) -> Any:
        """Safely bind tools to a model or runnable if tools are present."""
        if not tools:
            return model
        try:
            return model.bind_tools(tools)
        except (NotImplementedError, AttributeError):
            return model

    @staticmethod
    def _tag_runnable(runnable: Any, provider: str, model_name: str) -> Runnable:
        """Tag an output AIMessage with provider and model_name in response_metadata."""

        def _invoke(input_data: Any, *args: Any, **kwargs: Any) -> Any:
            if hasattr(runnable, "invoke"):
                res = runnable.invoke(input_data, *args, **kwargs)
            else:
                res = runnable(input_data, *args, **kwargs)

            if isinstance(res, AIMessage):
                meta = dict(res.response_metadata or {})
                meta.setdefault("provider", provider)
                meta.setdefault("model_name", model_name)
                res.response_metadata = meta
            return res

        return RunnableLambda(_invoke)

    @classmethod
    def create_resilient_chain(
        cls,
        primary: Any,
        fallbacks: list[Any] | None = None,
        tools: list[Any] | None = None,
        primary_provider: str = "gemini",
    ) -> Runnable:
        """Construct a resilient LangChain Runnable with automatic failover and tool binding.

        Each candidate model (primary and each fallback) binds tools safely and
        has its response tagged with its true provider and model identifier.
        """
        p_prov, p_name = extract_model_identifiers(primary, default_provider=primary_provider)
        p_bound = cls._bind_tools_safe(primary, tools)
        p_tagged = cls._tag_runnable(p_bound, provider=p_prov, model_name=p_name)

        if not fallbacks:
            return p_tagged

        fb_tagged_list: list[Runnable] = []
        for fb in fallbacks:
            fb_prov, fb_name = extract_model_identifiers(fb, default_provider="fallback")
            fb_bound = cls._bind_tools_safe(fb, tools)
            fb_tagged = cls._tag_runnable(fb_bound, provider=fb_prov, model_name=fb_name)
            fb_tagged_list.append(fb_tagged)

        return p_tagged.with_fallbacks(fb_tagged_list)

    @_ClassOrInstanceMethod
    def get_resilient_synthesizer_llm(
        self: Any,
        tools: list[Any] | None = None,
        temperature: float = 0.2,
        settings: Settings | None = None,
    ) -> Runnable:
        """Return a resilient synthesizer Runnable combining primary and configured fallbacks."""
        active_settings = LLMFactory._resolve_settings(self, settings)
        primary = LLMFactory.get_synthesizer_llm(temperature=temperature, settings=active_settings)
        fallbacks = LLMFactory.get_fallback_llms(
            model_type="synthesizer", temperature=temperature, settings=active_settings
        )
        return LLMFactory.create_resilient_chain(
            primary=primary,
            fallbacks=fallbacks,
            tools=tools,
            primary_provider=active_settings.llm_provider,
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
