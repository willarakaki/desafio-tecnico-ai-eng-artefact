"""Enterprise FastAPI REST & SSE Streaming Server for Empório da Música Assistant."""

from __future__ import annotations

import datetime
import json
import logging
import uuid
from collections.abc import Generator
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agent.graph import AgentTurnResult, EmporioAgent
from src.config import Settings, get_settings
from src.observability.telemetry import (
    TelemetryManager,
    TurnTelemetry,
    get_telemetry_manager,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic v2 Request & Response Schemas
# ============================================================================


class ChatRequest(BaseModel):
    """Payload for synchronous customer turn execution."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1500,
        description="Customer natural language query or message in Portuguese.",
        examples=["Gostaria de ver violões de até 1000 reais disponíveis na loja."],
    )
    session_id: str = Field(
        default_factory=lambda: f"session-{uuid.uuid4().hex[:8]}",
        description="Unique thread identifier for conversational memory.",
        examples=["session-customer-01"],
    )
    format: Literal["sse", "vercel"] = Field(
        default="sse",
        description="Stream formatting protocol: 'sse' for standard Server-Sent Events, or 'vercel' for Vercel AI SDK.",
    )


class ChatResponse(BaseModel):
    """Structured response for synchronous turn execution."""

    output_text: str = Field(
        description="Sanitized customer-facing assistant response in Portuguese."
    )
    session_id: str = Field(description="Conversation thread identifier.")
    cache_status: str = Field(
        description="Semantic Cache status ('HIT', 'MISS', 'BYPASS_TRANSACTIONAL')."
    )
    requires_human_intervention: bool = Field(
        default=False,
        description="Whether a Human-in-the-Loop (HITL) handoff ticket was generated.",
    )
    escalation_ticket: dict[str, Any] | None = Field(
        default=None,
        description="Structured escalation ticket details if handoff occurred.",
    )
    guardrail_blocked: bool = Field(
        default=False,
        description="Whether the request was intercepted by security guardrails.",
    )
    fallback_used: bool = Field(
        default=False,
        description="Whether fallback model was activated to resolve this turn.",
    )
    resolved_provider: str = Field(
        default="gemini",
        description="Active LLM provider that processed inference.",
    )
    resolved_model: str = Field(
        default="gemini-2.5-flash",
        description="Active LLM model that processed inference.",
    )
    tools_called: list[str] = Field(
        default_factory=list,
        description="List of deterministic tools invoked during this turn.",
    )
    telemetry: TurnTelemetry = Field(
        description="FinOps latency, token usage, and cost tracking metadata."
    )


class ComponentHealth(BaseModel):
    """Health status and metadata of an individual subsystem component."""

    status: Literal["ok", "degraded", "error"]
    details: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """System-wide readiness and liveness inspection report."""

    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: str
    version: str = "0.1.0"
    environment: str = "production"
    components: dict[str, ComponentHealth]


class MetricsResponse(BaseModel):
    """FinOps and observability telemetry metrics aggregation."""

    session_id: str | None = Field(
        default=None,
        description="Filtered session ID, or null if globally aggregated.",
    )
    metrics: dict[str, Any] = Field(
        description="Turn count, token totals, USD spend, and cache hit distribution."
    )


# ============================================================================
# Dependency Providers (Singleton Access / Test Injection)
# ============================================================================

_AGENT_SINGLETON: EmporioAgent | None = None


def get_agent() -> EmporioAgent:
    """Retrieve or initialize the singleton EmporioAgent instance."""
    global _AGENT_SINGLETON
    if _AGENT_SINGLETON is None:
        _AGENT_SINGLETON = EmporioAgent()
    return _AGENT_SINGLETON


def reset_api_agent() -> None:
    """Reset the singleton agent instance (for hermetic testing)."""
    global _AGENT_SINGLETON
    _AGENT_SINGLETON = None


# ============================================================================
# FastAPI Application Factory
# ============================================================================

app = FastAPI(
    title="Empório da Música — Intelligent Assistant API",
    version="0.1.0",
    description=(
        "Enterprise backend for Empório da Música customer support agent. "
        "Integrates LangGraph orchestration, DuckDB relational queries, "
        "Hybrid RAG (BM25 + FAISS), Presidio PII Masking, Context-Aware "
        "Semantic Cache, Human-in-the-Loop escalation, and FinOps telemetry."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Cross-Origin Resource Sharing (CORS) configuration for React/Next.js frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Routes
# ============================================================================


@app.get(
    "/",
    tags=["Root"],
    summary="Root entrypoint / Swagger documentation redirect",
)
def root() -> dict[str, Any]:
    """Provide API metadata and documentation links."""
    return {
        "service": "Empório da Música Intelligent Assistant API",
        "version": "0.1.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/api/health",
        "chat_endpoint": "/api/chat",
        "chat_stream_endpoint": "/api/chat/stream",
        "metrics_endpoint": "/api/metrics",
    }


@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["Observability"],
    summary="Production readiness and liveness probe",
)
def health_check(
    settings: Settings = Depends(get_settings),
    agent: EmporioAgent = Depends(get_agent),
) -> HealthResponse:
    """Verify that all core subsystems (DuckDB, Hybrid RAG, LLM, Semantic Cache) are operational."""
    components: dict[str, ComponentHealth] = {}
    is_healthy = True

    # 1. DuckDB Ingestion & Table Verification
    try:
        table_names = agent.duckdb_manager.get_table_names()
        prod_count_res = agent.duckdb_manager.execute_query("SELECT COUNT(*) AS cnt FROM products")
        prod_count = prod_count_res[0]["cnt"] if prod_count_res else 0

        components["duckdb"] = ComponentHealth(
            status="ok",
            details={
                "tables_count": len(table_names),
                "tables": sorted(table_names),
                "products_indexed": prod_count,
            },
        )
    except Exception as exc:
        is_healthy = False
        components["duckdb"] = ComponentHealth(status="error", details={"error": str(exc)})

    # 2. Hybrid RAG Retriever Verification
    try:
        doc_count = (
            len(agent.retriever.documents)
            if agent.retriever and hasattr(agent.retriever, "documents")
            else 0
        )
        faiss_ready = bool(
            agent.retriever
            and hasattr(agent.retriever, "vectorstore")
            and agent.retriever.vectorstore is not None
        )
        components["rag"] = ComponentHealth(
            status="ok",
            details={
                "type": "hybrid_bm25_faiss",
                "policy_documents": doc_count,
                "faiss_ready": faiss_ready,
            },
        )
    except Exception as exc:
        is_healthy = False
        components["rag"] = ComponentHealth(status="error", details={"error": str(exc)})

    # 3. LLM Configuration & Resilient Fallback Verification
    try:
        components["llm"] = ComponentHealth(
            status="ok",
            details={
                "primary_provider": settings.llm_provider,
                "synthesizer_model": settings.gemini_synthesizer_model,
                "fallback_providers": ["openai", "ollama"],
                "api_key_configured": bool(settings.gemini_api_key or settings.openai_api_key),
            },
        )
    except Exception as exc:
        components["llm"] = ComponentHealth(status="degraded", details={"error": str(exc)})

    # 4. Context-Aware Semantic Cache Verification
    try:
        cache_entries = len(agent.semantic_cache._entries) if agent.semantic_cache else 0
        components["semantic_cache"] = ComponentHealth(
            status="ok",
            details={
                "entries_cached": cache_entries,
                "similarity_threshold": settings.semantic_cache_threshold,
            },
        )
    except Exception as exc:
        components["semantic_cache"] = ComponentHealth(
            status="degraded", details={"error": str(exc)}
        )

    overall_status: Literal["healthy", "degraded", "unhealthy"] = (
        "healthy" if is_healthy else "unhealthy"
    )

    response = HealthResponse(
        status=overall_status,
        timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
        version="0.1.0",
        environment="production",
        components=components,
    )

    if not is_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.model_dump(),
        )

    return response


@app.get(
    "/api/metrics",
    response_model=MetricsResponse,
    tags=["Observability"],
    summary="FinOps and telemetry metrics",
)
def get_metrics(
    session_id: str | None = Query(
        default=None,
        description="Filter metrics to a specific conversation session.",
    ),
    telemetry_mgr: TelemetryManager = Depends(get_telemetry_manager),
) -> MetricsResponse:
    """Retrieve aggregated FinOps token consumption, inference costs, latency, and cache statistics."""
    if session_id:
        metrics = telemetry_mgr.get_session_metrics(session_id)
    else:
        metrics = telemetry_mgr.get_global_metrics()

    return MetricsResponse(session_id=session_id, metrics=metrics)


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Synchronous conversational turn execution",
)
def chat(
    request: ChatRequest,
    agent: EmporioAgent = Depends(get_agent),
) -> ChatResponse:
    """Process a natural language user query through the LangGraph agent pipeline."""
    try:
        turn_result: AgentTurnResult = agent.run(
            query=request.query,
            session_id=request.session_id,
        )
    except Exception as exc:
        logger.exception("Error executing turn in /api/chat: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha interna ao processar mensagem: {exc}",
        ) from exc

    return ChatResponse(
        output_text=turn_result.output_text,
        session_id=turn_result.session_id,
        cache_status=turn_result.cache_status,
        requires_human_intervention=turn_result.requires_human_intervention,
        escalation_ticket=turn_result.escalation_ticket,
        guardrail_blocked=turn_result.guardrail_blocked,
        fallback_used=turn_result.fallback_used,
        resolved_provider=turn_result.resolved_provider,
        resolved_model=turn_result.resolved_model,
        tools_called=turn_result.tools_called,
        telemetry=turn_result.telemetry,
    )


@app.post(
    "/api/chat/stream",
    tags=["Chat"],
    summary="Server-Sent Events (SSE) & Vercel AI SDK streaming endpoint",
)
def chat_stream(
    request: ChatRequest,
    agent: EmporioAgent = Depends(get_agent),
    x_vercel_ai_data_stream: str | None = Header(default=None),
) -> StreamingResponse:
    """Stream graph events, tool calls, and text tokens incrementally.

    Supports dual stream formats:
    - Standard SSE (`format='sse'`, default): event frames with 'step', 'tool_call', 'token', 'done'.
    - Vercel AI SDK (`format='vercel'` or `x-vercel-ai-data-stream: v1` header): Data Stream protocol
      compatible with `@ai-sdk/react` (`useChat`).
    """
    is_vercel_format = (request.format == "vercel") or (x_vercel_ai_data_stream == "v1")

    if is_vercel_format:
        return _stream_vercel_ai_sdk(
            agent=agent, query=request.query, session_id=request.session_id
        )

    return _stream_standard_sse(agent=agent, query=request.query, session_id=request.session_id)


# ============================================================================
# Streaming Helpers
# ============================================================================


def _stream_standard_sse(
    agent: EmporioAgent,
    query: str,
    session_id: str,
) -> StreamingResponse:
    """Yield standard Server-Sent Events (SSE) with `event` and `data` fields."""

    def event_generator() -> Generator[str, None, None]:
        try:
            for item in agent.stream(query=query, session_id=session_id):
                event_name = item.get("event", "message")
                data_payload = json.dumps(item.get("data", {}), ensure_ascii=False)
                yield f"event: {event_name}\ndata: {data_payload}\n\n"
        except Exception as exc:
            logger.error("Exception in SSE stream: %s", exc)
            err_data = json.dumps({"error": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {err_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _stream_vercel_ai_sdk(
    agent: EmporioAgent,
    query: str,
    session_id: str,
) -> StreamingResponse:
    """Yield lines conforming to the Vercel AI SDK UI Message / Data Stream protocol."""

    def vercel_generator() -> Generator[str, None, None]:
        try:
            for item in agent.stream(query=query, session_id=session_id):
                event_type = item.get("event")
                data = item.get("data", {})

                if event_type == "token":
                    # Text token part: 0:"<json_escaped_string>"\n
                    token_text = data.get("token", "")
                    yield f"0:{json.dumps(token_text, ensure_ascii=False)}\n"
                elif event_type in ("step", "tool_call", "tool_result"):
                    # Data part: 2:[<json_payload>]\n
                    part = json.dumps([{"type": event_type, "payload": data}], ensure_ascii=False)
                    yield f"2:{part}\n"
                elif event_type == "done":
                    # Finish / usage part: d:{"finishReason":"stop", ...}\n
                    finish_meta = {
                        "finishReason": "stop",
                        "session_id": session_id,
                        "telemetry": data.get("telemetry"),
                        "requires_human_intervention": data.get(
                            "requires_human_intervention", False
                        ),
                        "cache_status": data.get("cache_status", "MISS"),
                    }
                    yield f"d:{json.dumps(finish_meta, ensure_ascii=False)}\n"
                elif event_type == "error":
                    err_msg = data.get("error", "Erro de execução")
                    yield f"3:{json.dumps(err_msg, ensure_ascii=False)}\n"
        except Exception as exc:
            logger.error("Exception in Vercel stream: %s", exc)
            yield f"3:{json.dumps(str(exc), ensure_ascii=False)}\n"

    return StreamingResponse(
        vercel_generator(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "x-vercel-ai-data-stream": "v1",
        },
    )
