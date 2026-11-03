"""FinOps telemetry, token accounting, and latency observability for agent turns."""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

CacheStatusType = Literal["HIT", "MISS", "BYPASS_TRANSACTIONAL"]
GuardrailStatusType = Literal["passed", "blocked", "rate_limited"]

# Pricing in USD per 1,000,000 tokens
MODEL_PRICING_PER_1M: dict[str, dict[str, float]] = {
    # Gemini 2.5 Flash / Gemini 2.0 Flash
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    # OpenAI GPT-4o-mini
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # Ollama / Local Models
    "qwen2.5:7b": {"input": 0.0, "output": 0.0},
    "qwen2.5": {"input": 0.0, "output": 0.0},
    "default": {"input": 0.075, "output": 0.30},
}


def calculate_token_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate the estimated inference cost in USD for a given token usage.

    Args:
        model_name: Identifier of the active model (e.g., 'gemini-2.5-flash').
        input_tokens: Number of prompt/input tokens.
        output_tokens: Number of completion/output tokens.

    Returns:
        Estimated cost in USD rounded to 8 decimal places.
    """
    if input_tokens <= 0 and output_tokens <= 0:
        return 0.0

    key = model_name.strip().lower()
    rates = MODEL_PRICING_PER_1M.get(key)
    if rates is None:
        # Match by prefix (e.g. gemini-2.5-flash-latest -> gemini-2.5-flash)
        for candidate, candidate_rates in MODEL_PRICING_PER_1M.items():
            if candidate in key:
                rates = candidate_rates
                break
        if rates is None:
            rates = MODEL_PRICING_PER_1M["default"]

    cost = (input_tokens / 1_000_000.0) * rates["input"] + (output_tokens / 1_000_000.0) * rates[
        "output"
    ]
    return round(cost, 8)


# ============================================================================
# Pydantic v2 Telemetry Model
# ============================================================================


class TurnTelemetry(BaseModel):
    """Structured telemetry metadata captured for each user turn."""

    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(description="Unique session identifier for the user conversation.")
    turn_id: str = Field(
        default_factory=lambda: f"turn-{uuid.uuid4().hex[:8]}",
        description="Unique identifier for this specific interaction turn.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat(),
        description="ISO 8601 timestamp of turn execution.",
    )
    latency_ms: float = Field(
        default=0.0, description="End-to-end execution latency in milliseconds."
    )
    cache_status: CacheStatusType = Field(
        default="MISS",
        description="Semantic Cache status: 'HIT', 'MISS', or 'BYPASS_TRANSACTIONAL'.",
    )
    tools_called: list[str] = Field(
        default_factory=list,
        description="List of deterministic tools invoked during the turn.",
    )
    input_tokens: int = Field(default=0, description="Input/prompt token count.")
    output_tokens: int = Field(default=0, description="Output/completion token count.")
    total_tokens: int = Field(default=0, description="Combined total token count.")
    estimated_cost_usd: float = Field(default=0.0, description="Estimated inference cost in USD.")
    requires_human_intervention: bool = Field(
        default=False,
        description="Whether this turn triggered a Human-in-the-Loop (HITL) escalation ticket.",
    )
    escalation_ticket_id: str | None = Field(
        default=None,
        description="Ticket ID if an escalation was triggered (ex: 'TKT-20261102-A1B2').",
    )
    guardrail_status: GuardrailStatusType = Field(
        default="passed",
        description="Guardrail verdict: 'passed', 'blocked', or 'rate_limited'.",
    )
    provider: str = Field(
        default="gemini", description="LLM provider name ('gemini', 'openai', 'ollama')."
    )
    model_name: str = Field(default="gemini-2.5-flash", description="Underlying model name used.")
    fallback_used: bool = Field(
        default=False,
        description="Whether a fallback model was activated to resolve this turn.",
    )

    def format_summary(self) -> str:
        """Return a human-readable single-line summary of turn telemetry."""
        tools_str = ", ".join(self.tools_called) if self.tools_called else "None"
        hitl_str = (
            f" [HITL: {self.escalation_ticket_id}]" if self.requires_human_intervention else ""
        )
        return (
            f"Latency: {self.latency_ms:.1f}ms | Cache: {self.cache_status} | "
            f"Tools: [{tools_str}] | Tokens: {self.total_tokens} (In: {self.input_tokens}, "
            f"Out: {self.output_tokens}) | Cost: ${self.estimated_cost_usd:.6f}{hitl_str}"
            + (f" [Fallback: {self.provider}/{self.model_name}]" if self.fallback_used else "")
        )


# ============================================================================
# Telemetry Manager (Session & Global Aggregation)
# ============================================================================


class TelemetryManager:
    """In-memory telemetry collector and FinOps aggregator."""

    def __init__(self) -> None:
        self._turns: list[TurnTelemetry] = []

    def record_turn(self, telemetry: TurnTelemetry) -> None:
        """Record a completed turn's telemetry object."""
        self._turns.append(telemetry)
        logger.debug(
            "Recorded telemetry for session '%s' (turn %s): %s",
            telemetry.session_id,
            telemetry.turn_id,
            telemetry.format_summary(),
        )

    def get_session_turns(self, session_id: str) -> list[TurnTelemetry]:
        """Return all recorded turns for a specific session in chronological order."""
        return [t for t in self._turns if t.session_id == session_id]

    def get_session_metrics(self, session_id: str) -> dict[str, Any]:
        """Aggregate performance and FinOps metrics for a single session."""
        session_turns = self.get_session_turns(session_id)
        return self._aggregate(session_turns)

    def get_global_metrics(self) -> dict[str, Any]:
        """Aggregate performance and FinOps metrics across all sessions."""
        return self._aggregate(self._turns)

    def _aggregate(self, turns: list[TurnTelemetry]) -> dict[str, Any]:
        if not turns:
            return {
                "total_turns": 0,
                "total_latency_ms": 0.0,
                "average_latency_ms": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_bypasses": 0,
                "cache_hit_rate": 0.0,
                "hitl_escalations": 0,
                "fallback_turns": 0,
                "tools_invoked_count": 0,
                "tools_distribution": {},
            }

        total_turns = len(turns)
        total_latency = sum(t.latency_ms for t in turns)
        avg_latency = total_latency / total_turns
        total_in = sum(t.input_tokens for t in turns)
        total_out = sum(t.output_tokens for t in turns)
        total_tok = sum(t.total_tokens for t in turns)
        total_cost = sum(t.estimated_cost_usd for t in turns)

        cache_hits = sum(1 for t in turns if t.cache_status == "HIT")
        cache_misses = sum(1 for t in turns if t.cache_status == "MISS")
        cache_bypasses = sum(1 for t in turns if t.cache_status == "BYPASS_TRANSACTIONAL")
        cache_hit_rate = (cache_hits / total_turns) if total_turns > 0 else 0.0

        hitl_count = sum(1 for t in turns if t.requires_human_intervention)
        fallback_count = sum(1 for t in turns if t.fallback_used)

        tools_dist: dict[str, int] = {}
        for t in turns:
            for tool in t.tools_called:
                tools_dist[tool] = tools_dist.get(tool, 0) + 1

        return {
            "total_turns": total_turns,
            "total_latency_ms": round(total_latency, 2),
            "average_latency_ms": round(avg_latency, 2),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_tok,
            "total_cost_usd": round(total_cost, 6),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_bypasses": cache_bypasses,
            "cache_hit_rate": round(cache_hit_rate, 4),
            "hitl_escalations": hitl_count,
            "fallback_turns": fallback_count,
            "tools_invoked_count": sum(tools_dist.values()),
            "tools_distribution": tools_dist,
        }

    def clear(self) -> None:
        """Clear all recorded telemetry."""
        self._turns.clear()


_TELEMETRY_MANAGER_SINGLETON: TelemetryManager | None = None


def get_telemetry_manager() -> TelemetryManager:
    """Get or instantiate the singleton telemetry manager."""
    global _TELEMETRY_MANAGER_SINGLETON
    if _TELEMETRY_MANAGER_SINGLETON is None:
        _TELEMETRY_MANAGER_SINGLETON = TelemetryManager()
    return _TELEMETRY_MANAGER_SINGLETON


def reset_telemetry_manager() -> None:
    """Reset the singleton telemetry manager (for testing)."""
    global _TELEMETRY_MANAGER_SINGLETON
    _TELEMETRY_MANAGER_SINGLETON = None
