"""State schema definition for the Empório da Música LangGraph orchestrator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """State schema for the Empório da Música conversational graph.

    Maintains messages with LangGraph's `add_messages` reducer alongside
    observability, caching, security flags, and HITL ticket structures.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: str
    cache_status: str  # "HIT", "MISS", or "BYPASS_TRANSACTIONAL"
    requires_human_intervention: bool
    escalation_ticket: dict[str, Any] | None
    guardrail_blocked: bool
    guardrail_message: str | None
    tools_called: list[str]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    latency_ms: float
    output_text: str | None
