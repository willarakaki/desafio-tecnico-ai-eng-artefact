"""Stateful LangGraph orchestrator for Empório da Música assistant."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Generator
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field

from src.agent.llm_factory import LLMFactory, extract_model_identifiers
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.state import AgentState
from src.cache.semantic_cache import SemanticCache, get_semantic_cache
from src.config import Settings, get_settings
from src.database.duckdb_manager import DuckDBManager, get_duckdb_manager
from src.observability.telemetry import (
    TelemetryManager,
    TurnTelemetry,
    calculate_token_cost,
    get_telemetry_manager,
)
from src.rag.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from src.security.guardrails import InputGuardrail, get_input_guardrail
from src.security.pii_anonymizer import PIIAnonymizer, get_pii_anonymizer
from src.security.rate_limiter import SlidingWindowRateLimiter, get_rate_limiter
from src.tools.catalog_tools import create_catalog_tools
from src.tools.hitl_tools import create_hitl_tools, escalate_to_human
from src.tools.order_tools import create_order_tools
from src.tools.policy_tools import create_policy_tools

logger = logging.getLogger(__name__)

_PHANTOM_HITL_PATTERN = re.compile(
    r"(?:"
    r"vou\s+(?:registrar\s+[^.!?\n]{0,60}?\s+e\s+)?(?:transferir|encaminhar|acionar)|"
    r"j[aá]\s+(?:transferi|encaminhei|acionei|acionarei)|"
    r"estou\s+(?:transferindo|encaminhando|acionando)|"
    r"abri\s+um\s+chamado"
    r")\s+[^.!?\n]{0,90}?"
    r"(?:especialista\s+humano|equipe\s+humana|suporte\s+especializado|"
    r"atendente\s+humano|ger[eê]ncia|supervis[aã]o)",
    re.IGNORECASE,
)


def _claims_unbacked_human_transfer(text: str) -> bool:
    """Detect if LLM text promises an active human transfer without calling `escalate_to_human`."""
    return bool(_PHANTOM_HITL_PATTERN.search(text))


# ============================================================================
# High-Level Agent Turn Result Model
# ============================================================================


class AgentTurnResult(BaseModel):
    """Result of a single turn through the Empório da Música agent pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_text: str = Field(description="Final sanitized customer response in Portuguese.")
    session_id: str = Field(description="Conversation session identifier.")
    cache_status: str = Field(
        description="Semantic Cache status: 'HIT', 'MISS', or 'BYPASS_TRANSACTIONAL'."
    )
    requires_human_intervention: bool = Field(
        default=False, description="Whether human handoff was triggered."
    )
    escalation_ticket: dict[str, Any] | None = Field(
        default=None, description="Handoff ticket details if triggered."
    )
    guardrail_blocked: bool = Field(
        default=False, description="Whether the turn was rejected by security guardrails."
    )
    fallback_used: bool = Field(
        default=False, description="Whether fallback model was activated for this turn."
    )
    resolved_provider: str = Field(default="gemini", description="Provider that resolved the turn.")
    resolved_model: str = Field(
        default="gemini-2.5-flash", description="Model that resolved the turn."
    )
    tools_called: list[str] = Field(
        default_factory=list, description="List of tools invoked in this turn."
    )
    telemetry: TurnTelemetry = Field(
        description="Complete FinOps and latency telemetry for this turn."
    )
    messages: list[BaseMessage] = Field(
        default_factory=list, description="Updated message history."
    )


# ============================================================================
# Token Accounting Helpers
# ============================================================================


def extract_tokens(message: BaseMessage) -> tuple[int, int, int]:
    """Extract input, output, and total token usage from a LangChain message.

    Supports usage_metadata, response_metadata, and fallback heuristics.
    """
    usage = getattr(message, "usage_metadata", None)
    if usage and isinstance(usage, dict):
        inp = usage.get("input_tokens", 0) or 0
        out = usage.get("output_tokens", 0) or 0
        tot = usage.get("total_tokens", inp + out) or (inp + out)
        return inp, out, tot

    resp_meta = getattr(message, "response_metadata", {})
    if isinstance(resp_meta, dict):
        token_usage = resp_meta.get("token_usage") or resp_meta.get("usage")
        if isinstance(token_usage, dict):
            inp = token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0
            out = token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
            tot = token_usage.get("total_tokens") or (inp + out)
            return inp, out, tot

    # Heuristic approximation if provider omits token stats
    text = getattr(message, "content", "")
    if isinstance(text, str) and text:
        approx = max(1, len(text.split()))
        return approx, approx, approx * 2

    return 0, 0, 0


# ============================================================================
# Graph Builder
# ============================================================================


def get_default_tools(
    duckdb_manager: DuckDBManager | None = None,
    retriever: HybridRetriever | None = None,
) -> list[StructuredTool]:
    """Construct the standard tool suite for Empório da Música."""
    tools: list[StructuredTool] = []
    tools.extend(create_catalog_tools(db=duckdb_manager))
    tools.extend(create_order_tools(db=duckdb_manager))
    tools.extend(create_policy_tools(retriever=retriever))
    tools.extend(create_hitl_tools())
    return tools


def build_emporio_graph(
    model: Any,
    tools: list[StructuredTool],
    rate_limiter: SlidingWindowRateLimiter,
    guardrail: InputGuardrail,
    semantic_cache: SemanticCache,
    anonymizer: PIIAnonymizer,
    telemetry_manager: TelemetryManager,
    fallback_models: list[Any] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    """Build and compile the stateful LangGraph orchestrator.

    Args:
        model: BaseChatModel bound or bindable to tools.
        tools: List of executable StructuredTool instances.
        rate_limiter: In-memory sliding window rate limiter.
        guardrail: Multi-tier input guardrails for security and scope.
        semantic_cache: Context-aware semantic cache with bypass.
        anonymizer: Presidio PII mask for LGPD compliance.
        telemetry_manager: FinOps and latency aggregator.
        checkpointer: LangGraph checkpoint saver (defaults to MemorySaver).
        settings: Application settings.

    Returns:
        CompiledStateGraph ready for execution.
    """
    active_settings = settings or get_settings()
    provider_name, model_name = extract_model_identifiers(
        model, default_provider=active_settings.llm_provider
    )

    tools_by_name = {t.name: t for t in tools}

    # Build resilient model chain with fallbacks and tool bindings
    bound_model = LLMFactory.create_resilient_chain(
        primary=model,
        fallbacks=fallback_models,
        tools=tools,
        primary_provider=provider_name,
    )

    workflow = StateGraph(AgentState)

    # ------------------------------------------------------------------------
    # Node 1: Input Barrier (Rate Limiter, Input Guardrails & Semantic Cache)
    # ------------------------------------------------------------------------
    def input_barrier_node(state: AgentState) -> dict[str, Any]:
        session_id = state.get("session_id", "default")
        messages = state.get("messages", [])
        if not messages:
            return {"guardrail_blocked": False, "cache_status": "MISS"}

        # Find latest human message
        latest_human_msg = None
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                latest_human_msg = m
                break

        if latest_human_msg is None:
            return {"guardrail_blocked": False, "cache_status": "MISS"}

        query = str(latest_human_msg.content).strip()

        # 1. Rate limiter check
        rl_res = rate_limiter.check(session_id=session_id, message=query)
        if not rl_res.is_allowed:
            refusal = (
                "Você atingiu o limite de mensagens permitidas por minuto. "
                f"Por favor, aguarde {int(rl_res.retry_after_seconds)} segundos."
            )
            return {
                "guardrail_blocked": True,
                "guardrail_message": refusal,
                "output_text": refusal,
                "messages": [AIMessage(content=refusal)],
            }

        # 2. Input guardrails check
        gr_res = guardrail.validate(message=query)
        if not gr_res.is_safe:
            refusal = gr_res.refusal_message or (
                "Como assistente da Empório da Música, posso ajudar apenas com "
                "informações sobre nossos instrumentos musicais e compras."
            )
            return {
                "guardrail_blocked": True,
                "guardrail_message": refusal,
                "output_text": refusal,
                "messages": [AIMessage(content=refusal)],
            }

        # 3. Context-Aware Semantic Cache check
        lookup = semantic_cache.get(query)
        lookup_status = str(lookup.status)
        if lookup_status == "HIT" and lookup.response:
            return {
                "guardrail_blocked": False,
                "cache_status": "HIT",
                "output_text": lookup.response,
                "messages": [AIMessage(content=lookup.response)],
            }
        if lookup_status == "BYPASS_TRANSACTIONAL":
            return {"guardrail_blocked": False, "cache_status": "BYPASS_TRANSACTIONAL"}

        return {"guardrail_blocked": False, "cache_status": "MISS"}

    # ------------------------------------------------------------------------
    # Node 3: Orchestrator LLM
    # ------------------------------------------------------------------------
    def orchestrator_node(state: AgentState) -> dict[str, Any]:
        raw_messages = list(state.get("messages", []))

        # Ensure SystemPrompt is prepended
        has_system = any(isinstance(m, SystemMessage) for m in raw_messages)
        messages_to_send: list[BaseMessage] = []
        if not has_system:
            messages_to_send.append(SystemMessage(content=SYSTEM_PROMPT.strip()))
        messages_to_send.extend(raw_messages)

        try:
            response = bound_model.invoke(messages_to_send)
        except Exception as exc:
            logger.error("All LLM providers failed in orchestrator: %s", exc)
            from src.tools.hitl_tools import escalate_to_human

            escalation_res = escalate_to_human(
                reason="Falha total de provedores de IA (indisponibilidade técnica)",
                case_summary=(
                    "Incapacidade temporária de gerar resposta automática "
                    f"por falha em todos os modelos: {exc}"
                ),
                urgency_level="critical",
            )
            ticket_data = escalation_res.ticket
            fallback_msg = (
                "No momento, nossos servidores inteligentes estão enfrentando uma "
                "instabilidade momentânea. Para garantir seu atendimento, transferi sua "
                "solicitação diretamente para nossa equipe humana no WhatsApp (67) 3321-4500 "
                f"(Protocolo: {ticket_data.ticket_id})."
            )
            return {
                "messages": [AIMessage(content=fallback_msg)],
                "output_text": fallback_msg,
                "fallback_used": True,
                "resolved_provider": "all_failed_hitl",
                "resolved_model": "hitl_fallback",
                "requires_human_intervention": True,
                "escalation_ticket": ticket_data.model_dump(),
            }

        res_meta = getattr(response, "response_metadata", {}) or {}
        res_provider = str(res_meta.get("provider") or provider_name)
        res_model = str(res_meta.get("model_name") or model_name)
        primary_prov, _ = extract_model_identifiers(model, default_provider=provider_name)
        was_fallback = state.get("fallback_used", False) or (res_provider != primary_prov)

        in_tok, out_tok, tot_tok = extract_tokens(response)
        prev_in = state.get("input_tokens", 0)
        prev_out = state.get("output_tokens", 0)

        return {
            "messages": [response],
            "input_tokens": prev_in + in_tok,
            "output_tokens": prev_out + out_tok,
            "total_tokens": prev_in + prev_out + tot_tok,
            "fallback_used": was_fallback,
            "resolved_provider": res_provider,
            "resolved_model": res_model,
        }

    # ------------------------------------------------------------------------
    # Node 4: Deterministic Tools Execution
    # ------------------------------------------------------------------------
    def tools_node(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {}

        last_message = messages[-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        if not tool_calls:
            return {}

        new_tools_called = list(state.get("tools_called", []))
        requires_hitl = state.get("requires_human_intervention", False)
        escalation_ticket = state.get("escalation_ticket", None)
        tool_messages: list[ToolMessage] = []

        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {})
            call_id = tc.get("id", str(uuid.uuid4()))

            if name and name not in new_tools_called:
                new_tools_called.append(name)

            tool = tools_by_name.get(name)
            if tool:
                try:
                    res = tool.invoke(args)
                except Exception as exc:
                    logger.error("Error executing tool '%s': %s", name, exc)
                    res = f"Erro na execução da ferramenta {name}: {exc}"
            else:
                res = f"Ferramenta desconhecida: '{name}'."

            # Check if escalate_to_human was called
            if name == "escalate_to_human":
                requires_hitl = True
                try:
                    import json

                    if isinstance(res, str):
                        data = json.loads(res)
                        escalation_ticket = data.get("ticket", data)
                    elif isinstance(res, dict):
                        escalation_ticket = res.get("ticket", res)
                except Exception:
                    escalation_ticket = {
                        "status": "aberto",
                        "reason": str(args.get("reason", "Escalação humana")),
                    }

            tool_messages.append(ToolMessage(content=str(res), tool_call_id=call_id, name=name))

        return {
            "messages": tool_messages,
            "tools_called": new_tools_called,
            "requires_human_intervention": requires_hitl,
            "escalation_ticket": escalation_ticket,
        }

    # ------------------------------------------------------------------------
    # Node 5: Finalize, PII Mask & FinOps Telemetry
    # ------------------------------------------------------------------------
    def finalize_node(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        session_id = state.get("session_id", "default")
        cache_status = state.get("cache_status", "MISS")
        guardrail_blocked = state.get("guardrail_blocked", False)
        tools_called = state.get("tools_called", [])
        requires_hitl = state.get("requires_human_intervention", False)
        ticket = state.get("escalation_ticket")
        ticket_id = ticket.get("ticket_id") if isinstance(ticket, dict) else None

        # Extract output text
        output_text = state.get("output_text")
        if not output_text and messages:
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage):
                output_text = str(last_msg.content)
            else:
                output_text = "Atendimento finalizado com sucesso."

        # Apply PII Anonymizer to customer response
        sanitized_output = anonymizer.anonymize(output_text or "")

        # Extract latest query to populate cache if eligible
        latest_query = None
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                latest_query = str(m.content).strip()
                break

        # Deterministic anti-phantom-HITL guard: if the LLM promised an active human
        # transfer in text without invoking `escalate_to_human`, generate a real ticket.
        if (
            not requires_hitl
            and not guardrail_blocked
            and cache_status != "HIT"
            and _claims_unbacked_human_transfer(sanitized_output)
        ):
            logger.warning(
                "Detected unbacked human transfer claim in session '%s'; auto-creating HITL ticket.",
                session_id,
            )
            auto_escalation = escalate_to_human(
                reason="Transferência para especialista solicitada durante síntese consultiva",
                case_summary=anonymizer.anonymize(latest_query or sanitized_output[:240]),
                urgency_level="medium",
            )
            requires_hitl = True
            ticket = auto_escalation.ticket.model_dump()
            ticket_id = auto_escalation.ticket.ticket_id
            tools_called = [*tools_called, "escalate_to_human"]
            sanitized_output = (
                f"{sanitized_output.rstrip()}\n\n"
                f"📋 **Protocolo registrado:** `{ticket_id}` "
                "(nossa equipe humana assumirá o atendimento em até 24 horas úteis)."
            )

        # Populate cache on eligible cache MISS
        can_cache = False
        if latest_query:
            should_by, _ = semantic_cache.check_bypass(latest_query)
            can_cache = not should_by

        # Strict data freshness guard: Never cache responses generated from dynamic database lookups or human escalation
        transactional_tool_names = {
            "search_products",
            "get_product_details",
            "get_active_promotions",
            "get_order_status",
            "escalate_to_human",
        }
        if any(t in transactional_tool_names for t in tools_called):
            can_cache = False

        if (
            cache_status == "MISS"
            and latest_query
            and not guardrail_blocked
            and not requires_hitl
            and can_cache
        ):
            try:
                semantic_cache.set(query=latest_query, response=sanitized_output)
            except Exception as exc:
                logger.warning("Could not store response in semantic cache: %s", exc)

        # FinOps Token Accounting
        if cache_status == "HIT" or guardrail_blocked:
            inp_tok = 0
            out_tok = 0
            tot_tok = 0
            cost_usd = 0.0
        else:
            inp_tok = state.get("input_tokens", 0)
            out_tok = state.get("output_tokens", 0)
            tot_tok = inp_tok + out_tok
            cost_usd = calculate_token_cost(model_name, inp_tok, out_tok)

        latency_ms = state.get("latency_ms", 0.0)

        guardrail_status: Literal["passed", "blocked", "rate_limited"] = "passed"
        if guardrail_blocked:
            guardrail_status = "blocked"

        telemetry = TurnTelemetry(
            session_id=session_id,
            latency_ms=round(latency_ms, 2),
            cache_status=cache_status,  # type: ignore[arg-type]
            tools_called=tools_called,
            input_tokens=inp_tok,
            output_tokens=out_tok,
            total_tokens=tot_tok,
            estimated_cost_usd=cost_usd,
            requires_human_intervention=requires_hitl,
            escalation_ticket_id=ticket_id,
            guardrail_status=guardrail_status,
            provider=str(state.get("resolved_provider") or provider_name),
            model_name=str(state.get("resolved_model") or model_name),
            fallback_used=state.get("fallback_used", False),
        )

        telemetry_manager.record_turn(telemetry)

        return {
            "output_text": sanitized_output,
            "estimated_cost_usd": cost_usd,
            "requires_human_intervention": requires_hitl,
            "escalation_ticket": ticket,
            "tools_called": tools_called,
        }

    # ------------------------------------------------------------------------
    # Edges & Conditional Routing
    # ------------------------------------------------------------------------
    def route_after_input_barrier(state: AgentState) -> str:
        if state.get("guardrail_blocked") or state.get("cache_status") == "HIT":
            return "finalize"
        return "orchestrator"

    def route_after_orchestrator(state: AgentState) -> str:
        if state.get("requires_human_intervention") and state.get("output_text"):
            return "finalize"
        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            if getattr(last, "tool_calls", None):
                return "tools"
        return "finalize"

    workflow.add_node("input_barrier", input_barrier_node)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("finalize", finalize_node)

    workflow.add_edge(START, "input_barrier")
    workflow.add_conditional_edges(
        "input_barrier",
        route_after_input_barrier,
        {"finalize": "finalize", "orchestrator": "orchestrator"},
    )
    workflow.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {"tools": "tools", "finalize": "finalize"},
    )
    workflow.add_edge("tools", "orchestrator")
    workflow.add_edge("finalize", END)

    saver = checkpointer if checkpointer is not None else MemorySaver()
    return workflow.compile(checkpointer=saver)


# ============================================================================
# High-Level Agent Runner
# ============================================================================


class EmporioAgent:
    """Production-grade agent runner encapsulating the LangGraph workflow.

    Provides a clean .run() entrypoint with thread-based session memory,
    rate limiting, caching, PII masking, and telemetry tracking.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        model: Any | None = None,
        fallback_models: list[Any] | None = None,
        tools: list[StructuredTool] | None = None,
        duckdb_manager: DuckDBManager | None = None,
        retriever: HybridRetriever | None = None,
        rate_limiter: SlidingWindowRateLimiter | None = None,
        guardrail: InputGuardrail | None = None,
        semantic_cache: SemanticCache | None = None,
        anonymizer: PIIAnonymizer | None = None,
        telemetry_manager: TelemetryManager | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        recursion_limit: int = 16,
    ) -> None:
        self.settings = settings or get_settings()
        logging.getLogger("src").setLevel(
            getattr(logging, self.settings.log_level.upper(), logging.INFO)
        )
        self.duckdb_manager = duckdb_manager or get_duckdb_manager(data_dir=self.settings.data_dir)
        self.retriever = retriever or get_hybrid_retriever()
        self.rate_limiter = rate_limiter or get_rate_limiter(settings=self.settings)
        self.guardrail = guardrail or get_input_guardrail()
        self.semantic_cache = semantic_cache or get_semantic_cache(settings=self.settings)
        self.anonymizer = anonymizer or get_pii_anonymizer()
        self.telemetry_manager = telemetry_manager or get_telemetry_manager()
        self.checkpointer = checkpointer or MemorySaver()
        self.recursion_limit = recursion_limit

        self.model = model or LLMFactory(settings=self.settings).get_synthesizer_llm()
        if fallback_models is not None:
            self.fallback_models = fallback_models
        elif model is None:
            self.fallback_models = LLMFactory(settings=self.settings).get_fallback_llms()
        else:
            self.fallback_models = []
        self.tools = (
            tools
            if tools is not None
            else get_default_tools(
                duckdb_manager=self.duckdb_manager,
                retriever=self.retriever,
            )
        )

        self.graph = build_emporio_graph(
            model=self.model,
            tools=self.tools,
            rate_limiter=self.rate_limiter,
            guardrail=self.guardrail,
            semantic_cache=self.semantic_cache,
            anonymizer=self.anonymizer,
            telemetry_manager=self.telemetry_manager,
            fallback_models=self.fallback_models,
            checkpointer=self.checkpointer,
            settings=self.settings,
        )

    def run(
        self,
        query: str,
        session_id: str = "default",
    ) -> AgentTurnResult:
        """Process a single customer turn with stateful conversational memory.

        Args:
            query: User's natural language input string.
            session_id: Thread identifier for conversational session state.

        Returns:
            Structured AgentTurnResult with output text, telemetry, and execution flags.
        """
        start_time = time.perf_counter()

        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "session_id": session_id,
            "cache_status": "MISS",
            "requires_human_intervention": False,
            "escalation_ticket": None,
            "guardrail_blocked": False,
            "guardrail_message": None,
            "tools_called": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "latency_ms": 0.0,
            "output_text": None,
        }

        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": self.recursion_limit,
        }

        try:
            final_state = self.graph.invoke(initial_state, config=config)
        except GraphRecursionError:
            logger.warning(
                "Graph recursion limit (%d) reached for session '%s'",
                self.recursion_limit,
                session_id,
            )
            escalation_res = escalate_to_human(
                reason="Solicitação multietapas excedeu o limite de iterações automáticas",
                case_summary=self.anonymizer.anonymize(query[:300]),
                urgency_level="high",
            )
            ticket_data = escalation_res.ticket
            fallback_msg = (
                "Sua solicitação exigiu múltiplas validações cruzadas além do limite automático. "
                "Para garantir que seu caso seja resolvido com precisão, registrei o protocolo "
                f"**{ticket_data.ticket_id}** para um especialista humano da nossa equipe assumir "
                "neste atendimento em até 24 horas úteis (ou, se preferir contato imediato, "
                "nosso WhatsApp direto é (67) 3321-4500)."
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            telemetry = TurnTelemetry(
                session_id=session_id,
                latency_ms=round(elapsed_ms, 2),
                cache_status="MISS",
                tools_called=["recursion_limit_fallback", "escalate_to_human"],
                requires_human_intervention=True,
                escalation_ticket_id=ticket_data.ticket_id,
                provider=self.settings.llm_provider,
                model_name="recursion_fallback",
            )
            self.telemetry_manager.record_turn(telemetry)
            return AgentTurnResult(
                output_text=fallback_msg,
                session_id=session_id,
                cache_status="MISS",
                requires_human_intervention=True,
                escalation_ticket=ticket_data.model_dump(),
                guardrail_blocked=False,
                tools_called=["recursion_limit_fallback", "escalate_to_human"],
                telemetry=telemetry,
                messages=[HumanMessage(content=query), AIMessage(content=fallback_msg)],
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Retrieve the latest telemetry recorded for this session
        session_turns = self.telemetry_manager.get_session_turns(session_id)
        if session_turns:
            telemetry = session_turns[-1]
            telemetry.latency_ms = round(elapsed_ms, 2)
        else:
            telemetry = TurnTelemetry(
                session_id=session_id,
                latency_ms=round(elapsed_ms, 2),
                cache_status=final_state.get("cache_status", "MISS"),
                tools_called=final_state.get("tools_called", []),
                fallback_used=final_state.get("fallback_used", False),
                provider=str(final_state.get("resolved_provider") or self.settings.llm_provider),
                model_name=str(
                    final_state.get("resolved_model") or self.settings.gemini_synthesizer_model
                ),
                requires_human_intervention=final_state.get("requires_human_intervention", False),
            )

        return AgentTurnResult(
            output_text=final_state.get("output_text") or "Atendimento concluído.",
            session_id=session_id,
            cache_status=final_state.get("cache_status", "MISS"),
            requires_human_intervention=final_state.get("requires_human_intervention", False),
            escalation_ticket=final_state.get("escalation_ticket"),
            guardrail_blocked=final_state.get("guardrail_blocked", False),
            tools_called=final_state.get("tools_called", []),
            fallback_used=final_state.get("fallback_used", False),
            resolved_provider=str(
                final_state.get("resolved_provider") or self.settings.llm_provider
            ),
            resolved_model=str(
                final_state.get("resolved_model") or self.settings.gemini_synthesizer_model
            ),
            telemetry=telemetry,
            messages=list(final_state.get("messages", [])),
        )

    def stream(
        self,
        query: str,
        session_id: str = "default",
    ) -> Generator[dict[str, Any], None, None]:
        """Stream a customer turn yielding step events, tool calls, and text tokens.

        Yields dictionaries with format:
            {"event": "<event_type>", "data": { ... }}

        Where <event_type> is one of:
            - 'step': Graph workflow step progress
            - 'tool_call': Name and arguments of tools being invoked
            - 'tool_result': Summary of tool execution
            - 'token': Incremental text chunks of the assistant response
            - 'done': Final turn result, telemetry, and execution flags
            - 'error': Graph execution or timeout error details
        """
        start_time = time.perf_counter()

        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "session_id": session_id,
            "cache_status": "MISS",
            "requires_human_intervention": False,
            "escalation_ticket": None,
            "guardrail_blocked": False,
            "guardrail_message": None,
            "tools_called": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "latency_ms": 0.0,
            "output_text": None,
        }

        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": self.recursion_limit,
        }

        yield {
            "event": "step",
            "data": {"step": "start", "session_id": session_id},
        }

        final_state: dict[str, Any] = {}
        try:
            for step_update in self.graph.stream(
                initial_state, config=config, stream_mode="updates"
            ):
                for node_name, node_state in step_update.items():
                    final_state.update(node_state)
                    if node_name in ("input_barrier", "guardrail_and_cache"):
                        if node_state.get("guardrail_blocked"):
                            yield {
                                "event": "step",
                                "data": {
                                    "step": "guardrail",
                                    "status": "blocked",
                                    "message": node_state.get("guardrail_message"),
                                },
                            }
                        elif node_state.get("cache_status") == "HIT":
                            yield {
                                "event": "step",
                                "data": {"step": "cache", "status": "HIT"},
                            }
                        else:
                            yield {
                                "event": "step",
                                "data": {
                                    "step": "guardrail_and_cache",
                                    "status": "passed",
                                    "cache_status": node_state.get("cache_status", "MISS"),
                                },
                            }
                    elif node_name == "orchestrator":
                        messages = node_state.get("messages", [])
                        if messages:
                            last_msg = messages[-1]
                            tool_calls = getattr(last_msg, "tool_calls", [])
                            if tool_calls:
                                for tc in tool_calls:
                                    yield {
                                        "event": "tool_call",
                                        "data": {
                                            "name": tc.get("name"),
                                            "args": tc.get("args", {}),
                                        },
                                    }
                            else:
                                yield {
                                    "event": "step",
                                    "data": {"step": "orchestrator", "status": "synthesized"},
                                }
                    elif node_name == "tools":
                        yield {
                            "event": "tool_result",
                            "data": {
                                "tools_called": node_state.get("tools_called", []),
                                "requires_human_intervention": node_state.get(
                                    "requires_human_intervention", False
                                ),
                            },
                        }
                    elif node_name == "finalize":
                        yield {
                            "event": "step",
                            "data": {"step": "finalize", "status": "completed"},
                        }
        except GraphRecursionError:
            logger.warning(
                "Graph recursion limit (%d) reached for session '%s' during streaming",
                self.recursion_limit,
                session_id,
            )
            escalation_res = escalate_to_human(
                reason="Solicitação multietapas excedeu o limite de iterações automáticas",
                case_summary=self.anonymizer.anonymize(query[:300]),
                urgency_level="high",
            )
            ticket_data = escalation_res.ticket
            fallback_msg = (
                "Sua solicitação exigiu múltiplas validações cruzadas além do limite automático. "
                "Para garantir que seu caso seja resolvido com precisão, registrei o protocolo "
                f"**{ticket_data.ticket_id}** para um especialista humano da nossa equipe assumir "
                "neste atendimento em até 24 horas úteis (ou, se preferir contato imediato, "
                "nosso WhatsApp direto é (67) 3321-4500)."
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            telemetry = TurnTelemetry(
                session_id=session_id,
                latency_ms=round(elapsed_ms, 2),
                cache_status="MISS",
                tools_called=["recursion_limit_fallback", "escalate_to_human"],
                requires_human_intervention=True,
                escalation_ticket_id=ticket_data.ticket_id,
                provider=self.settings.llm_provider,
                model_name="recursion_fallback",
            )
            self.telemetry_manager.record_turn(telemetry)
            for chunk in _chunk_text(fallback_msg):
                yield {"event": "token", "data": {"token": chunk}}
            yield {
                "event": "done",
                "data": AgentTurnResult(
                    output_text=fallback_msg,
                    session_id=session_id,
                    cache_status="MISS",
                    requires_human_intervention=True,
                    escalation_ticket=ticket_data.model_dump(),
                    guardrail_blocked=False,
                    tools_called=["recursion_limit_fallback", "escalate_to_human"],
                    telemetry=telemetry,
                    messages=[HumanMessage(content=query), AIMessage(content=fallback_msg)],
                ).model_dump(),
            }
            return
        except Exception as exc:
            logger.error("Error during agent stream: %s", exc)
            yield {"event": "error", "data": {"error": str(exc)}}
            return

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        session_turns = self.telemetry_manager.get_session_turns(session_id)
        if session_turns:
            telemetry = session_turns[-1]
            telemetry.latency_ms = round(elapsed_ms, 2)
        else:
            telemetry = TurnTelemetry(
                session_id=session_id,
                latency_ms=round(elapsed_ms, 2),
                cache_status=final_state.get("cache_status", "MISS"),
                tools_called=final_state.get("tools_called", []),
                fallback_used=final_state.get("fallback_used", False),
                provider=str(final_state.get("resolved_provider") or self.settings.llm_provider),
                model_name=str(
                    final_state.get("resolved_model") or self.settings.gemini_synthesizer_model
                ),
                requires_human_intervention=final_state.get("requires_human_intervention", False),
            )

        output_text = final_state.get("output_text") or "Atendimento concluído."

        for chunk in _chunk_text(output_text):
            yield {"event": "token", "data": {"token": chunk}}

        turn_result = AgentTurnResult(
            output_text=output_text,
            session_id=session_id,
            cache_status=final_state.get("cache_status", "MISS"),
            requires_human_intervention=final_state.get("requires_human_intervention", False),
            escalation_ticket=final_state.get("escalation_ticket"),
            guardrail_blocked=final_state.get("guardrail_blocked", False),
            tools_called=final_state.get("tools_called", []),
            fallback_used=final_state.get("fallback_used", False),
            resolved_provider=str(
                final_state.get("resolved_provider") or self.settings.llm_provider
            ),
            resolved_model=str(
                final_state.get("resolved_model") or self.settings.gemini_synthesizer_model
            ),
            telemetry=telemetry,
            messages=list(final_state.get("messages", [])),
        )

        yield {
            "event": "done",
            "data": turn_result.model_dump(),
        }


def _chunk_text(text: str, chunk_size: int = 4) -> list[str]:
    """Split text into conversational word chunks for progressive streaming."""
    words = text.split(" ")
    chunks: list[str] = []
    current: list[str] = []
    for i, w in enumerate(words):
        current.append(w)
        if len(current) >= chunk_size or i == len(words) - 1:
            chunk = " ".join(current)
            if i < len(words) - 1:
                chunk += " "
            chunks.append(chunk)
            current = []
    return chunks if chunks else [text]
