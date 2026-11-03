"""Streamlit web application for Empório da Música intelligent assistant."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import streamlit as st
from src.agent.graph import AgentTurnResult, EmporioAgent
from src.config import Settings, get_settings
from src.observability.telemetry import TurnTelemetry

logger = logging.getLogger(__name__)


# ============================================================================
# Helper & Formatting Utilities
# ============================================================================


def format_cache_badge(status: str) -> str:
    """Format the semantic cache status into a styled human-readable string."""
    stat = str(status).upper().strip()
    if stat == "HIT":
        return "🟢 HIT (Zero Token / <50ms)"
    if stat == "BYPASS_TRANSACTIONAL":
        return "🔵 BYPASS_TRANSACTIONAL (Tempo Real / PII)"
    return "🟡 MISS"


def format_usd_cost(cost: float) -> str:
    """Format USD inference cost with 6 decimal places."""
    return f"${cost:.6f}"


@st.cache_resource(show_spinner="Carregando modelos e catálogo...")
def _get_cached_agent() -> EmporioAgent:
    """Instantiate and cache the EmporioAgent singleton across Streamlit reruns."""
    return EmporioAgent()


def get_agent() -> EmporioAgent:
    """Return the active agent instance, allowing session_state override for testing."""
    if "custom_agent" in st.session_state and st.session_state["custom_agent"] is not None:
        return st.session_state["custom_agent"]
    return _get_cached_agent()


# ============================================================================
# Modular Component Renderers
# ============================================================================


def render_sidebar(
    settings: Settings,
    agent: EmporioAgent,
    session_id: str,
    last_telemetry: TurnTelemetry | None,
    hitl_tickets: list[dict[str, Any]],
) -> None:
    """Render the Observability, FinOps, and HITL sidebar."""
    with st.sidebar:
        st.title("📊 Observabilidade & FinOps")
        st.caption(
            f"Provedor Ativo: **{settings.llm_provider.upper()}** | "
            f"Modelo: `{settings.gemini_synthesizer_model}`"
        )
        st.divider()

        # Section 1: Live Turn Telemetry (Most Recent Turn)
        st.subheader("⚡ Telemetria do Último Turno")
        if last_telemetry is not None:
            st.markdown(f"**Cache Semântico:** {format_cache_badge(last_telemetry.cache_status)}")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Latência", f"{last_telemetry.latency_ms:.1f} ms")
            with col2:
                st.metric("Custo do Turno", format_usd_cost(last_telemetry.estimated_cost_usd))

            st.caption(
                f"**Tokens:** {last_telemetry.total_tokens} "
                f"(Prompt: {last_telemetry.input_tokens} | "
                f"Completion: {last_telemetry.output_tokens})"
            )

            if last_telemetry.tools_called:
                st.markdown("**Ferramentas Executadas:**")
                for t in last_telemetry.tools_called:
                    st.badge(f"🛠 {t}")
            else:
                st.caption("🛠 *Nenhuma ferramenta externa (Resposta direta ou Cache)*")

            if getattr(last_telemetry, "fallback_used", False):
                st.warning(
                    f"⚠️ **Contingência (Fallback):** Modelo secundário "
                    f"`{last_telemetry.provider}/{last_telemetry.model_name}` acionado!"
                )
        else:
            st.info("Aguardando primeira interação para registrar telemetria.")

        st.divider()

        # Section 2: Session Cumulative Metrics
        st.subheader("📈 Métricas Cumulativas da Sessão")
        session_metrics = agent.telemetry_manager.get_session_metrics(session_id)

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Turnos Totais", session_metrics["total_turns"])
            st.metric("Taxa Cache Hit", f"{session_metrics['cache_hit_rate'] * 100:.1f}%")
        with m_col2:
            st.metric("Custo Total (USD)", format_usd_cost(session_metrics["total_cost_usd"]))
            st.metric("Latência Média", f"{session_metrics['average_latency_ms']:.1f} ms")

        st.caption(
            f"**Distribuição do Cache:** "
            f"Hits: {session_metrics['cache_hits']} | "
            f"Misses: {session_metrics['cache_misses']} | "
            f"Bypasses: {session_metrics['cache_bypasses']}"
        )

        tools_dist = session_metrics.get("tools_distribution", {})
        if tools_dist:
            with st.expander("Distribuição de Ferramentas"):
                for tool_name, count in tools_dist.items():
                    st.write(f"• `{tool_name}`: **{count}**")

        st.divider()

        # Section 3: Human-in-the-Loop (HITL) Alert Card
        if hitl_tickets:
            st.subheader("🚨 Atendimento Humano (HITL)")
            st.error(f"**{len(hitl_tickets)} Protocolo(s) de Escalação Ativo(s)**")
            with st.expander("Ver Detalhes dos Protocolos", expanded=True):
                for idx, ticket in enumerate(hitl_tickets, 1):
                    st.markdown(f"**Ticket #{idx}:** `{ticket.get('ticket_id')}`")
                    st.markdown(f"• **Motivo:** {ticket.get('reason')}")
                    st.markdown(f"• **Urgência:** `{str(ticket.get('urgency_level', '')).upper()}`")
                    st.markdown(f"• **SLA:** até {ticket.get('contact_sla_hours', 24)} horas úteis")
                    st.markdown("---")

        st.divider()

        # Section 4: Session Controls
        if st.button("🧹 Nova Conversa / Limpar Sessão", use_container_width=True):
            st.session_state["session_id"] = f"web-{uuid.uuid4().hex[:8]}"
            st.session_state["messages"] = []
            st.session_state["hitl_tickets"] = []
            st.session_state["last_telemetry"] = None
            st.rerun()


def render_chat_history(messages: list[dict[str, Any]]) -> None:
    """Render conversation message history and any embedded escalation cards."""
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("ticket"):
                t = msg["ticket"]
                st.warning(
                    f"🚨 **Escalação para Atendimento Humano Registrada**\n\n"
                    f"• **Protocolo:** `{t.get('ticket_id')}`\n"
                    f"• **Motivo:** {t.get('reason')}\n"
                    f"• **Urgência:** `{str(t.get('urgency_level', '')).upper()}`\n"
                    f"• **Prazo de Retorno:** até {t.get('contact_sla_hours', 24)} horas úteis\n\n"
                    f"*Nossa equipe especializada entrará em contato ou você pode nos chamar "
                    f"diretamente no WhatsApp **(67) 3321-4500**.*"
                )


def process_user_turn(agent: EmporioAgent, query: str, session_id: str) -> AgentTurnResult:
    """Invoke the agent with the user's query and active session ID."""
    return agent.run(query=query, session_id=session_id)


# ============================================================================
# Application Entrypoint
# ============================================================================


def main() -> None:
    st.set_page_config(
        page_title="Empório da Música — Assistente Virtual",
        page_icon="🎸",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    settings = get_settings()
    agent = get_agent()

    # Session State Initialization
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = f"web-{uuid.uuid4().hex[:8]}"

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Olá! Seja muito bem-vindo(a) à **Empório da Música**! 🎶\n\n"
                    "Sou seu consultor virtual especialista em instrumentos musicais direto da "
                    "nossa loja física em **Campo Grande/MS** (Rua 14 de Maio, 3200 — Centro).\n\n"
                    "Como posso te ajudar hoje? Posso consultar nosso catálogo em tempo real, "
                    "verificar o estoque de violões e instrumentos, conferir promoções ativas, "
                    "rastrear seu pedido ou esclarecer dúvidas sobre nossas políticas!"
                ),
                "ticket": None,
            }
        ]

    if "last_telemetry" not in st.session_state:
        st.session_state["last_telemetry"] = None

    if "hitl_tickets" not in st.session_state:
        st.session_state["hitl_tickets"] = []

    # Render Sidebar
    render_sidebar(
        settings=settings,
        agent=agent,
        session_id=st.session_state["session_id"],
        last_telemetry=st.session_state["last_telemetry"],
        hitl_tickets=st.session_state["hitl_tickets"],
    )

    # Main Chat View
    st.title("🎸 Empório da Música")
    st.markdown(
        "**Assistente Consultivo Especialista em Instrumentos Musicais** • "
        "📍 *Loja Física: Rua 14 de Maio, 3200 — Centro, Campo Grande/MS*"
    )

    # Quick Suggestion Pills
    prompt_to_submit: str | None = None
    cols = st.columns(4)
    with cols[0]:
        if st.button("🎸 Violões até R$ 1.000", use_container_width=True):
            prompt_to_submit = "Quais violões vocês têm disponíveis por até 1000 reais?"
    with cols[1]:
        if st.button("🔍 Takamine GD20", use_container_width=True):
            prompt_to_submit = "Qual o preço e estoque do violão Takamine GD20?"
    with cols[2]:
        if st.button("📍 Endereço e Horário", use_container_width=True):
            prompt_to_submit = (
                "Onde fica a loja física em Campo Grande e qual o horário de funcionamento?"
            )
    with cols[3]:
        if st.button("🔄 Política de Devolução", use_container_width=True):
            prompt_to_submit = (
                "Como funciona a política de trocas e devoluções para compras online?"
            )

    st.divider()

    # Render Messages
    render_chat_history(st.session_state["messages"])

    # User Input
    user_input = st.chat_input("Digite sua dúvida sobre instrumentos, pedidos ou políticas...")
    final_query = prompt_to_submit or user_input

    if final_query:
        # Append User Message
        st.session_state["messages"].append(
            {"role": "user", "content": final_query, "ticket": None}
        )
        with st.chat_message("user"):
            st.markdown(final_query)

        # Process Turn
        with (
            st.chat_message("assistant"),
            st.spinner("Consultando dados e políticas da loja..."),
        ):
            turn_result = process_user_turn(
                agent=agent,
                query=final_query,
                session_id=st.session_state["session_id"],
            )

            st.markdown(turn_result.output_text)

            if turn_result.requires_human_intervention:
                t = turn_result.escalation_ticket or {}
                sla = t.get("contact_sla_hours", 24)
                st.warning(
                    f"🚨 **Escalação para Atendimento Humano Registrada**\n\n"
                    f"• **Protocolo:** `{t.get('ticket_id', 'TKT-PENDING')}`\n"
                    f"• **Motivo:** {t.get('reason', 'Solicitação fora da política padrão')}\n"
                    f"• **Urgência:** `{str(t.get('urgency_level', 'medium')).upper()}`\n"
                    f"• **Prazo de Retorno:** até {sla} horas úteis\n\n"
                    f"*Nossa equipe especializada entrará em contato ou você pode nos chamar "
                    f"diretamente no WhatsApp **(67) 3321-4500**.*"
                )

        # Update State & Telemetry
        st.session_state["last_telemetry"] = turn_result.telemetry
        if turn_result.requires_human_intervention and turn_result.escalation_ticket:
            st.session_state["hitl_tickets"].append(turn_result.escalation_ticket)

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": turn_result.output_text,
                "ticket": (
                    turn_result.escalation_ticket
                    if turn_result.requires_human_intervention
                    else None
                ),
            }
        )

        st.rerun()


if __name__ == "__main__":
    main()
