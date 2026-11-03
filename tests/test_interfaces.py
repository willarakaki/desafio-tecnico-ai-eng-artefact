"""Tests for user interaction interfaces: Rich CLI and Streamlit web application."""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import MagicMock, patch

from rich.console import Console
from src.agent.graph import AgentTurnResult
from src.config import Settings
from src.observability.telemetry import TurnTelemetry

# ============================================================================
# Dummy Fixtures & Helpers
# ============================================================================


def create_dummy_turn_result(
    output_text: str = "Olá! Como posso ajudar você hoje?",
    session_id: str = "test-session",
    cache_status: str = "MISS",
    requires_human_intervention: bool = False,
    escalation_ticket: dict[str, Any] | None = None,
    guardrail_blocked: bool = False,
    tools_called: list[str] | None = None,
    latency_ms: float = 120.5,
    input_tokens: int = 150,
    output_tokens: int = 50,
    estimated_cost_usd: float = 0.000026,
) -> AgentTurnResult:
    """Helper to construct a realistic AgentTurnResult for interface testing."""
    tools = tools_called or []
    tot_tok = input_tokens + output_tokens
    telemetry = TurnTelemetry(
        session_id=session_id,
        latency_ms=latency_ms,
        cache_status=cache_status,  # type: ignore[arg-type]
        tools_called=tools,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=tot_tok,
        estimated_cost_usd=estimated_cost_usd,
        requires_human_intervention=requires_human_intervention,
        escalation_ticket_id=escalation_ticket.get("ticket_id") if escalation_ticket else None,
        guardrail_status="blocked" if guardrail_blocked else "passed",
    )
    return AgentTurnResult(
        output_text=output_text,
        session_id=session_id,
        cache_status=cache_status,
        requires_human_intervention=requires_human_intervention,
        escalation_ticket=escalation_ticket,
        guardrail_blocked=guardrail_blocked,
        tools_called=tools,
        telemetry=telemetry,
        messages=[],
    )


# ============================================================================
# Rich CLI Tests
# ============================================================================


def test_cli_initialization():
    """Verify that RichCLI initializes with dedicated session ID and console."""
    from cli import RichCLI

    mock_agent = MagicMock()
    cli = RichCLI(agent=mock_agent, session_id="custom-session-123")
    assert cli.session_id == "custom-session-123"
    assert cli.agent is mock_agent
    assert isinstance(cli.console, Console)


def test_cli_commands_exit():
    """Verify that /exit and /quit commands signal loop termination."""
    from cli import RichCLI

    mock_agent = MagicMock()
    console = Console(file=io.StringIO(), record=True)
    cli = RichCLI(agent=mock_agent, console=console)

    should_continue_exit = cli.handle_command("/exit")
    assert should_continue_exit is False
    output = console.export_text()
    assert "Atendimento encerrado" in output or "Até logo" in output

    console.clear()
    should_continue_quit = cli.handle_command("/quit")
    assert should_continue_quit is False


def test_cli_commands_clear():
    """Verify that /clear resets conversation state and rotates session ID."""
    from cli import RichCLI

    mock_agent = MagicMock()
    console = Console(file=io.StringIO(), record=True)
    cli = RichCLI(agent=mock_agent, session_id="session-alpha", console=console)

    original_session = cli.session_id
    should_continue = cli.handle_command("/clear")
    assert should_continue is True
    assert cli.session_id != original_session
    output = console.export_text()
    assert "Sessão reiniciada" in output or "limpa" in output


def test_cli_commands_help():
    """Verify that /help displays available commands."""
    from cli import RichCLI

    mock_agent = MagicMock()
    console = Console(file=io.StringIO(), record=True)
    cli = RichCLI(agent=mock_agent, console=console)

    should_continue = cli.handle_command("/help")
    assert should_continue is True
    output = console.export_text()
    assert "/exit" in output
    assert "/clear" in output
    assert "/metrics" in output


def test_cli_commands_metrics():
    """Verify that /metrics renders a summary table from the telemetry manager."""
    from cli import RichCLI

    mock_agent = MagicMock()
    mock_agent.telemetry_manager.get_session_metrics.return_value = {
        "total_turns": 4,
        "total_latency_ms": 320.0,
        "average_latency_ms": 80.0,
        "total_input_tokens": 600,
        "total_output_tokens": 200,
        "total_tokens": 800,
        "total_cost_usd": 0.000105,
        "cache_hits": 2,
        "cache_misses": 1,
        "cache_bypasses": 1,
        "cache_hit_rate": 0.50,
        "hitl_escalations": 1,
        "tools_invoked_count": 2,
        "tools_distribution": {"search_products": 1, "consult_store_policies": 1},
    }

    console = Console(file=io.StringIO(), record=True)
    cli = RichCLI(agent=mock_agent, session_id="metrics-session", console=console)

    should_continue = cli.handle_command("/metrics")
    assert should_continue is True
    output = console.export_text()
    assert "Métricas da Sessão" in output or "FinOps" in output
    assert "800" in output
    assert "search_products" in output


def test_cli_process_turn_regular():
    """Verify standard turn processing and live telemetry panel output."""
    from cli import RichCLI

    mock_agent = MagicMock()
    mock_agent.run.return_value = create_dummy_turn_result(
        output_text="Temos o Takamine GD20 em estoque por R$ 3.200,00.",
        cache_status="MISS",
        tools_called=["get_product_details"],
        latency_ms=95.2,
    )

    console = Console(file=io.StringIO(), record=True)
    cli = RichCLI(agent=mock_agent, session_id="regular-session", console=console)

    result = cli.process_turn("Tem Takamine GD20?")
    assert result.output_text == "Temos o Takamine GD20 em estoque por R$ 3.200,00."
    assert "get_product_details" in result.tools_called

    output = console.export_text()
    assert "Takamine GD20" in output
    assert "MISS" in output
    assert "95.2" in output or "95ms" in output or "get_product_details" in output


def test_cli_process_turn_cache_hit():
    """Verify that Cache HIT is visually formatted in the CLI telemetry panel."""
    from cli import RichCLI

    mock_agent = MagicMock()
    mock_agent.run.return_value = create_dummy_turn_result(
        output_text="Nossa loja física fica na Rua 14 de Maio, 3200 — Centro, Campo Grande/MS.",
        cache_status="HIT",
        tools_called=[],
        latency_ms=18.4,
        input_tokens=0,
        output_tokens=0,
        estimated_cost_usd=0.0,
    )

    console = Console(file=io.StringIO(), record=True)
    cli = RichCLI(agent=mock_agent, session_id="cache-session", console=console)

    cli.process_turn("Qual o endereço da loja física?")
    output = console.export_text()
    assert "Rua 14 de Maio, 3200" in output
    assert "HIT" in output


def test_cli_process_turn_hitl_alert():
    """Verify that a Human-in-the-Loop escalation renders an alert card in the CLI."""
    from cli import RichCLI

    ticket = {
        "ticket_id": "TKT-20261102-TEST",
        "reason": "Exceção de devolução fora do prazo",
        "urgency_level": "medium",
        "contact_sla_hours": 24,
    }
    mock_agent = MagicMock()
    mock_agent.run.return_value = create_dummy_turn_result(
        output_text="Encaminhei seu caso para nossa gerência humana.",
        requires_human_intervention=True,
        escalation_ticket=ticket,
        tools_called=["escalate_to_human"],
    )

    console = Console(file=io.StringIO(), record=True)
    cli = RichCLI(agent=mock_agent, console=console)

    cli.process_turn("Preciso de uma exceção de devolução após 30 dias.")
    output = console.export_text()
    assert "TKT-20261102-TEST" in output
    assert (
        "HUMAN-IN-THE-LOOP" in output
        or "Escalação Humana" in output
        or "Atendimento Humano" in output
    )


def test_cli_welcome_banner():
    """Verify that the welcome banner renders store information."""
    from cli import RichCLI

    mock_agent = MagicMock()
    console = Console(file=io.StringIO(), record=True)
    cli = RichCLI(agent=mock_agent, console=console)
    cli.display_welcome_banner()
    output = console.export_text()
    assert "Empório da Música" in output
    assert "Campo Grande/MS" in output
    assert "/clear" in output


# ============================================================================
# Streamlit Web UI Tests
# ============================================================================


def test_streamlit_helpers():
    """Verify formatting helper utilities used in app.py."""
    import app

    assert "HIT" in app.format_cache_badge("HIT")
    assert "MISS" in app.format_cache_badge("MISS")
    assert "BYPASS" in app.format_cache_badge("BYPASS_TRANSACTIONAL")
    assert "$0.000025" in app.format_usd_cost(0.000025)
    assert app.format_usd_cost(0.0) == "$0.000000"


def test_streamlit_process_user_turn():
    """Verify process_user_turn delegates properly to the agent instance."""
    import app

    mock_agent = MagicMock()
    mock_agent.run.return_value = create_dummy_turn_result(
        output_text="Violão Giannini por R$ 650,00",
        session_id="session-st-test",
    )

    res = app.process_user_turn(mock_agent, "Violões baratos", "session-st-test")
    assert res.output_text == "Violão Giannini por R$ 650,00"
    mock_agent.run.assert_called_once_with(query="Violões baratos", session_id="session-st-test")


@patch("app.st")
def test_streamlit_render_sidebar_metrics_and_telemetry(mock_st):
    """Verify render_sidebar displays metrics, telemetry badges, and tool counts."""
    import app

    mock_st.columns.side_effect = lambda n: [
        MagicMock() for _ in range(n if isinstance(n, int) else len(n))
    ]
    settings = Settings(llm_provider="gemini")
    mock_agent = MagicMock()
    mock_agent.telemetry_manager.get_session_metrics.return_value = {
        "total_turns": 2,
        "total_latency_ms": 150.0,
        "average_latency_ms": 75.0,
        "total_input_tokens": 300,
        "total_output_tokens": 100,
        "total_tokens": 400,
        "total_cost_usd": 0.00005,
        "cache_hits": 1,
        "cache_misses": 1,
        "cache_bypasses": 0,
        "cache_hit_rate": 0.5,
        "hitl_escalations": 0,
        "tools_invoked_count": 1,
        "tools_distribution": {"search_products": 1},
    }

    dummy_tel = TurnTelemetry(
        session_id="test-session",
        latency_ms=45.2,
        cache_status="HIT",
        tools_called=["search_products"],
        total_tokens=150,
        input_tokens=100,
        output_tokens=50,
        estimated_cost_usd=0.00002,
    )

    app.render_sidebar(
        settings=settings,
        agent=mock_agent,
        session_id="test-session",
        last_telemetry=dummy_tel,
        hitl_tickets=[],
    )

    # Verify calls to Streamlit components
    assert mock_st.title.called
    assert mock_st.metric.called
    assert mock_st.badge.called


@patch("app.st")
def test_streamlit_render_sidebar_hitl_escalations(mock_st):
    """Verify render_sidebar highlights active HITL tickets."""
    import app

    mock_st.columns.side_effect = lambda n: [
        MagicMock() for _ in range(n if isinstance(n, int) else len(n))
    ]
    settings = Settings(llm_provider="gemini")
    mock_agent = MagicMock()
    mock_agent.telemetry_manager.get_session_metrics.return_value = {
        "total_turns": 1,
        "total_latency_ms": 100.0,
        "average_latency_ms": 100.0,
        "total_input_tokens": 200,
        "total_output_tokens": 50,
        "total_tokens": 250,
        "total_cost_usd": 0.00003,
        "cache_hits": 0,
        "cache_misses": 1,
        "cache_bypasses": 0,
        "cache_hit_rate": 0.0,
        "hitl_escalations": 1,
        "tools_invoked_count": 1,
        "tools_distribution": {"escalate_to_human": 1},
    }

    tickets = [
        {
            "ticket_id": "TKT-20261102-STREAMLIT",
            "reason": "Exceção de devolução",
            "urgency_level": "high",
            "contact_sla_hours": 24,
        }
    ]

    app.render_sidebar(
        settings=settings,
        agent=mock_agent,
        session_id="test-session",
        last_telemetry=None,
        hitl_tickets=tickets,
    )

    # Verify that error notification for active tickets was triggered
    assert mock_st.error.called
    call_args_str = str(mock_st.error.call_args)
    assert "Escalação(ões) Ativa(s)" in call_args_str or "Protocolo" in call_args_str


@patch("app.st")
def test_streamlit_render_chat_history(mock_st):
    """Verify render_chat_history renders normal messages and warning tickets."""
    import app

    messages = [
        {"role": "user", "content": "Olá", "ticket": None},
        {
            "role": "assistant",
            "content": "Encaminhei para nossa equipe.",
            "ticket": {
                "ticket_id": "TKT-12345",
                "reason": "Defeito fora de garantia",
                "urgency_level": "medium",
                "contact_sla_hours": 24,
            },
        },
    ]

    app.render_chat_history(messages)
    assert mock_st.chat_message.called
    assert mock_st.markdown.called
    assert mock_st.warning.called
    warning_args_str = str(mock_st.warning.call_args)
    assert "TKT-12345" in warning_args_str


@patch("app.render_chat_history")
@patch("app.render_sidebar")
@patch("app.get_agent")
@patch("app.st")
def test_streamlit_main_entrypoint_initialization(
    mock_st, mock_get_agent, mock_render_sidebar, mock_render_chat
):
    """Verify that main() sets page config, initializes session state, and invokes renderers."""
    import app

    mock_agent = MagicMock()
    mock_get_agent.return_value = mock_agent
    mock_st.session_state = {}
    mock_st.button.return_value = False
    mock_st.columns.side_effect = lambda n: [
        MagicMock() for _ in range(n if isinstance(n, int) else len(n))
    ]
    mock_st.chat_input.return_value = None

    app.main()

    assert mock_st.set_page_config.called
    assert "session_id" in mock_st.session_state
    assert "messages" in mock_st.session_state
    assert len(mock_st.session_state["messages"]) == 1
    assert mock_render_sidebar.called
    assert mock_render_chat.called


@patch("app.render_chat_history")
@patch("app.render_sidebar")
@patch("app.process_user_turn")
@patch("app.get_agent")
@patch("app.st")
def test_streamlit_main_entrypoint_with_user_input(
    mock_st, mock_get_agent, mock_process_turn, mock_render_sidebar, mock_render_chat
):
    """Verify that main() processes submitted chat input, appends messages, and calls rerun."""
    import app

    mock_agent = MagicMock()
    mock_get_agent.return_value = mock_agent
    mock_st.session_state = {
        "session_id": "test-session-123",
        "messages": [],
        "last_telemetry": None,
        "hitl_tickets": [],
    }
    mock_st.button.return_value = False
    mock_st.columns.side_effect = lambda n: [
        MagicMock() for _ in range(n if isinstance(n, int) else len(n))
    ]
    mock_st.chat_input.return_value = "Qual o endereço da loja física?"

    mock_process_turn.return_value = create_dummy_turn_result(
        output_text="Fica na Rua 14 de Maio, 3200",
        cache_status="HIT",
    )

    app.main()

    assert mock_process_turn.called
    assert len(mock_st.session_state["messages"]) == 2
    assert mock_st.session_state["last_telemetry"] is not None
    assert mock_st.rerun.called
