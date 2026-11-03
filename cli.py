"""Interactive Rich CLI for Empório da Música intelligent assistant."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from src.agent.graph import AgentTurnResult, EmporioAgent
from src.observability.telemetry import TurnTelemetry

logger = logging.getLogger(__name__)


class RichCLI:
    """Rich-powered terminal interface for Empório da Música customer support."""

    def __init__(
        self,
        agent: EmporioAgent | None = None,
        session_id: str | None = None,
        console: Console | None = None,
    ) -> None:
        self.console = console or Console()
        self.agent = agent or EmporioAgent()
        self.session_id = session_id or f"cli-{uuid.uuid4().hex[:8]}"

    def display_welcome_banner(self) -> None:
        """Render store branding, physical location, and quick instructions."""
        title = Text("🎸 Empório da Música — Atendimento Especialista", style="bold yellow")
        subtitle = (
            "[bold white]Loja Física:[/] Rua 14 de Maio, 3200 — Centro, Campo Grande/MS\n"
            "[bold white]Horário:[/] Seg–Sex 09:00–18:00 | Sáb 09:00–13:00\n"
            "[bold white]WhatsApp Suporte:[/] (67) 3321-4500\n\n"
            "[italic dim]Comandos especiais: /clear (reiniciar sessão), "
            "/metrics (observabilidade e FinOps), /exit (sair)[/]"
        )
        banner_panel = Panel(
            subtitle,
            title=title,
            border_style="cyan",
            padding=(1, 2),
        )
        self.console.print(banner_panel)

    def handle_command(self, user_input: str) -> bool:
        """Process special operator slash commands.

        Returns:
            bool: True if interactive loop should continue, False to exit.
        """
        cmd = user_input.strip().lower()

        if cmd in {"/exit", "/quit"}:
            farewell = Text(
                "Atendimento encerrado. Obrigado pela visita à Empório da Música! Até logo! 🎶",
                style="bold green",
            )
            self.console.print(farewell)
            return False

        if cmd == "/clear":
            old_session = self.session_id
            self.session_id = f"cli-{uuid.uuid4().hex[:8]}"
            logger.info(
                "Cleared CLI session '%s' -> new session '%s'",
                old_session,
                self.session_id,
            )
            self.console.print(
                Panel(
                    "Sessão reiniciada com sucesso!\n"
                    f"Novo identificador de sessão: [bold cyan]{self.session_id}[/]",
                    title="🧹 Sessão Limpa",
                    border_style="green",
                )
            )
            return True

        if cmd == "/metrics":
            self.display_session_metrics()
            return True

        if cmd == "/help":
            help_table = Table(title="Comandos Disponíveis", border_style="dim")
            help_table.add_column("Comando", style="bold cyan")
            help_table.add_column("Descrição", style="white")
            help_table.add_row("/clear", "Limpa o histórico de memória e inicia uma nova sessão.")
            help_table.add_row(
                "/metrics", "Exibe o painel de telemetria FinOps e métricas da sessão atual."
            )
            help_table.add_row("/help", "Exibe esta lista de instruções.")
            help_table.add_row("/exit, /quit", "Encerra o assistente.")
            self.console.print(help_table)
            return True

        return True

    def display_session_metrics(self) -> None:
        """Display session FinOps and performance aggregation table."""
        metrics = self.agent.telemetry_manager.get_session_metrics(self.session_id)

        table = Table(
            title=f"📊 Painel de Métricas da Sessão & FinOps [{self.session_id}]",
            border_style="blue",
        )
        table.add_column("Métrica", style="bold white")
        table.add_column("Valor", style="yellow")

        table.add_row("Total de Turnos", str(metrics["total_turns"]))
        table.add_row("Latência Média", f"{metrics['average_latency_ms']:.2f} ms")
        table.add_row("Tokens de Entrada (Prompt)", f"{metrics['total_input_tokens']:,}")
        table.add_row("Tokens de Saída (Completion)", f"{metrics['total_output_tokens']:,}")
        table.add_row("Total de Tokens", f"{metrics['total_tokens']:,}")
        table.add_row("Custo Estimado (USD)", f"${metrics['total_cost_usd']:.6f}")
        table.add_row("Taxa de Cache Hit", f"{metrics['cache_hit_rate'] * 100:.1f}%")
        table.add_row(
            "Distribuição do Cache",
            (
                f"Hits: {metrics['cache_hits']} | "
                f"Misses: {metrics['cache_misses']} | "
                f"Bypasses: {metrics['cache_bypasses']}"
            ),
        )
        table.add_row("Escalações Humanas (HITL)", str(metrics["hitl_escalations"]))

        tools_dist = metrics.get("tools_distribution", {})
        if tools_dist:
            tools_str = ", ".join(f"{k}: {v}" for k, v in tools_dist.items())
        else:
            tools_str = "Nenhuma ferramenta executada"
        table.add_row("Ferramentas Executadas", tools_str)

        self.console.print(table)

    def display_turn_telemetry(self, telemetry: TurnTelemetry) -> None:
        """Render a compact metadata footer under each assistant response."""
        # Format cache status color
        if telemetry.cache_status == "HIT":
            cache_tag = "[bold green]HIT (Zero Token / <50ms)[/]"
        elif telemetry.cache_status == "BYPASS_TRANSACTIONAL":
            cache_tag = "[bold cyan]BYPASS_TRANSACTIONAL (Dados em tempo real / PII)[/]"
        else:
            cache_tag = "[bold yellow]MISS[/]"

        # Format tools called
        if telemetry.tools_called:
            tools_tag = f"[bold magenta]{', '.join(telemetry.tools_called)}[/]"
        else:
            tools_tag = "[dim]Nenhuma (Direto / Cache)[/]"

        tokens_info = (
            f"{telemetry.total_tokens} "
            f"(Entrada: {telemetry.input_tokens}, Saída: {telemetry.output_tokens})"
        )
        meta_text = (
            f"⏱ [bold white]Latência:[/] {telemetry.latency_ms:.1f}ms  |  "
            f"💾 [bold white]Cache:[/] {cache_tag}  |  "
            f"🛠 [bold white]Ferramentas:[/] {tools_tag}\n"
            f"🪙 [bold white]Tokens:[/] {tokens_info}  |  "
            f"💵 [bold white]Custo:[/] ${telemetry.estimated_cost_usd:.6f}"
        )
        if telemetry.fallback_used:
            meta_text += (
                f"\n⚠️ [bold yellow]Contingência (Fallback):[/] "
                f"Provedor {telemetry.provider} ({telemetry.model_name}) acionado com sucesso!"
            )
        panel = Panel(
            meta_text,
            title="[dim]Telemetria & Observabilidade[/dim]",
            border_style="dim",
            padding=(0, 1),
        )
        self.console.print(panel)

    def display_hitl_alert(self, ticket: dict[str, Any] | None) -> None:
        """Render prominent escalation ticket alert when human intervention is needed."""
        ticket_data = ticket or {}
        ticket_id = ticket_data.get("ticket_id", "TKT-PENDING")
        reason = ticket_data.get("reason", "Solicitação fora do padrão ou exceção")
        urgency = str(ticket_data.get("urgency_level", "medium")).upper()
        sla = ticket_data.get("contact_sla_hours", 24)

        alert_body = (
            f"[bold red]ATENÇÃO: Caso transferido para a equipe humana da loja física.[/]\n\n"
            f"[bold white]Protocolo:[/] [bold yellow]{ticket_id}[/]\n"
            f"[bold white]Motivo:[/] {reason}\n"
            f"[bold white]Nível de Urgência:[/] [bold red]{urgency}[/]\n"
            f"[bold white]Prazo de Retorno:[/] até {sla} horas úteis\n"
            f"[bold white]Canal de Apoio:[/] WhatsApp (67) 3321-4500"
        )
        alert_panel = Panel(
            alert_body,
            title="🚨 HUMAN-IN-THE-LOOP (Escalação Humana)",
            border_style="bold red",
            padding=(1, 2),
        )
        self.console.print(alert_panel)

    def process_turn(self, user_input: str) -> AgentTurnResult:
        """Execute a single query through the agent and display the response."""
        result = self.agent.run(query=user_input, session_id=self.session_id)

        # 1. Print assistant response
        self.console.print()
        response_panel = Panel(
            Markdown(result.output_text),
            title="[bold yellow]Empório da Música[/bold yellow]",
            border_style="green",
            padding=(1, 2),
        )
        self.console.print(response_panel)

        # 2. Print HITL alert if human handoff was triggered
        if result.requires_human_intervention:
            self.display_hitl_alert(result.escalation_ticket)

        # 3. Print live telemetry panel
        self.display_turn_telemetry(result.telemetry)
        self.console.print()

        return result

    def run(self) -> None:
        """Interactive loop running in the terminal."""
        self.display_welcome_banner()

        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]Você[/bold cyan]")
                if not user_input or not user_input.strip():
                    continue

                if user_input.strip().startswith("/"):
                    should_continue = self.handle_command(user_input)
                    if not should_continue:
                        break
                    continue

                self.process_turn(user_input.strip())

            except (KeyboardInterrupt, EOFError):
                self.console.print(
                    "\n[bold yellow]Sessão encerrada pelo usuário. Até breve![/bold yellow]"
                )
                break
            except Exception as exc:
                logger.exception("Unexpected error in CLI interaction: %s", exc)
                self.console.print(f"[bold red]Erro inesperado no atendimento:[/] {exc}")


def main() -> None:
    """Entry point for Rich CLI execution."""
    cli = RichCLI()
    cli.run()


if __name__ == "__main__":
    main()
