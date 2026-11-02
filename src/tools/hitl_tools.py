"""Human-in-the-Loop (HITL) escalation tool for policy exceptions and critical customer cases."""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

UrgencyLevel = Literal["low", "medium", "high", "critical"]

_URGENCY_NORMALIZATION: dict[str, UrgencyLevel] = {
    "low": "low",
    "baixa": "low",
    "baixo": "low",
    "medium": "medium",
    "media": "medium",
    "média": "medium",
    "medio": "medium",
    "médio": "medium",
    "high": "high",
    "alta": "high",
    "alto": "high",
    "critical": "critical",
    "critica": "critical",
    "crítica": "critical",
    "critico": "critical",
    "crítico": "critical",
}


# ============================================================================
# Pydantic v2 Schemas
# ============================================================================


class EscalateToHumanInput(BaseModel):
    """Input schema for escalating a conversation to human customer support."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(
        description=(
            "Motivo detalhado da transferência para atendimento humano (ex: 'Exceção de prazo "
            "de devolução após 7 dias CDC', 'Reclamação grave de atraso de entrega', 'Negociação "
            "de lote personalizado de instrumentos')."
        ),
    )
    case_summary: str = Field(
        description=(
            "Resumo conciso do caso do cliente, incluindo números de pedidos, produtos citados "
            "e a solicitação exata que foge às regras padrão da loja."
        ),
    )
    urgency_level: str = Field(
        default="medium",
        description=(
            "Nível de urgência: 'low' (baixa), 'medium' (média), 'high' (alta) ou 'critical'."
        ),
    )


class HandoffTicket(BaseModel):
    """Structured human handoff ticket recorded for the store management team."""

    ticket_id: str = Field(description="Identificador único do ticket (ex: 'TKT-20261102-A1B2').")
    created_at: str = Field(description="Data e hora de abertura do ticket em formato ISO 8601.")
    reason: str = Field(description="Motivo da escalação.")
    case_summary: str = Field(description="Resumo do contexto e solicitação do cliente.")
    urgency_level: UrgencyLevel = Field(description="Nível normalizado de urgência.")
    sla: str = Field(
        default="24 horas úteis",
        description="Prazo de resposta acordado com o cliente (Seção 7.3 do Manual de Políticas).",
    )
    status: str = Field(
        default="aberto",
        description="Status inicial do ticket no sistema de atendimento humano.",
    )
    customer_message: str = Field(
        description="Mensagem cordial e empática tranquilizando o cliente sobre a transferência.",
    )


class HITLEscalationResult(BaseModel):
    """Final result returned by the `escalate_to_human` tool."""

    escalated: bool = Field(
        default=True,
        description="Indica que o caso foi transferido com sucesso.",
    )
    ticket: HandoffTicket = Field(description="Dados do ticket gerado.")
    message: str = Field(description="Mensagem de retorno para o cliente.")


# ============================================================================
# In-Memory Ticket Registry (Audit Trail & UI Inspection)
# ============================================================================


class HandoffTicketRegistry:
    """Thread-safe in-memory store for generated human escalation tickets."""

    def __init__(self) -> None:
        self._tickets: list[HandoffTicket] = []

    def record(self, ticket: HandoffTicket) -> None:
        """Register a new handoff ticket."""
        self._tickets.append(ticket)
        logger.info(
            "Recorded human handoff ticket '%s' (urgency: %s)",
            ticket.ticket_id,
            ticket.urgency_level,
        )

    def get_by_id(self, ticket_id: str) -> HandoffTicket | None:
        """Retrieve a ticket by its identifier."""
        for ticket in self._tickets:
            if ticket.ticket_id == ticket_id:
                return ticket
        return None

    def list_all(self) -> list[HandoffTicket]:
        """Return all generated tickets in chronological order."""
        return list(self._tickets)

    def clear(self) -> None:
        """Clear all registered tickets."""
        self._tickets.clear()


_TICKET_REGISTRY_SINGLETON: HandoffTicketRegistry | None = None


def get_ticket_registry() -> HandoffTicketRegistry:
    """Get or instantiate the singleton ticket registry."""
    global _TICKET_REGISTRY_SINGLETON
    if _TICKET_REGISTRY_SINGLETON is None:
        _TICKET_REGISTRY_SINGLETON = HandoffTicketRegistry()
    return _TICKET_REGISTRY_SINGLETON


def reset_ticket_registry() -> None:
    """Reset the singleton ticket registry (for testing)."""
    global _TICKET_REGISTRY_SINGLETON
    _TICKET_REGISTRY_SINGLETON = None


# ============================================================================
# Tool Implementation
# ============================================================================


def normalize_urgency_level(level: str) -> UrgencyLevel:
    """Normalize input urgency strings into standard literals."""
    clean = level.strip().lower()
    return _URGENCY_NORMALIZATION.get(clean, "medium")


def escalate_to_human(
    reason: str,
    case_summary: str,
    urgency_level: str = "medium",
    registry: HandoffTicketRegistry | None = None,
) -> HITLEscalationResult:
    """Create a human handoff ticket for policy exceptions or critical requests.

    Args:
        reason: Justification for escalation.
        case_summary: Summary of customer case.
        urgency_level: Priority level ('low', 'medium', 'high', 'critical').
        registry: Optional injected ticket registry.

    Returns:
        Structured HITLEscalationResult with ticket details and customer message.
    """
    now = datetime.datetime.now(datetime.UTC)
    date_str = now.strftime("%Y%m%d")
    random_suffix = uuid.uuid4().hex[:6].upper()
    ticket_id = f"TKT-{date_str}-{random_suffix}"
    normalized_urgency = normalize_urgency_level(urgency_level)

    customer_message = (
        "Compreendo perfeitamente a sua situação! Para garantir o melhor encaminhamento, "
        "abri um chamado prioritário junto à gerência de atendimento da Empório da Música. "
        f"Seu protocolo de atendimento é **{ticket_id}**. Nossa gerência analisará os "
        "detalhes do seu caso e entrará em contato diretamente com você em até 24 horas úteis. "
        "Se preferir atendimento imediato, nosso WhatsApp institucional é (67) 3321-4500."
    )

    ticket = HandoffTicket(
        ticket_id=ticket_id,
        created_at=now.isoformat(),
        reason=reason.strip(),
        case_summary=case_summary.strip(),
        urgency_level=normalized_urgency,
        sla="24 horas úteis",
        status="aberto",
        customer_message=customer_message,
    )

    active_registry = registry if registry is not None else get_ticket_registry()
    active_registry.record(ticket)

    return HITLEscalationResult(
        escalated=True,
        ticket=ticket,
        message=customer_message,
    )


def create_hitl_tools(registry: HandoffTicketRegistry | None = None) -> list[StructuredTool]:
    """Create LangChain `StructuredTool` instances for human escalation.

    Args:
        registry: Optional injected ticket registry.

    Returns:
        List containing the `escalate_to_human` StructuredTool.
    """

    def _escalate_to_human_tool(
        reason: str,
        case_summary: str,
        urgency_level: str = "medium",
    ) -> str:
        res = escalate_to_human(
            reason=reason,
            case_summary=case_summary,
            urgency_level=urgency_level,
            registry=registry,
        )
        return res.model_dump_json(indent=2)

    return [
        StructuredTool.from_function(
            func=_escalate_to_human_tool,
            name="escalate_to_human",
            description=(
                "Transfere o atendimento para um especialista humano da Empório da Música. "
                "Deve ser utilizada SEMPRE que o cliente solicitar uma exceção às políticas "
                "oficiais da loja (ex: devoluções fora de prazo, pedidos de garantia negados) "
                "ou demonstrar insatisfação crítica persistente. SLA: 24 horas úteis."
            ),
            args_schema=EscalateToHumanInput,
        )
    ]
