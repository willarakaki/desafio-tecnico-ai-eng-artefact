"""Integration and unit tests for LangGraph Orchestrator, HITL, Persona, and FinOps Telemetry."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from src.agent.graph import AgentTurnResult, EmporioAgent, extract_tokens
from src.agent.prompts import get_system_prompt
from src.cache.semantic_cache import SemanticCache, reset_semantic_cache
from src.config import Settings
from src.database.duckdb_manager import reset_duckdb_manager
from src.observability.telemetry import (
    TelemetryManager,
    TurnTelemetry,
    calculate_token_cost,
    reset_telemetry_manager,
)
from src.rag.hybrid_retriever import reset_hybrid_retriever
from src.security.guardrails import reset_input_guardrail
from src.security.pii_anonymizer import reset_pii_anonymizer
from src.security.rate_limiter import reset_rate_limiter
from src.tools.hitl_tools import (
    HandoffTicketRegistry,
    create_hitl_tools,
    escalate_to_human,
    normalize_urgency_level,
    reset_ticket_registry,
)


class DeterministicMockEmbeddings(Embeddings):
    """Deterministic in-memory embedding model for repeatable testing without API calls."""

    def __init__(self) -> None:
        self.clusters: dict[str, list[float]] = {
            "address": [1.0, 0.2, 0.0, 0.0, 0.0],
            "hours": [0.0, 1.0, 0.2, 0.0, 0.0],
            "payment": [0.0, 0.0, 1.0, 0.2, 0.0],
            "returns": [0.0, 0.0, 0.0, 1.0, 0.2],
            "unrelated": [0.1, 0.1, 0.1, 0.1, 0.1],
        }

    def _get_vector(self, text: str) -> list[float]:
        import hashlib

        t = text.lower()
        if "endereço" in t or "endereco" in t or "rua" in t or "campo grande" in t:
            base = list(self.clusters["address"])
        elif "horário" in t or "horario" in t or "funciona" in t or "aberto" in t:
            base = list(self.clusters["hours"])
        elif "pagamento" in t or "pix" in t or "cartão" in t or "parcela" in t:
            base = list(self.clusters["payment"])
        elif "troca" in t or "devolu" in t or "defeito" in t:
            base = list(self.clusters["returns"])
        else:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            base = [float(b) / 255.0 for b in h[:5]]

        vec = np.array(base, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._get_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._get_vector(text)


# ============================================================================
# Deterministic Mock Chat Model for Offline CI Testing
# ============================================================================


class DeterministicAgentChatModel(BaseChatModel):
    """Mock Chat Model that simulates tool calling and consultative synthesis offline."""

    turn_counter: int = Field(default=0)
    canned_responses: dict[str, Any] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "deterministic-agent-mock"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.turn_counter += 1

        # Check if the last message is a ToolMessage (meaning a tool just executed)
        last_msg = messages[-1]
        if isinstance(last_msg, ToolMessage):
            # Synthesizer step: return final consultative text
            tool_name = getattr(last_msg, "name", "")
            tool_content = str(last_msg.content)

            if tool_name == "search_products":
                content = (
                    "Olá! Verifiquei nosso catálogo e temos excelentes violões até R$ 1.000, "
                    "como o Giannini Start GF-1D por R$ 680,00 e o Rozini RX210 por R$ 950,00! "
                    "Ambos possuem ótima tocabilidade e timbre equilibrado para iniciantes."
                )
            elif tool_name == "get_product_details":
                content = (
                    "O violão Takamine GD20 Dreadnought está disponível por R$ 3.890,00! "
                    "Esclareço que a promoção de 12% em nosso histórico já expirou, "
                    "portanto o valor vigente é o de tabela. No PIX, você garante 5% de desconto."
                )
            elif tool_name == "get_order_status":
                content = (
                    "Consultei o sistema da Empório da Música: seu pedido está confirmado com "
                    f"código de rastreio BR123456789BR. Detalhes: {tool_content[:100]}..."
                )
            elif tool_name == "consult_store_policies":
                content = (
                    "Nossa loja física fica na Rua 14 de Maio, 3200 — Centro, Campo Grande - MS "
                    "(CEP 79202-333). Horário: segunda a sexta das 09:00 às 18:00 e sábados das "
                    "09:00 às 13:00. Aos domingos e feriados estamos fechados."
                )
            elif tool_name == "escalate_to_human":
                content = (
                    "Entendo a sua situação. Como a solicitação envolve uma exceção fora das "
                    "regras padrão, já transferi o atendimento para a gerência com prazo de "
                    "retorno em até 24 horas úteis."
                )
            else:
                content = f"Informações verificadas no sistema da loja: {tool_content[:100]}."

            generation = ChatGeneration(
                message=AIMessage(
                    content=content,
                    usage_metadata={"input_tokens": 120, "output_tokens": 40, "total_tokens": 160},
                )
            )
            return ChatResult(generations=[generation])

        # Otherwise, inspect latest HumanMessage to decide tool call or direct answer
        human_text = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                human_text = str(m.content).lower()
                break

        # Multi-source scenario: asks about returning an order received
        if "devolver" in human_text and "pedido" in human_text:
            generation = ChatGeneration(
                message=AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "get_order_status", "args": {"order_id": 1}, "id": "call_multi_1"}
                    ],
                    usage_metadata={"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
                )
            )
            return ChatResult(generations=[generation])

        # Exception scenario: asks for return after deadline or dispute -> escalate_to_human
        if (
            "45 dias" in human_text
            or "exceção" in human_text
            or "excecao" in human_text
            or "insatisfeito" in human_text
        ):
            generation = ChatGeneration(
                message=AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "escalate_to_human",
                            "args": {
                                "reason": "Exceção de devolução fora do prazo legal",
                                "case_summary": human_text,
                                "urgency_level": "high",
                            },
                            "id": "call_hitl_1",
                        }
                    ],
                    usage_metadata={"input_tokens": 70, "output_tokens": 30, "total_tokens": 100},
                )
            )
            return ChatResult(generations=[generation])

        # Catalog search: violões under 1000
        if "viol" in human_text and (
            "1000" in human_text or "1.000" in human_text or "barato" in human_text
        ):
            generation = ChatGeneration(
                message=AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_products",
                            "args": {
                                "category": "violões",
                                "max_price": 1000.0,
                                "in_stock_only": True,
                            },
                            "id": "call_cat_1",
                        }
                    ],
                    usage_metadata={"input_tokens": 60, "output_tokens": 25, "total_tokens": 85},
                )
            )
            return ChatResult(generations=[generation])

        # Specific product details: Takamine GD20
        if "takamine" in human_text or "gd20" in human_text:
            generation = ChatGeneration(
                message=AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_product_details",
                            "args": {"product_name_or_id": "Takamine GD20"},
                            "id": "call_prod_1",
                        }
                    ],
                    usage_metadata={"input_tokens": 65, "output_tokens": 25, "total_tokens": 90},
                )
            )
            return ChatResult(generations=[generation])

        # Order status lookup
        if "pedido" in human_text or "rastre" in human_text:
            generation = ChatGeneration(
                message=AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_order_status",
                            "args": {"order_id": 5},
                            "id": "call_order_1",
                        }
                    ],
                    usage_metadata={"input_tokens": 60, "output_tokens": 20, "total_tokens": 80},
                )
            )
            return ChatResult(generations=[generation])

        # Policies lookup: address, hours, payments
        if (
            "endereço" in human_text
            or "endereco" in human_text
            or "horário" in human_text
            or "horario" in human_text
            or "onde fica" in human_text
        ):
            generation = ChatGeneration(
                message=AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "consult_store_policies",
                            "args": {"question": "qual o endereco e horario da loja fisica?"},
                            "id": "call_pol_1",
                        }
                    ],
                    usage_metadata={"input_tokens": 70, "output_tokens": 25, "total_tokens": 95},
                )
            )
            return ChatResult(generations=[generation])

        # Default conversational reply
        generation = ChatGeneration(
            message=AIMessage(
                content=(
                    "Olá! Sou o especialista da Empório da Música. "
                    "Como posso ajudar com seu instrumento musical hoje?"
                ),
                usage_metadata={"input_tokens": 50, "output_tokens": 25, "total_tokens": 75},
            )
        )
        return ChatResult(generations=[generation])


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clean_environment() -> None:
    """Reset singletons before and after each test."""
    reset_ticket_registry()
    reset_telemetry_manager()
    reset_semantic_cache()
    reset_rate_limiter()
    reset_input_guardrail()
    reset_duckdb_manager()
    reset_hybrid_retriever()
    reset_pii_anonymizer()
    yield
    reset_ticket_registry()
    reset_telemetry_manager()
    reset_semantic_cache()
    reset_rate_limiter()
    reset_input_guardrail()
    reset_duckdb_manager()
    reset_hybrid_retriever()
    reset_pii_anonymizer()


@pytest.fixture
def test_agent() -> EmporioAgent:
    """Instantiate a fully wired EmporioAgent with deterministic mock model."""
    settings = Settings(
        rate_limit_max_requests=20,
        rate_limit_window_seconds=60,
        max_input_chars=1500,
        semantic_cache_threshold=0.92,
    )
    mock_model = DeterministicAgentChatModel()
    mock_embeddings = DeterministicMockEmbeddings()
    cache = SemanticCache(embeddings=mock_embeddings, settings=settings)
    return EmporioAgent(settings=settings, model=mock_model, semantic_cache=cache)


# ============================================================================
# 1. HITL Tooling Unit Tests
# ============================================================================


def test_hitl_urgency_normalization() -> None:
    """Verify Brazilian Portuguese and English urgency strings normalize correctly."""
    assert normalize_urgency_level("low") == "low"
    assert normalize_urgency_level("baixa") == "low"
    assert normalize_urgency_level("média") == "medium"
    assert normalize_urgency_level("alta") == "high"
    assert normalize_urgency_level("crítica") == "critical"
    assert normalize_urgency_level("desconhecido") == "medium"


def test_hitl_escalate_to_human_creates_ticket_and_sla() -> None:
    """Verify escalate_to_human generates a structured TKT- ticket with 24 business hours SLA."""
    registry = HandoffTicketRegistry()
    res = escalate_to_human(
        reason="Cliente solicita devolução 40 dias após recebimento por insatisfação",
        case_summary="Violão Yamaha F310, cliente alega que não se adaptou às cordas de aço",
        urgency_level="alta",
        registry=registry,
    )

    assert res.escalated is True
    assert res.ticket.ticket_id.startswith("TKT-")
    assert res.ticket.sla == "24 horas úteis"
    assert res.ticket.urgency_level == "high"
    assert "24 horas úteis" in res.message
    assert res.ticket.ticket_id in res.message

    stored = registry.get_by_id(res.ticket.ticket_id)
    assert stored is not None
    assert stored.reason == "Cliente solicita devolução 40 dias após recebimento por insatisfação"


def test_hitl_structured_tool_langchain_parity() -> None:
    """Verify create_hitl_tools produces a valid LangChain StructuredTool."""
    tools = create_hitl_tools()
    assert len(tools) == 1
    hitl_tool = tools[0]
    assert hitl_tool.name == "escalate_to_human"

    raw_output = hitl_tool.invoke(
        {
            "reason": "Pedido de desconto especial para compra de 10 violões para escola",
            "case_summary": "Cliente institucional solicita 15% de desconto à vista",
            "urgency_level": "medium",
        }
    )
    data = json.loads(raw_output)
    assert data["escalated"] is True
    assert "ticket" in data
    assert data["ticket"]["ticket_id"].startswith("TKT-")


# ============================================================================
# 2. Telemetry & FinOps Unit Tests
# ============================================================================


def test_token_cost_calculation_by_provider() -> None:
    """Verify USD inference cost formula for Gemini, OpenAI, and Ollama."""
    # Gemini 2.5 Flash: $0.075/1M in, $0.30/1M out
    cost_gemini = calculate_token_cost(
        "gemini-2.5-flash", input_tokens=1_000_000, output_tokens=1_000_000
    )
    assert pytest.approx(cost_gemini, 0.0001) == 0.375

    # OpenAI GPT-4o-mini: $0.15/1M in, $0.60/1M out
    cost_openai = calculate_token_cost(
        "gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000
    )
    assert pytest.approx(cost_openai, 0.0001) == 0.75

    # Local Ollama: $0.0
    cost_ollama = calculate_token_cost("qwen2.5:7b", input_tokens=500_000, output_tokens=500_000)
    assert cost_ollama == 0.0

    # Zero tokens returns 0.0
    assert calculate_token_cost("gemini-2.5-flash", 0, 0) == 0.0


def test_telemetry_manager_recording_and_aggregation() -> None:
    """Verify TelemetryManager aggregates turn latency, token sums, and tool distributions."""
    manager = TelemetryManager()

    t1 = TurnTelemetry(
        session_id="sess_1",
        latency_ms=120.0,
        cache_status="MISS",
        tools_called=["search_products"],
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        estimated_cost_usd=0.00005,
    )
    t2 = TurnTelemetry(
        session_id="sess_1",
        latency_ms=15.0,
        cache_status="HIT",
        tools_called=[],
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        estimated_cost_usd=0.0,
    )

    manager.record_turn(t1)
    manager.record_turn(t2)

    metrics = manager.get_session_metrics("sess_1")
    assert metrics["total_turns"] == 2
    assert metrics["total_latency_ms"] == 135.0
    assert metrics["average_latency_ms"] == 67.5
    assert metrics["total_tokens"] == 150
    assert metrics["cache_hits"] == 1
    assert metrics["cache_misses"] == 1
    assert metrics["cache_hit_rate"] == 0.5
    assert metrics["tools_distribution"] == {"search_products": 1}


def test_token_extraction_from_messages() -> None:
    """Verify token counts are extracted from LangChain usage_metadata and response_metadata."""
    msg_with_usage = AIMessage(
        content="Olá",
        usage_metadata={"input_tokens": 42, "output_tokens": 18, "total_tokens": 60},
    )
    inp, out, tot = extract_tokens(msg_with_usage)
    assert inp == 42
    assert out == 18
    assert tot == 60

    msg_fallback = AIMessage(content="Um dois três quatro cinco")
    inp_f, out_f, tot_f = extract_tokens(msg_fallback)
    assert inp_f == 5
    assert out_f == 5
    assert tot_f == 10


# ============================================================================
# 3. Persona & Prompt Tests
# ============================================================================


def test_system_prompt_contains_ground_truth_and_negative_constraints() -> None:
    """Verify SYSTEM_PROMPT contains mandatory store facts and strict negative constraints."""
    prompt = get_system_prompt()

    # Ground truth entities
    assert "Empório da Música" in prompt
    assert "Campo Grande" in prompt
    assert "14 de Maio, 3200" in prompt
    assert "79202-333" in prompt
    assert "(67) 3321-4500" in prompt
    assert "(67) 3341-4444" in prompt
    assert "09:00 às 18:00" in prompt

    # Commercial & payment rules
    assert "5% de desconto" in prompt
    assert "NÃO é cumulativo" in prompt
    assert "12x" in prompt
    assert "R$ 500,00" in prompt

    # Negative constraints
    assert "NÃO comercializa acessórios avulsos" in prompt
    assert "NUNCA invente" in prompt
    assert "escalate_to_human" in prompt


# ============================================================================
# 4. Orchestrator Integration Tests
# ============================================================================


def test_catalog_routing_violoes_under_1000(test_agent: EmporioAgent) -> None:
    """Verify agent routes to search_products for violões up to R$ 1.000 and returns specs."""
    result = test_agent.run(
        "Quais violões vocês têm disponíveis até R$ 1.000?", session_id="user_1"
    )

    assert isinstance(result, AgentTurnResult)
    assert "search_products" in result.tools_called
    assert "violões" in result.output_text.lower() or "giannini" in result.output_text.lower()
    assert result.cache_status == "MISS"
    assert result.requires_human_intervention is False
    assert result.guardrail_blocked is False
    assert result.telemetry.total_tokens > 0


def test_catalog_routing_takamine_gd20_expired_promo(test_agent: EmporioAgent) -> None:
    """Verify Takamine GD20 lookup transparently discloses expired promotion."""
    result = test_agent.run(
        "Qual o preço do violão Takamine GD20? Tem promoção?", session_id="user_2"
    )

    assert "get_product_details" in result.tools_called
    assert "3.890" in result.output_text or "3890" in result.output_text
    assert "expirou" in result.output_text.lower() or "encerr" in result.output_text.lower()
    assert result.requires_human_intervention is False


def test_order_routing_by_order_id(test_agent: EmporioAgent) -> None:
    """Verify order query routes to get_order_status and returns tracking details."""
    result = test_agent.run("Gostaria de saber o status do meu pedido #5", session_id="user_3")

    assert "get_order_status" in result.tools_called
    assert "BR123456789BR" in result.output_text
    assert result.cache_status == "BYPASS_TRANSACTIONAL"  # Transactional bypass strictly enforced


def test_policy_routing_campo_grande_address_and_hours(test_agent: EmporioAgent) -> None:
    """Verify institutional policy question routes to consult_store_policies."""
    result = test_agent.run(
        "Onde fica a loja física em Campo Grande e qual o horário?", session_id="user_4"
    )

    assert "consult_store_policies" in result.tools_called
    assert "14 de Maio, 3200" in result.output_text
    assert "09:00 às 18:00" in result.output_text


def test_hitl_escalation_on_policy_exception(test_agent: EmporioAgent) -> None:
    """Verify policy exception triggers escalate_to_human and sets handoff ticket in state."""
    result = test_agent.run(
        "Recebi meu instrumento há 45 dias mas não gostei do som. Quero uma exceção para devolver!",
        session_id="user_5",
    )

    assert "escalate_to_human" in result.tools_called
    assert result.requires_human_intervention is True
    assert result.escalation_ticket is not None
    assert result.escalation_ticket.get("ticket_id", "").startswith("TKT-")
    assert result.telemetry.requires_human_intervention is True
    assert result.telemetry.escalation_ticket_id == result.escalation_ticket.get("ticket_id")


def test_guardrail_blocking_and_persona_refusal(test_agent: EmporioAgent) -> None:
    """Verify input guardrail actively blocks SQL injection and out-of-scope topics."""
    # 1. SQL Injection attempt
    res_sqli = test_agent.run("Violão ' OR '1'='1 --", session_id="user_sec_1")
    assert res_sqli.guardrail_blocked is True
    assert len(res_sqli.tools_called) == 0
    assert res_sqli.telemetry.total_tokens == 0
    assert res_sqli.telemetry.estimated_cost_usd == 0.0

    # 2. Out-of-scope cooking recipe
    res_recipe = test_agent.run("Como fazer bolo de chocolate fofinho?", session_id="user_sec_2")
    assert res_recipe.guardrail_blocked is True
    assert len(res_recipe.tools_called) == 0
    assert "Empório da Música" in res_recipe.output_text


def test_semantic_cache_hit_and_bypass_flow(test_agent: EmporioAgent) -> None:
    """Verify miss caches response; subsequent identical query hits cache in <50ms."""
    query = "Qual o endereço da loja em Campo Grande?"

    # First turn: Miss and populates cache
    res1 = test_agent.run(query, session_id="user_cache_1")
    assert res1.cache_status == "MISS"
    assert "14 de Maio" in res1.output_text

    # Second turn: Cache HIT
    res2 = test_agent.run(query, session_id="user_cache_2")
    assert res2.cache_status == "HIT"
    assert res2.telemetry.total_tokens == 0
    assert res2.telemetry.estimated_cost_usd == 0.0
    assert "14 de Maio" in res2.output_text

    # Transactional order query strictly bypasses cache
    res_order = test_agent.run("Rastrear o pedido 5", session_id="user_cache_3")
    assert res_order.cache_status == "BYPASS_TRANSACTIONAL"


def test_multi_turn_conversation_stateful_memory(test_agent: EmporioAgent) -> None:
    """Verify conversational memory is maintained across multiple turns with the same session_id."""
    session = "multi_turn_session"

    # Turn 1
    res1 = test_agent.run("Olá, procuro um violão", session_id=session)
    assert len(res1.messages) >= 2

    # Turn 2: Follow-up question under the same session thread
    res2 = test_agent.run("Quais violões vocês têm disponíveis até R$ 1.000?", session_id=session)
    assert len(res2.messages) >= 4  # Message history accumulated
    assert "search_products" in res2.tools_called


def test_pii_anonymization_in_agent_output() -> None:
    """Verify that any accidental PII in LLM output is redacted before returning to customer."""
    mock_model = MagicMock(spec=BaseChatModel)
    mock_model.bind_tools.return_value = mock_model
    # Simulate LLM returning a string containing a Brazilian CPF
    mock_model.invoke.return_value = AIMessage(
        content="Seu cadastro foi localizado com sucesso para o CPF 123.456.789-01.",
        usage_metadata={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
    )

    settings = Settings()
    cache = SemanticCache(embeddings=DeterministicMockEmbeddings(), settings=settings)
    agent = EmporioAgent(settings=settings, model=mock_model, semantic_cache=cache)
    result = agent.run("Qual meu cadastro?", session_id="pii_test")

    # CPF must be masked by PIIAnonymizer in finalize_node
    assert "123.456.789-01" not in result.output_text
    assert "***.***.***-**" in result.output_text
