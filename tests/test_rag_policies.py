"""Unit tests for the Hybrid RAG pipeline over Empório da Música store policies PDF."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from src.config import Settings
from src.rag.hybrid_retriever import (
    DEFAULT_BM25_WEIGHT,
    DEFAULT_FAISS_WEIGHT,
    HybridRetriever,
    get_hybrid_retriever,
    preprocess_pt_bm25,
    reset_hybrid_retriever,
)
from src.rag.pdf_loader import PDFPolicyLoader, load_store_policy_documents
from src.tools.policy_tools import (
    ConsultStorePoliciesInput,
    StorePoliciesResult,
    consult_store_policies,
    create_policy_tools,
)

if TYPE_CHECKING:
    from pathlib import Path


# ============================================================================
# Deterministic Semantic Embeddings Fixture for Offline CI Testing
# ============================================================================

_CONCEPT_CLUSTERS: list[set[str]] = [
    # 0: Store address, location, CNPJ, institutional data
    {
        "endereco",
        "rua",
        "maio",
        "3200",
        "centro",
        "campo",
        "grande",
        "cep",
        "79202",
        "cnpj",
        "localizacao",
        "fica",
        "onde",
        "loja",
        "fisica",
        "visitar",
        "presencialmente",
        "razao",
        "social",
    },
    # 1: Opening hours & schedule
    {
        "horario",
        "funcionamento",
        "abre",
        "fecha",
        "aberto",
        "fechado",
        "expediente",
        "segunda",
        "sexta",
        "sabado",
        "domingo",
        "feriado",
        "09:00",
        "18:00",
        "13:00",
        "horas",
        "atende",
    },
    # 2: Payment methods, PIX discount, credit card installments
    {
        "pagamento",
        "pagar",
        "pix",
        "avista",
        "vista",
        "desconto",
        "5%",
        "boleto",
        "debito",
        "credito",
        "cartao",
        "parcela",
        "parcelamento",
        "parcelar",
        "12x",
        "juros",
        "combinacao",
    },
    # 3: Returns, right of withdrawal (CDC 7 days), refunds
    {
        "arrependimento",
        "devolucao",
        "devolver",
        "desistir",
        "desisti",
        "mudei",
        "ideia",
        "reembolso",
        "estorno",
        "dinheiro",
        "volta",
        "online",
        "internet",
        "sete",
        "corridos",
        "embalagem",
    },
    # 4: Exchanges for manufacturing defect (30 days) & preference
    {
        "defeito",
        "fabricacao",
        "trinta",
        "30",
        "intermediar",
        "preferencia",
    },
    # 5: Shipping, carriers (Correios PAC/SEDEX, Jadlog), tracking
    {
        "frete",
        "entrega",
        "entregas",
        "envio",
        "motoboy",
        "gratis",
        "500",
        "correios",
        "pac",
        "sedex",
        "jadlog",
        "rastreamento",
        "rastreio",
        "codigo",
        "seguro",
        "avaria",
    },
    # 6: Promotions & non-cumulative rules
    {
        "promocao",
        "promocoes",
        "cumulativa",
        "cumulativas",
        "cumulatividade",
        "black",
        "friday",
        "aniversario",
        "queima",
        "rain",
        "check",
    },
    # 7: WhatsApp service & special situations
    {
        "whatsapp",
        "3321",
        "4500",
        "reclamacao",
        "esgotado",
        "indisponivel",
        "descontinuado",
        "acessorios",
        "cordas",
        "palhetas",
        "pedais",
        "amplificadores",
    },
    # 8: Warranty (90 days legal + manufacturer)
    {
        "garantia",
        "noventa",
        "90",
        "certificado",
        "desgaste",
        "trastes",
        "queda",
        "esteticos",
    },
    # 9: LGPD privacy & data deletion
    {
        "privacidade",
        "lgpd",
        "13709",
        "pessoais",
        "exclusao",
        "terceiros",
        "marketing",
        "protecao",
    },
    # 10: Ineligible items for exchange (mouthpieces, custom setup, final sale)
    {
        "boquilha",
        "boquilhas",
        "higiene",
        "personalizacao",
        "regulagem",
        "setup",
        "elegiveis",
        "liquidacao",
    },
]


class DeterministicSemanticEmbeddings(Embeddings):
    """Deterministic local embeddings combining semantic concept clusters and hashed n-grams.

    Produces unit-normalized float vectors without external network calls or model weights,
    enabling genuine FAISS vector similarity testing in offline CI environments.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    @staticmethod
    def _normalize_text(text: str) -> str:
        nfd = unicodedata.normalize("NFD", text.lower())
        no_accents = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
        return re.sub(r"\s+", " ", no_accents).strip()

    def _encode(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        norm = self._normalize_text(text)
        tokens = re.findall(r"[a-z0-9]+%?|[0-9]+:[0-9]+", norm)

        # 1. Semantic concept dimensions (dims 0..len(_CONCEPT_CLUSTERS)-1)
        for cluster_idx, keywords in enumerate(_CONCEPT_CLUSTERS):
            hits = sum(1 for tok in tokens if tok in keywords or tok.rstrip("s") in keywords)
            if hits > 0:
                vec[cluster_idx] = 3.0 * math.log1p(hits)

        # 2. Hashed token + character 4-gram dimensions (dims 16..dim-1)
        hash_space = self.dim - 16
        for tok in tokens:
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest()
            bucket = 16 + (int.from_bytes(digest, "little") % hash_space)
            vec[bucket] += 1.0

        compact = norm.replace(" ", "")
        for i in range(max(0, len(compact) - 3)):
            gram = compact[i : i + 4]
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=4).digest()
            bucket = 16 + (int.from_bytes(digest, "little") % hash_space)
            vec[bucket] += 0.25

        norm_l2 = math.sqrt(sum(v * v for v in vec))
        if norm_l2 > 0:
            vec = [v / norm_l2 for v in vec]
        else:
            vec[0] = 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


@pytest.fixture(scope="module")
def policy_documents() -> list[Document]:
    """Load real store policy chunks once per test module."""
    return load_store_policy_documents()


@pytest.fixture(scope="module")
def deterministic_embeddings() -> DeterministicSemanticEmbeddings:
    """Provide deterministic semantic embeddings for offline FAISS testing."""
    return DeterministicSemanticEmbeddings()


@pytest.fixture(scope="module")
def hybrid_retriever(
    policy_documents: list[Document],
    deterministic_embeddings: DeterministicSemanticEmbeddings,
) -> HybridRetriever:
    """Build HybridRetriever (0.4 BM25 + 0.6 FAISS) over the real store policies PDF."""
    return HybridRetriever(
        documents=policy_documents,
        embeddings=deterministic_embeddings,
        bm25_weight=0.4,
        faiss_weight=0.6,
        top_k=4,
    )


# ============================================================================
# 1. PDF Loader & Topic-Aware Chunking Tests
# ============================================================================


def test_pdf_loader_page_extraction_and_cleaning() -> None:
    """Verify clean extraction of all 8 pages with running headers and footers stripped."""
    loader = PDFPolicyLoader()
    pages = loader.extract_pages()

    assert len(pages) == 8
    assert [p["page_number"] for p in pages] == [1, 2, 3, 4, 5, 6, 7, 8]

    for page in pages:
        text = page["text"]
        assert text.strip() != ""
        # Running header on pages 2..8 and "Página X" footer must be stripped
        assert "Empório da Música Manual de Políticas e Procedimentos" not in text
        assert not re.search(r"(?m)^Página\s+\d+\s*$", text)
        # Word-per-line wrapping artifacts must be joined into clean prose
        assert "Campo\nGrande,\nMato" not in text


def test_pdf_loader_topic_chunking_and_metadata(policy_documents: list[Document]) -> None:
    """Verify all major sections and topics are extracted with rich page/section metadata."""
    assert len(policy_documents) >= 12

    sections_found = {doc.metadata["section_number"] for doc in policy_documents}
    expected_sections = {
        "1.2",
        "2",
        "3",
        "3.1",
        "4.1",
        "4.2",
        "4.3",
        "4.4",
        "5.1",
        "5.2",
        "5.3",
        "6.1",
        "6.2",
        "7.1",
        "7.2",
        "7.3",
        "8.1",
        "8.2",
        "8.3",
        "9",
        "10",
    }
    assert expected_sections.issubset(sections_found)

    for doc in policy_documents:
        meta = doc.metadata
        assert meta["source"] == "políticas_da_loja.pdf"
        assert isinstance(meta["section_title"], str) and meta["section_title"]
        assert isinstance(meta["topic"], str) and meta["topic"]
        assert 1 <= meta["page_start"] <= meta["page_end"] <= 8
        assert isinstance(meta["pages"], list) and len(meta["pages"]) >= 1
        assert meta["pages"][0] == meta["page_start"]
        assert meta["pages"][-1] == meta["page_end"]


def test_pdf_loader_cross_page_continuity(policy_documents: list[Document]) -> None:
    """Validate that sections crossing page breaks (5.2, 7.2, 9) stay attached to their bodies."""
    by_sec = {doc.metadata["section_number"]: doc for doc in policy_documents}

    # Section 5.2 starts at bottom of Page 4 and has PAC/SEDEX/Jadlog table on Page 5
    sec_5_2 = by_sec["5.2"]
    assert sec_5_2.metadata["page_start"] == 4
    assert sec_5_2.metadata["page_end"] == 5
    assert sec_5_2.metadata["pages"] == [4, 5]
    assert "Entregas para Outras Cidades" in sec_5_2.page_content
    assert "SEDEX" in sec_5_2.page_content
    assert "Jadlog" in sec_5_2.page_content

    # Section 7.2 starts on Page 6 (steps 1-2) and finishes on Page 7 (steps 3-5)
    sec_7_2 = by_sec["7.2"]
    assert sec_7_2.metadata["page_start"] == 6
    assert sec_7_2.metadata["page_end"] == 7
    assert "Saudação" in sec_7_2.page_content
    assert "Consulta:" in sec_7_2.page_content
    assert "Fechamento:" in sec_7_2.page_content

    # Section 9 starts on Page 7 (LGPD header) and finishes bullet rules on Page 8
    sec_9 = by_sec["9"]
    assert sec_9.metadata["page_start"] == 7
    assert sec_9.metadata["page_end"] == 8
    assert "13.709/2018" in sec_9.page_content
    assert "terceiros para fins de marketing" in sec_9.page_content


def test_pdf_loader_missing_file_raises(tmp_path: Path) -> None:
    """Ensure PDFPolicyLoader raises FileNotFoundError on non-existent PDF path."""
    missing = tmp_path / "nao_existe.pdf"
    with pytest.raises(FileNotFoundError):
        PDFPolicyLoader(pdf_path=missing)


# ============================================================================
# 2. Hybrid Retriever (BM25 0.4 + FAISS 0.6) Tests
# ============================================================================


def test_hybrid_retriever_weights_and_validation(
    policy_documents: list[Document],
    deterministic_embeddings: DeterministicSemanticEmbeddings,
) -> None:
    """Validate default weights (0.4 BM25, 0.6 FAISS), bounds, and LLMFactory wiring."""
    assert DEFAULT_BM25_WEIGHT == 0.4
    assert DEFAULT_FAISS_WEIGHT == 0.6

    retriever = HybridRetriever(
        documents=policy_documents,
        embeddings=deterministic_embeddings,
    )
    assert retriever.bm25_weight == pytest.approx(0.4)
    assert retriever.faiss_weight == pytest.approx(0.6)
    assert not retriever.is_degraded_lexical_only

    with pytest.raises(ValueError, match="weights"):
        HybridRetriever(
            documents=policy_documents,
            embeddings=deterministic_embeddings,
            bm25_weight=0.0,
            faiss_weight=0.0,
        )

    with pytest.raises(ValueError, match="top_k"):
        HybridRetriever(
            documents=policy_documents,
            embeddings=deterministic_embeddings,
            top_k=0,
        )

    # Confirm HybridRetriever delegates to LLMFactory.get_embeddings when embeddings=None
    dummy_settings = Settings(
        llm_provider="gemini",
        gemini_api_key="dummy-key",
        _env_file=None,
    )
    with patch(
        "src.rag.hybrid_retriever.LLMFactory.get_embeddings",
        return_value=deterministic_embeddings,
    ) as mock_get_emb:
        via_factory = HybridRetriever(documents=policy_documents, settings=dummy_settings)
        mock_get_emb.assert_called_once_with(settings=dummy_settings)
        docs = via_factory.retrieve("horário de funcionamento", top_k=2)
        assert len(docs) == 2


def test_preprocess_pt_bm25_normalization() -> None:
    """Ensure BM25 tokenizer strips accents/punctuation and preserves numbers and percentages."""
    tokens = preprocess_pt_bm25("Qual o horário de Sábado? PIX tem 5% em Campo Grande!")
    assert "horario" in tokens
    assert "sabado" in tokens
    assert "5%" in tokens
    assert "campo" in tokens
    assert "grande" in tokens


# ============================================================================
# 3. Retrieval Accuracy Tests (Address, Hours, Returns/Refunds, Payment, Scope)
# ============================================================================


def test_consult_store_policies_campo_grande_address(hybrid_retriever: HybridRetriever) -> None:
    """Validate retrieval accuracy for the Campo Grande/MS store physical address and contacts."""
    res = consult_store_policies(
        "Qual o endereço da loja física em Campo Grande/MS e o telefone?",
        retriever=hybrid_retriever,
    )
    assert isinstance(res, StorePoliciesResult)
    assert res.found is True
    assert len(res.chunks) >= 1

    combined = "\n".join(c.content for c in res.chunks)
    assert "Rua 14 de Maio, 3200" in combined
    assert "Campo Grande" in combined
    assert "79202-333" in combined
    assert "3341-4444" in combined

    top_sections = {c.section_number for c in res.chunks[:2]}
    assert "1.2" in top_sections or "1" in top_sections


def test_consult_store_policies_opening_hours(hybrid_retriever: HybridRetriever) -> None:
    """Validate retrieval accuracy for weekday, Saturday, and Sunday/holiday opening hours."""
    res = consult_store_policies(
        "Que horas a loja abre de segunda a sexta e aos sábados?",
        retriever=hybrid_retriever,
    )
    assert res.found is True
    top_chunk = res.chunks[0]
    assert top_chunk.section_number == "2"
    assert top_chunk.page_start == 2
    assert "09:00 às 18:00" in top_chunk.content
    assert "09:00 às 13:00" in top_chunk.content
    assert "Fechado" in top_chunk.content


def test_consult_store_policies_return_and_refund_rules(hybrid_retriever: HybridRetriever) -> None:
    """Validate retrieval for online withdrawal (7d / 10d refund) and defect exchange (30d)."""
    # Paraphrased semantic query for online withdrawal & refund
    res_withdrawal = consult_store_policies(
        "Comprei pela internet e mudei de ideia, quero devolver e receber estorno do dinheiro",
        retriever=hybrid_retriever,
    )
    assert res_withdrawal.found is True
    combined_withdrawal = "\n".join(c.content for c in res_withdrawal.chunks[:2])
    assert "7 (sete) dias corridos" in combined_withdrawal
    assert "10 dias úteis" in combined_withdrawal
    assert "embalagem original" in combined_withdrawal
    assert any(c.page_start == 4 for c in res_withdrawal.chunks[:2])

    # Manufacturing defect exchange query
    res_defect = consult_store_policies(
        "Meu instrumento veio com defeito de fabricação, qual o prazo de troca na loja?",
        retriever=hybrid_retriever,
    )
    assert res_defect.found is True
    combined_defect = "\n".join(c.content for c in res_defect.chunks[:2])
    assert "30 (trinta) dias corridos" in combined_defect
    assert "fabricante" in combined_defect

    # Ineligible items query (mouthpieces, custom setup, final sale)
    res_ineligible = consult_store_policies(
        "Posso trocar boquilhas de sopro ou instrumento com regulagem setup sob encomenda?",
        retriever=hybrid_retriever,
    )
    assert res_ineligible.found is True
    combined_ineligible = "\n".join(c.content for c in res_ineligible.chunks[:2])
    assert "Boquilhas" in combined_ineligible
    assert "venda final" in combined_ineligible


def test_consult_store_policies_shipping_and_accessories_scope(
    hybrid_retriever: HybridRetriever,
) -> None:
    """Validate retrieval for Campo Grande free shipping (> R$ 500) and non-accessory scope."""
    res_ship = consult_store_policies(
        "Qual o valor do frete para Campo Grande e prazo via SEDEX ou Jadlog?",
        retriever=hybrid_retriever,
    )
    assert res_ship.found is True
    combined_ship = "\n".join(c.content for c in res_ship.chunks)
    assert "R$ 500,00" in combined_ship
    assert "R$ 35,00" in combined_ship
    assert "SEDEX" in combined_ship

    res_scope = consult_store_policies(
        "A loja vende acessórios como cordas, palhetas, cabos, pedais ou amplificadores?",
        retriever=hybrid_retriever,
    )
    assert res_scope.found is True
    combined_scope = "\n".join(c.content for c in res_scope.chunks[:2])
    assert "exclusivamente com instrumentos musicais" in combined_scope


def test_consult_store_policies_payment_and_non_cumulative_promo(
    hybrid_retriever: HybridRetriever,
) -> None:
    """Validate retrieval for 5% PIX discount, 12x installments, and non-cumulative promo rules."""
    res_pay = consult_store_policies(
        "Tem desconto pagando no PIX à vista? Como funciona o parcelamento sem juros no cartão?",
        retriever=hybrid_retriever,
    )
    assert res_pay.found is True
    combined_pay = "\n".join(c.content for c in res_pay.chunks)
    assert "5% de desconto" in combined_pay
    assert "12x sem juros" in combined_pay
    assert "R$ 80,00" in combined_pay or "R$ 100,00" in combined_pay

    res_cumul = consult_store_policies(
        "O desconto de 5% do PIX acumula com produtos em promoção?",
        retriever=hybrid_retriever,
    )
    assert res_cumul.found is True
    combined_cumul = "\n".join(c.content for c in res_cumul.chunks[:2])
    assert "não são cumulativas" in combined_cumul


def test_consult_store_policies_empty_query_and_bm25_resilience(
    policy_documents: list[Document],
) -> None:
    """Verify graceful empty query handling and resilient BM25 fallback without API key."""
    # Empty query must never crash or call embedding APIs
    empty_res = consult_store_policies("   ")
    assert empty_res.found is False
    assert empty_res.chunks == []
    assert empty_res.message != ""

    # Resilient BM25 fallback when LLMFactory raises ValueError due to missing API key
    no_key_settings = Settings(llm_provider="gemini", gemini_api_key=None, _env_file=None)
    fallback_retriever = HybridRetriever(
        documents=policy_documents,
        settings=no_key_settings,
        allow_bm25_fallback=True,
    )
    assert fallback_retriever.is_degraded_lexical_only is True
    res = consult_store_policies("Rua 14 de Maio Campo Grande", retriever=fallback_retriever)
    assert res.found is True
    assert any("Rua 14 de Maio" in c.content for c in res.chunks)


def test_structured_policy_tool_langchain_parity(hybrid_retriever: HybridRetriever) -> None:
    """Verify LangChain StructuredTool schema validation, invocation, and JSON serialization."""
    reset_hybrid_retriever()
    tools = create_policy_tools(retriever=hybrid_retriever)
    assert len(tools) == 1

    tool = tools[0]
    assert tool.name == "consult_store_policies"
    assert tool.args_schema is ConsultStorePoliciesInput

    raw_json = tool.invoke({"question": "Qual o prazo da garantia legal e do fabricante?"})
    parsed = json.loads(raw_json)
    assert parsed["found"] is True
    assert len(parsed["chunks"]) >= 1
    combined = "\n".join(c["content"] for c in parsed["chunks"])
    assert "90 (noventa) dias" in combined
    assert "6 meses a 2 anos" in combined

    # Singleton helper smoke test with injected embeddings
    singleton = get_hybrid_retriever(embeddings=DeterministicSemanticEmbeddings())
    assert singleton is get_hybrid_retriever()
    reset_hybrid_retriever()
