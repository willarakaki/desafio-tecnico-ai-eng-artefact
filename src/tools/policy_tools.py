"""Store policies RAG consultation tool with strict Pydantic v2 schemas."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from src.rag.hybrid_retriever import DEFAULT_TOP_K, HybridRetriever, get_hybrid_retriever

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic v2 Input & Output Schemas
# ============================================================================


class ConsultStorePoliciesInput(BaseModel):
    """Input schema for consulting the Empório da Música store policies manual."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        description=(
            "Pergunta ou tema institucional sobre a Empório da Música (ex: endereço físico em "
            "Campo Grande/MS, horário de funcionamento, formas de pagamento, desconto PIX, "
            "parcelamento, trocas e devoluções, prazos de frete, garantia ou privacidade LGPD)."
        ),
    )


class PolicyChunkResult(BaseModel):
    """Structured policy chunk extracted from `políticas_da_loja.pdf` with citation metadata."""

    section_number: str = Field(description="Número da seção no manual (ex: '1.2', '2', '4.1').")
    section: str = Field(description="Título completo da seção normativa.")
    parent_section: str | None = Field(
        default=None,
        description="Seção principal hierárquica, quando aplicável.",
    )
    topic: str = Field(description="Tópico semântico da política.")
    page_start: int = Field(description="Página inicial (base 1) no manual em PDF.")
    page_end: int = Field(description="Página final (base 1) no manual em PDF.")
    pages: list[int] = Field(
        default_factory=list,
        description="Lista ordenada das páginas abrangidas pelo trecho.",
    )
    source: str = Field(
        default="políticas_da_loja.pdf",
        description="Arquivo fonte oficial da política.",
    )
    content: str = Field(description="Texto normativo extraído integralmente da seção.")


class StorePoliciesResult(BaseModel):
    """Structured output returned by `consult_store_policies`."""

    found: bool = Field(description="Indica se foram encontrados trechos relevantes no manual.")
    query: str = Field(description="Pergunta original consultada.")
    chunks: list[PolicyChunkResult] = Field(
        default_factory=list,
        description="Trechos normativos ranqueados pelo recuperador híbrido (BM25 + FAISS).",
    )
    summary: str = Field(description="Resumo das seções recuperadas ou mensagem de orientação.")

    @property
    def message(self) -> str:
        """Alias maintaining interface parity with catalog and order tool results."""
        return self.summary


# ============================================================================
# Tool Implementation
# ============================================================================


def _document_to_chunk_result(doc: Document) -> PolicyChunkResult:
    """Convert a LangChain Document into a validated `PolicyChunkResult`."""
    meta = doc.metadata or {}
    page_start = int(meta.get("page_start", 1))
    page_end = int(meta.get("page_end", page_start))
    raw_pages = meta.get("pages")
    pages = (
        [int(p) for p in raw_pages]
        if isinstance(raw_pages, list) and raw_pages
        else list(range(page_start, page_end + 1))
    )

    return PolicyChunkResult(
        section_number=str(meta.get("section_number", "")),
        section=str(meta.get("section_title", "Política Institucional")),
        parent_section=meta.get("parent_section"),
        topic=str(meta.get("topic", "politicas_gerais")),
        page_start=page_start,
        page_end=page_end,
        pages=pages,
        source=str(meta.get("source", "políticas_da_loja.pdf")),
        content=doc.page_content,
    )


def consult_store_policies(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    retriever: HybridRetriever | None = None,
) -> StorePoliciesResult:
    """Consult the official Empório da Música store policies manual via Hybrid RAG.

    Combines sparse lexical retrieval (`BM25Retriever`, weight 0.4) and dense semantic
    retrieval (`FAISS`, weight 0.6) over `data/políticas_da_loja.pdf`.

    Args:
        question: Natural-language question about store rules, hours, address, or warranty.
        top_k: Maximum number of policy chunks to return (default 4).
        retriever: Optional pre-configured `HybridRetriever` instance.

    Returns:
        `StorePoliciesResult` containing ranked policy chunks with section and page metadata.
    """
    clean_question = question.strip() if isinstance(question, str) else ""
    if not clean_question:
        return StorePoliciesResult(
            found=False,
            query=question or "",
            chunks=[],
            summary=(
                "Por favor, informe uma dúvida ou tema sobre as políticas da Empório da Música "
                "(como endereço em Campo Grande/MS, horário de funcionamento, formas de pagamento, "
                "trocas e devoluções, frete ou garantia)."
            ),
        )

    try:
        active_retriever = retriever if retriever is not None else get_hybrid_retriever()
        docs = active_retriever.retrieve(clean_question, top_k=top_k)

        if not docs:
            return StorePoliciesResult(
                found=False,
                query=clean_question,
                chunks=[],
                summary=(
                    "Não foram encontrados trechos específicos no Manual de Políticas para "
                    f"'{clean_question}'. Recomende atendimento humano caso trate de exceção."
                ),
            )

        chunk_results = [_document_to_chunk_result(d) for d in docs]
        cited_sections = ", ".join(
            f"{c.section} (p. {c.page_start}"
            + (f"-{c.page_end})" if c.page_end != c.page_start else ")")
            for c in chunk_results
        )
        return StorePoliciesResult(
            found=True,
            query=clean_question,
            chunks=chunk_results,
            summary=(
                f"Foram encontrados {len(chunk_results)} trecho(s) no Manual de Políticas e "
                f"Procedimentos da Empório da Música: {cited_sections}."
            ),
        )
    except Exception as exc:
        logger.error("Error consulting store policies for '%s': %s", clean_question, exc)
        return StorePoliciesResult(
            found=False,
            query=clean_question,
            chunks=[],
            summary=(
                "Ocorreu uma indisponibilidade temporária ao consultar o Manual de Políticas da "
                "Empório da Música. Para informações imediatas, nosso WhatsApp é (67) 3321-4500."
            ),
        )


def create_policy_tools(retriever: HybridRetriever | None = None) -> list[StructuredTool]:
    """Create LangChain `StructuredTool` instances for store policy consultation.

    Args:
        retriever: Optional injected `HybridRetriever` (lazily resolved on call if None).

    Returns:
        List containing the `consult_store_policies` StructuredTool.
    """

    def _consult_store_policies_tool(question: str) -> str:
        active = retriever if retriever is not None else get_hybrid_retriever()
        res = consult_store_policies(question=question, retriever=active)
        return res.model_dump_json(indent=2)

    return [
        StructuredTool.from_function(
            func=_consult_store_policies_tool,
            name="consult_store_policies",
            description=(
                "Consulta o Manual Oficial de Políticas e Procedimentos da Empório da Música "
                "(endereço físico em Campo Grande/MS, telefone/WhatsApp, horário de funcionamento, "
                "formas de pagamento, desconto de 5% no PIX, regras de parcelamento até 12x, "
                "trocas, devoluções em 7 ou 30 dias, frete/entregas, garantia e LGPD)."
            ),
            args_schema=ConsultStorePoliciesInput,
        )
    ]
