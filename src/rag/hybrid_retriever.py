"""Hybrid RAG Retriever combining sparse BM25 (0.4) and dense FAISS (0.6) search."""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from typing import TYPE_CHECKING

import faiss as _faiss  # noqa: F401 — Eager load avoids lazy Win32 DLL probe issues
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.agent.llm_factory import LLMFactory
from src.rag.pdf_loader import PDFPolicyLoader

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.embeddings import Embeddings

    from src.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_BM25_WEIGHT: float = 0.4
DEFAULT_FAISS_WEIGHT: float = 0.6
DEFAULT_TOP_K: int = 4

_PT_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "ao",
        "aos",
        "as",
        "com",
        "como",
        "da",
        "das",
        "de",
        "do",
        "dos",
        "e",
        "em",
        "estar",
        "estao",
        "eu",
        "foi",
        "ha",
        "isso",
        "isto",
        "la",
        "lhe",
        "mais",
        "mas",
        "me",
        "meu",
        "minha",
        "na",
        "nas",
        "nem",
        "no",
        "nos",
        "num",
        "numa",
        "o",
        "onde",
        "os",
        "ou",
        "para",
        "pela",
        "pelas",
        "pelo",
        "pelos",
        "por",
        "qual",
        "quais",
        "quando",
        "que",
        "quem",
        "se",
        "sem",
        "ser",
        "seu",
        "sua",
        "suas",
        "seus",
        "so",
        "sobre",
        "tem",
        "ter",
        "um",
        "uma",
        "umas",
        "uns",
        "voce",
    }
)

_TOKEN_PATTERN = re.compile(r"\d{2}:\d{2}|\d+(?:[.,/-]\d+)+%?|[a-z0-9]+%?")


def preprocess_pt_bm25(text: str) -> list[str]:
    """Tokenize Portuguese text for BM25 with accent folding and entity preservation.

    Preserves exact numeric entities (percentages like `5%`, hours like `09:00`,
    CEPs like `79202-333`, CNPJs, and installment counts like `12x`) while stripping
    diacritics, removing grammatical stopwords, and adding lightweight singular stems.
    """
    if not text or not text.strip():
        return [""]

    nfd = unicodedata.normalize("NFD", text.lower())
    folded = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    raw_tokens = _TOKEN_PATTERN.findall(folded)
    if not raw_tokens:
        return [folded.strip()]

    filtered = [tok for tok in raw_tokens if tok not in _PT_STOPWORDS]
    base_tokens = filtered if filtered else raw_tokens

    expanded: list[str] = []
    for tok in base_tokens:
        expanded.append(tok)
        # Split compound hyphen/slash numeric identifiers so partial lookups also match
        if "-" in tok or "/" in tok:
            for part in re.split(r"[-/]", tok):
                if len(part) >= 2:
                    expanded.append(part)
        # Lightweight Portuguese plural normalization for lexical recall
        elif tok.isalpha() and len(tok) >= 5:
            if tok.endswith("coes"):
                expanded.append(tok[:-4] + "cao")
            elif tok.endswith("oes"):
                expanded.append(tok[:-3] + "ao")
            elif tok.endswith("ais"):
                expanded.append(tok[:-3] + "al")
            elif tok.endswith("eis"):
                expanded.append(tok[:-3] + "el")
            elif tok.endswith("s") and not tok.endswith("ss"):
                expanded.append(tok[:-1])

    return expanded


class _PositiveScoreBM25Retriever(BM25Retriever):
    """BM25Retriever subclass that excludes zero-score non-matching documents.

    Standard `rank_bm25.get_top_n` pads results with zero-score documents from the
    tail of the corpus when fewer than `k` chunks match lexically, which would
    otherwise inject false-positive rank credit into `EnsembleRetriever` RRF fusion.
    """

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: object = None,
    ) -> list[Document]:
        if self.vectorizer is None:
            return []
        processed_query = self.preprocess_func(query)
        scores = self.vectorizer.get_scores(processed_query)
        ranked_indices = sorted(
            (idx for idx, score in enumerate(scores) if score > 0.0),
            key=lambda idx: float(scores[idx]),
            reverse=True,
        )
        return [self.docs[idx] for idx in ranked_indices[: self.k]]


class HybridRetriever:
    """Weighted Hybrid Retriever combining BM25 lexical search and FAISS vector search.

    Default ensemble weights follow the system architecture specification:
    - `bm25_weight = 0.4` for exact lexical entities (deadlines, percentages, addresses)
    - `faiss_weight = 0.6` for paraphrased conceptual customer questions
    """

    def __init__(
        self,
        documents: list[Document] | None = None,
        embeddings: Embeddings | None = None,
        pdf_path: Path | str | None = None,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        faiss_weight: float = DEFAULT_FAISS_WEIGHT,
        top_k: int = DEFAULT_TOP_K,
        settings: Settings | None = None,
        allow_bm25_fallback: bool = False,
    ) -> None:
        """Initialize BM25 and FAISS indices over the store policy documents.

        Args:
            documents: Pre-loaded LangChain Documents; loads from `PDFPolicyLoader` if None.
            embeddings: Optional injected Embeddings model; resolves via `LLMFactory` if None.
            pdf_path: Optional path to the store policies PDF when `documents` is None.
            bm25_weight: Weight assigned to sparse BM25 retriever (default 0.4).
            faiss_weight: Weight assigned to dense FAISS retriever (default 0.6).
            top_k: Default number of top fused chunks returned per query.
            settings: Optional Settings override forwarded to `LLMFactory.get_embeddings`.
            allow_bm25_fallback: If True, gracefully degrades to lexical BM25 when
                dense embeddings cannot be initialized (e.g. missing API key or outage).
        """
        if bm25_weight < 0.0 or faiss_weight < 0.0 or (bm25_weight + faiss_weight) <= 0.0:
            raise ValueError(
                "Retriever weights must be non-negative and sum to > 0 "
                f"(got bm25_weight={bm25_weight}, faiss_weight={faiss_weight})."
            )
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")

        self._lock = threading.RLock()
        self.bm25_weight: float = float(bm25_weight)
        self.faiss_weight: float = float(faiss_weight)
        self.top_k: int = top_k
        self.allow_bm25_fallback: bool = allow_bm25_fallback
        self.is_degraded_lexical_only: bool = False

        loaded_docs = documents if documents is not None else PDFPolicyLoader(pdf_path).load()
        if not loaded_docs:
            raise ValueError("Cannot initialize HybridRetriever with an empty document list.")
        self.documents: list[Document] = loaded_docs

        fetch_k = min(len(self.documents), max(self.top_k * 2, 8))
        self.bm25_retriever: BM25Retriever = _PositiveScoreBM25Retriever.from_documents(
            self.documents,
            preprocess_func=preprocess_pt_bm25,
            k=fetch_k,
        )

        self.vectorstore: FAISS | None = None
        self.ensemble_retriever: EnsembleRetriever | None = None

        if self.faiss_weight > 0.0:
            try:
                active_embeddings = (
                    embeddings
                    if embeddings is not None
                    else LLMFactory.get_embeddings(settings=settings)
                )
                self.vectorstore = FAISS.from_documents(self.documents, active_embeddings)
                faiss_retriever = self.vectorstore.as_retriever(search_kwargs={"k": fetch_k})
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[self.bm25_retriever, faiss_retriever],
                    weights=[self.bm25_weight, self.faiss_weight],
                )
            except Exception as exc:
                if not self.allow_bm25_fallback:
                    raise
                logger.warning(
                    "Dense FAISS index unavailable (%s); falling back to BM25 lexical search.",
                    exc,
                )
                self.is_degraded_lexical_only = True
        else:
            self.is_degraded_lexical_only = True

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Retrieve top-k relevant policy chunks using weighted hybrid fusion.

        Args:
            query: User policy question in natural language.
            top_k: Optional override for maximum chunks returned.

        Returns:
            Ranked list of `Document` chunks with preserved section and page metadata.
        """
        clean_query = query.strip() if query else ""
        if not clean_query:
            return []

        limit = top_k if top_k is not None else self.top_k
        if limit < 1:
            return []

        with self._lock:
            if self.ensemble_retriever is not None and not self.is_degraded_lexical_only:
                try:
                    fused = self.ensemble_retriever.invoke(clean_query)
                    return fused[:limit]
                except Exception as exc:
                    if not self.allow_bm25_fallback:
                        raise
                    logger.warning(
                        "Hybrid dense query failed (%s); serving BM25 fallback results.",
                        exc,
                    )

            lexical_docs = self.bm25_retriever.invoke(clean_query)
            return lexical_docs[:limit]

    def invoke(self, query: str) -> list[Document]:
        """LangChain Runnable-compatible alias for `retrieve`."""
        return self.retrieve(query)


_hybrid_retriever_instance: HybridRetriever | None = None
_retriever_lock = threading.RLock()


def get_hybrid_retriever(
    pdf_path: Path | str | None = None,
    embeddings: Embeddings | None = None,
    allow_bm25_fallback: bool = True,
) -> HybridRetriever:
    """Return a cached singleton HybridRetriever (or create an isolated one for custom paths).

    Args:
        pdf_path: Optional custom PDF path (bypasses singleton cache when provided).
        embeddings: Optional explicit Embeddings instance for initialization.
        allow_bm25_fallback: Whether to degrade gracefully if no embedding API key is set.

    Returns:
        Configured `HybridRetriever` instance.
    """
    global _hybrid_retriever_instance
    with _retriever_lock:
        if pdf_path is not None:
            return HybridRetriever(
                pdf_path=pdf_path,
                embeddings=embeddings,
                allow_bm25_fallback=allow_bm25_fallback,
            )
        if _hybrid_retriever_instance is None:
            _hybrid_retriever_instance = HybridRetriever(
                embeddings=embeddings,
                allow_bm25_fallback=allow_bm25_fallback,
            )
        return _hybrid_retriever_instance


def reset_hybrid_retriever() -> None:
    """Reset the cached singleton HybridRetriever instance."""
    global _hybrid_retriever_instance
    with _retriever_lock:
        _hybrid_retriever_instance = None
