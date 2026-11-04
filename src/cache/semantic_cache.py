"""Context-aware in-memory semantic cache with strict transactional and PII bypass."""

from __future__ import annotations

import re
import threading
import time
from enum import StrEnum
from typing import Any, ClassVar

import numpy as np
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

from src.agent.llm_factory import LLMFactory
from src.config import Settings, get_settings
from src.security.pii_anonymizer import PIIAnonymizer, get_pii_anonymizer


class CacheStatus(StrEnum):
    """Execution status returned by a semantic cache lookup."""

    HIT = "HIT"
    MISS = "MISS"
    BYPASS_TRANSACTIONAL = "BYPASS_TRANSACTIONAL"


class CacheLookupResult(BaseModel):
    """Detailed result of a semantic cache lookup attempt."""

    status: CacheStatus = Field(
        ...,
        description="Lookup status (HIT, MISS, or BYPASS_TRANSACTIONAL)",
    )
    response: str | None = Field(
        default=None,
        description="Cached response text if status is HIT; None otherwise",
    )
    similarity: float | None = Field(
        default=None,
        description="Cosine similarity score of the best matched query (0.0 to 1.0)",
    )
    matched_query: str | None = Field(
        default=None,
        description="The original user query stored in the cache that matched",
    )
    latency_ms: float = Field(
        default=0.0,
        description="Execution duration of the cache lookup in milliseconds",
    )
    bypass_reason: str | None = Field(
        default=None,
        description="Explanation if lookup was bypassed due to transactional/PII rules",
    )


class CacheStats(BaseModel):
    """Cumulative observability statistics for the semantic cache."""

    total_requests: int = Field(default=0, description="Total cache lookup requests")
    hits: int = Field(default=0, description="Count of cache hits")
    misses: int = Field(default=0, description="Count of cache misses")
    bypasses: int = Field(default=0, description="Count of transactional/PII bypasses")
    hit_ratio: float = Field(default=0.0, description="Ratio of hits over total requests")
    entries_count: int = Field(default=0, description="Number of entries currently cached")


class CachedItem:
    """Internal representation of a cached query-response vector pair."""

    def __init__(
        self,
        query: str,
        embedding: list[float],
        response: str,
        metadata: dict[str, Any] | None = None,
        created_at: float | None = None,
    ) -> None:
        self.query = query
        self.embedding = np.array(embedding, dtype=np.float32)
        # Precompute L2 norm for fast normalized cosine similarity
        norm = np.linalg.norm(self.embedding)
        self.normalized_embedding = self.embedding / norm if norm > 0 else self.embedding
        self.response = response
        self.metadata = metadata or {}
        self.created_at = created_at or time.time()


class SemanticCache:
    """Context-aware in-memory semantic cache.

    Implements cosine similarity matching for institutional questions (address, hours, policies)
    and strictly enforces zero-token bypass on transactional, order, inventory, and PII queries.
    """

    # Transactional intent patterns that MUST bypass the semantic cache
    ORDER_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(r"\bpedido\s*#?\s*\d+\b", re.IGNORECASE),
        re.compile(r"\b(meu\s+pedido|status\s+do\s+pedido|rastre(io|amento))\b", re.IGNORECASE),
        re.compile(r"\bc[oó]digo\s+de\s+rastreio\b", re.IGNORECASE),
        re.compile(r"\bBR[A-Z0-9]{9}BR\b", re.IGNORECASE),
        re.compile(
            r"\b(onde\s+est[aá]\s+meu\s+pedido|quando\s+chega\s+meu\s+pedido)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(cancelar|cancelamento\s+de)\s+pedido\b", re.IGNORECASE),
    ]

    STOCK_INVENTORY_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(r"\b(tem|t[eê]m|h[aá])\s+(dispon[ií]ve(l|is)\s+)?em\s+estoque\b", re.IGNORECASE),
        re.compile(r"\bdispon[ií]ve(l|is)\s+(no|em)\s+estoque\b", re.IGNORECASE),
        re.compile(r"\bquant(as|os)\s+unidades\b", re.IGNORECASE),
        re.compile(r"\b(qual\s+o|tem|t[eê]m)\s+estoque\b", re.IGNORECASE),
        re.compile(r"\bainda\s+(tem|t[eê]m)\b", re.IGNORECASE),
        re.compile(r"\b(em\s+estoque|no\s+estoque)\b", re.IGNORECASE),
    ]

    def __init__(
        self,
        embeddings: Embeddings | None = None,
        settings: Settings | None = None,
        anonymizer: PIIAnonymizer | None = None,
    ) -> None:
        self.settings: Settings = settings or get_settings()
        self.threshold: float = self.settings.semantic_cache_threshold
        self.anonymizer: PIIAnonymizer = anonymizer or get_pii_anonymizer()

        self._embeddings: Embeddings | None = embeddings
        self._entries: list[CachedItem] = []
        self._lock = threading.Lock()

        # Observability counters
        self._total_requests: int = 0
        self._hits: int = 0
        self._misses: int = 0
        self._bypasses: int = 0

    @property
    def embeddings(self) -> Embeddings:
        """Lazily initialize or return the embeddings model."""
        if self._embeddings is None:
            self._embeddings = LLMFactory.get_embeddings(self.settings)
        return self._embeddings

    def check_bypass(self, query: str) -> tuple[bool, str | None]:
        """Check whether a query must strictly bypass the semantic cache.

        Args:
            query: The incoming customer query.

        Returns:
            Tuple of (should_bypass: bool, reason: str | None).
        """
        # 1. PII detection (CPF, email, phone)
        if self.anonymizer.contains_pii(query):
            return True, "Consulta contém dados pessoais (PII: CPF, e-mail ou telefone)."

        # 2. Order status and shipment tracking
        for pattern in self.ORDER_PATTERNS:
            if pattern.search(query):
                return True, "Consulta transacional referente a status ou rastreamento de pedido."

        # 3. Live stock and inventory inquiries
        for pattern in self.STOCK_INVENTORY_PATTERNS:
            if pattern.search(query):
                return True, "Consulta de disponibilidade dinâmica de estoque."

        return False, None

    def get(self, query: str) -> CacheLookupResult:
        """Look up a semantic match for the incoming user query.

        If the query violates transactional or PII safety rules, it immediately
        bypasses the cache with zero token consumption and minimal latency.

        Args:
            query: Customer inquiry string.

        Returns:
            CacheLookupResult with status (HIT, MISS, BYPASS_TRANSACTIONAL) and metadata.
        """
        start_time = time.perf_counter()

        # Step 1: Check bypass rules (instant, zero embedding tokens)
        should_bypass, bypass_reason = self.check_bypass(query)
        if should_bypass:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            with self._lock:
                self._total_requests += 1
                self._bypasses += 1

            return CacheLookupResult(
                status=CacheStatus.BYPASS_TRANSACTIONAL,
                response=None,
                similarity=None,
                matched_query=None,
                latency_ms=round(latency_ms, 2),
                bypass_reason=bypass_reason,
            )

        with self._lock:
            if not self._entries:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                self._total_requests += 1
                self._misses += 1
                return CacheLookupResult(
                    status=CacheStatus.MISS,
                    response=None,
                    similarity=None,
                    matched_query=None,
                    latency_ms=round(latency_ms, 2),
                    bypass_reason=None,
                )

        # Step 2: Generate embedding for query
        raw_query_emb = self.embeddings.embed_query(query)
        query_vec = np.array(raw_query_emb, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        normalized_query_vec = query_vec / norm if norm > 0 else query_vec

        # Step 3: Compute cosine similarity against all cached entries
        with self._lock:
            best_similarity = -1.0
            best_entry: CachedItem | None = None

            for entry in self._entries:
                sim = float(np.dot(normalized_query_vec, entry.normalized_embedding))
                if sim > best_similarity:
                    best_similarity = sim
                    best_entry = entry

            self._total_requests += 1
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            if best_entry is not None and best_similarity >= self.threshold:
                self._hits += 1
                return CacheLookupResult(
                    status=CacheStatus.HIT,
                    response=best_entry.response,
                    similarity=round(best_similarity, 4),
                    matched_query=best_entry.query,
                    latency_ms=round(latency_ms, 2),
                    bypass_reason=None,
                )

            self._misses += 1
            return CacheLookupResult(
                status=CacheStatus.MISS,
                response=None,
                similarity=round(best_similarity, 4) if best_similarity >= 0 else None,
                matched_query=None,
                latency_ms=round(latency_ms, 2),
                bypass_reason=None,
            )

    def set(
        self,
        query: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store a verified query-response pair in the semantic cache.

        Refuses storage if the query violates bypass rules or if the response
        contains sensitive customer PII.

        Args:
            query: The verified institutional customer query.
            response: The accurate institutional response to cache.
            metadata: Optional additional context or category metadata.

        Returns:
            True if cached successfully; False if rejected by bypass/safety rules.
        """
        # Refuse to cache queries that trigger transactional or PII bypass
        should_bypass, _ = self.check_bypass(query)
        if should_bypass:
            return False

        # Refuse to cache responses containing PII
        if self.anonymizer.contains_pii(response):
            return False

        # Generate embedding and store
        raw_emb = self.embeddings.embed_query(query)
        item = CachedItem(
            query=query,
            embedding=raw_emb,
            response=response,
            metadata=metadata,
        )

        with self._lock:
            self._entries.append(item)

        return True

    def clear(self) -> None:
        """Clear all cached entries and reset operational counters."""
        with self._lock:
            self._entries.clear()
            self._total_requests = 0
            self._hits = 0
            self._misses = 0
            self._bypasses = 0

    def stats(self) -> CacheStats:
        """Return cumulative observability metrics for the cache."""
        with self._lock:
            hit_ratio = (
                round(self._hits / self._total_requests, 4) if self._total_requests > 0 else 0.0
            )
            return CacheStats(
                total_requests=self._total_requests,
                hits=self._hits,
                misses=self._misses,
                bypasses=self._bypasses,
                hit_ratio=hit_ratio,
                entries_count=len(self._entries),
            )


_global_semantic_cache: SemanticCache | None = None
_semantic_cache_lock = threading.Lock()


def get_semantic_cache(
    embeddings: Embeddings | None = None,
    settings: Settings | None = None,
) -> SemanticCache:
    """Return a thread-safe singleton instance of SemanticCache."""
    global _global_semantic_cache
    with _semantic_cache_lock:
        if _global_semantic_cache is None:
            _global_semantic_cache = SemanticCache(
                embeddings=embeddings,
                settings=settings,
            )
        return _global_semantic_cache


def reset_semantic_cache() -> None:
    """Reset the global semantic cache singleton."""
    global _global_semantic_cache
    with _semantic_cache_lock:
        if _global_semantic_cache is not None:
            _global_semantic_cache.clear()
        _global_semantic_cache = None
