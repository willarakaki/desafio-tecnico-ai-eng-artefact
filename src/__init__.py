"""Empório da Música Intelligent Assistant package."""

from src.agent.llm_factory import LLMFactory
from src.cache.semantic_cache import (
    CacheLookupResult,
    CacheStats,
    CacheStatus,
    SemanticCache,
    get_semantic_cache,
    reset_semantic_cache,
)
from src.config import Settings, get_settings
from src.database.duckdb_manager import DuckDBManager, get_duckdb_manager, reset_duckdb_manager
from src.rag.hybrid_retriever import (
    HybridRetriever,
    get_hybrid_retriever,
    reset_hybrid_retriever,
)
from src.rag.pdf_loader import PDFPolicyLoader, load_store_policy_documents
from src.security.guardrails import (
    GuardrailCategory,
    GuardrailResult,
    InputGuardrail,
    get_input_guardrail,
    reset_input_guardrail,
)
from src.security.pii_anonymizer import (
    PIIAnonymizer,
    PIIEntityResult,
    get_pii_anonymizer,
    reset_pii_anonymizer,
)
from src.security.rate_limiter import (
    RateLimitResult,
    SlidingWindowRateLimiter,
    get_rate_limiter,
    reset_rate_limiter,
)
from src.tools.catalog_tools import (
    ActivePromotionsResult,
    CatalogSearchResult,
    ProductDetail,
    ProductDetailsResult,
    ProductSummary,
    PromotionDetail,
    PromotionSummary,
    create_catalog_tools,
    get_active_promotions,
    get_product_details,
    search_products,
)
from src.tools.order_tools import (
    OrderDetail,
    OrderItemDetail,
    OrderStatusResult,
    create_order_tools,
    get_order_status,
)
from src.tools.policy_tools import (
    ConsultStorePoliciesInput,
    PolicyChunkResult,
    StorePoliciesResult,
    consult_store_policies,
    create_policy_tools,
)

__all__: list[str] = [
    "ActivePromotionsResult",
    "CacheLookupResult",
    "CacheStats",
    "CacheStatus",
    "CatalogSearchResult",
    "ConsultStorePoliciesInput",
    "DuckDBManager",
    "GuardrailCategory",
    "GuardrailResult",
    "HybridRetriever",
    "InputGuardrail",
    "LLMFactory",
    "OrderDetail",
    "OrderItemDetail",
    "OrderStatusResult",
    "PDFPolicyLoader",
    "PIIAnonymizer",
    "PIIEntityResult",
    "PolicyChunkResult",
    "ProductDetail",
    "ProductDetailsResult",
    "ProductSummary",
    "PromotionDetail",
    "PromotionSummary",
    "RateLimitResult",
    "SemanticCache",
    "Settings",
    "SlidingWindowRateLimiter",
    "StorePoliciesResult",
    "consult_store_policies",
    "create_catalog_tools",
    "create_order_tools",
    "create_policy_tools",
    "get_active_promotions",
    "get_duckdb_manager",
    "get_hybrid_retriever",
    "get_input_guardrail",
    "get_order_status",
    "get_pii_anonymizer",
    "get_product_details",
    "get_rate_limiter",
    "get_semantic_cache",
    "get_settings",
    "load_store_policy_documents",
    "reset_duckdb_manager",
    "reset_hybrid_retriever",
    "reset_input_guardrail",
    "reset_pii_anonymizer",
    "reset_rate_limiter",
    "reset_semantic_cache",
    "search_products",
]
