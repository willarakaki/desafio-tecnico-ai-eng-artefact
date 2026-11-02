"""Empório da Música Intelligent Assistant package."""

from src.agent.llm_factory import LLMFactory
from src.config import Settings, get_settings
from src.database.duckdb_manager import DuckDBManager, get_duckdb_manager, reset_duckdb_manager
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

__all__: list[str] = [
    "ActivePromotionsResult",
    "CatalogSearchResult",
    "DuckDBManager",
    "LLMFactory",
    "OrderDetail",
    "OrderItemDetail",
    "OrderStatusResult",
    "ProductDetail",
    "ProductDetailsResult",
    "ProductSummary",
    "PromotionDetail",
    "PromotionSummary",
    "Settings",
    "create_catalog_tools",
    "create_order_tools",
    "get_active_promotions",
    "get_duckdb_manager",
    "get_order_status",
    "get_product_details",
    "get_settings",
    "reset_duckdb_manager",
    "search_products",
]
