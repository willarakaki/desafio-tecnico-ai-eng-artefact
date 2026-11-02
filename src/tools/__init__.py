"""Deterministic tools package for Empório da Música."""

from src.tools.catalog_tools import (
    ActivePromotionsResult,
    CatalogSearchResult,
    GetActivePromotionsInput,
    GetProductDetailsInput,
    ProductDetail,
    ProductDetailsResult,
    ProductSummary,
    PromotionDetail,
    PromotionSummary,
    SearchProductsInput,
    create_catalog_tools,
    get_active_promotions,
    get_product_details,
    search_products,
)
from src.tools.catalog_tools import (
    GetActivePromotionsInput as GetPromotionsSchema,
)
from src.tools.order_tools import (
    GetOrderStatusInput,
    OrderDetail,
    OrderItemDetail,
    OrderStatusResult,
    create_order_tools,
    get_order_status,
)

__all__: list[str] = [
    "ActivePromotionsResult",
    "CatalogSearchResult",
    "GetActivePromotionsInput",
    "GetOrderStatusInput",
    "GetProductDetailsInput",
    "GetPromotionsSchema",
    "OrderDetail",
    "OrderItemDetail",
    "OrderStatusResult",
    "ProductDetail",
    "ProductDetailsResult",
    "ProductSummary",
    "PromotionDetail",
    "PromotionSummary",
    "SearchProductsInput",
    "create_catalog_tools",
    "create_order_tools",
    "get_active_promotions",
    "get_order_status",
    "get_product_details",
    "search_products",
]
