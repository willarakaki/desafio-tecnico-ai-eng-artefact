"""Deterministic catalog and promotion tools for Empório da Música."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from src.database.duckdb_manager import DuckDBManager, get_duckdb_manager

logger = logging.getLogger(__name__)

# Category mapping supporting Portuguese singular, plural, and variations
CATEGORY_ALIASES: dict[str, int] = {
    # 1: Guitarras
    "guitarra": 1,
    "guitarras": 1,
    "eletrica": 1,
    "eletricas": 1,
    # 2: Baixos
    "baixo": 2,
    "baixos": 2,
    "contrabaixo": 2,
    "contrabaixos": 2,
    "bass": 2,
    # 3: Baterias e Percussão
    "bateria": 3,
    "baterias": 3,
    "percussao": 3,
    "percussoes": 3,
    "percussão": 3,
    # 4: Teclados e Pianos
    "teclado": 4,
    "teclados": 4,
    "piano": 4,
    "pianos": 4,
    "sintetizador": 4,
    "sintetizadores": 4,
    "synth": 4,
    # 5: Violões
    "violao": 5,
    "violoes": 5,
    "violão": 5,
    "violões": 5,
    "acustico": 5,
    "acusticos": 5,
    "eletroacustico": 5,
    "eletroacusticos": 5,
    # 6: Instrumentos de Sopro (Madeiras)
    "sopro": 6,
    "sopros": 6,
    "madeiras": 6,
    "madeira": 6,
    "flauta": 6,
    "flautas": 6,
    "clarinete": 6,
    "clarinetes": 6,
    "saxofone": 6,
    "saxofones": 6,
    "sax": 6,
    "oboe": 6,
    "oboes": 6,
    # 7: Instrumentos de Sopro (Metais)
    "metais": 7,
    "metal": 7,
    "trompete": 7,
    "trompetes": 7,
    "trombone": 7,
    "trombones": 7,
    "tuba": 7,
    "tubas": 7,
    "trompa": 7,
    "trompas": 7,
    # 8: Cordas Orquestrais
    "cordas orquestrais": 8,
    "orquestral": 8,
    "orquestrais": 8,
    "violino": 8,
    "violinos": 8,
    "viola": 8,
    "violas": 8,
    "violoncelo": 8,
    "violoncelos": 8,
    "cello": 8,
    # 9: Ukuleles
    "ukulele": 9,
    "ukuleles": 9,
    "ukelele": 9,
    "ukeleles": 9,
    "uku": 9,
}

# Accessory keywords that store does not sell (Section 1.2 & 7.2)
ACCESSORY_KEYWORDS = [
    "corda",
    "cordas",
    "encordoamento",
    "encordoamentos",
    "palheta",
    "palhetas",
    "cabo",
    "cabos",
    "pedal",
    "pedais",
    "amplificador",
    "amplificadores",
    "case",
    "cases",
    "capa",
    "capas",
    "afinador",
    "afinadores",
    "suporte",
    "suportes",
    "bateria 9v",
    "correia",
    "correias",
]


# ============================================================================
# Pydantic v2 Input & Output Schemas
# ============================================================================


class SearchProductsInput(BaseModel):
    """Input parameters for searching products in the store catalog."""

    model_config = ConfigDict(extra="forbid")

    query: str | None = Field(
        default=None,
        description="Termo de busca textual (ex: 'Yamaha', 'nylon', 'dreadnought').",
    )
    category: str | None = Field(
        default=None,
        description="Categoria do instrumento (ex: 'Violões', 'Guitarras', 'Ukuleles').",
    )
    max_price: float | None = Field(
        default=None,
        description="Preço máximo em reais (BRL).",
    )
    in_stock_only: bool = Field(
        default=True,
        description="Se True, filtra produtos ativos e com estoque > 0. Se False, inclui todos.",
    )


class PromotionSummary(BaseModel):
    """Summary of a product promotion."""

    promotion_id: int
    discount_percent: int
    description: str
    is_active: bool


class ProductSummary(BaseModel):
    """Summary representation of a catalog product."""

    product_id: int
    name: str
    category_id: int
    category_name: str
    price_brl: float
    promotional_price_brl: float | None = None
    discount_percent: int | None = None
    stock_quantity: int
    status: str
    is_available: bool
    specs: dict[str, Any] = Field(default_factory=dict)
    description: str


class CatalogSearchResult(BaseModel):
    """Result of catalog product search."""

    total_found: int
    items: list[ProductSummary]
    filters_applied: dict[str, Any]
    message: str


class GetProductDetailsInput(BaseModel):
    """Input parameters for retrieving full details of a specific product."""

    model_config = ConfigDict(extra="forbid")

    product_name_or_id: int | str = Field(
        description="Identificador numérico (ex: 95) ou nome do instrumento (ex: 'Takamine GD20').",
    )


class ProductDetail(BaseModel):
    """Full detail of a catalog product."""

    product_id: int
    name: str
    category_id: int
    category_name: str
    price_brl: float
    promotional_price_brl: float | None = None
    discount_percent: int | None = None
    stock_quantity: int
    status: str
    is_available: bool
    specs: dict[str, Any] = Field(default_factory=dict)
    description: str
    created_at: str


class ProductDetailsResult(BaseModel):
    """Result of looking up product details."""

    found: bool
    product: ProductDetail | None = None
    active_promotion: PromotionSummary | None = None
    expired_promotions: list[PromotionSummary] = Field(default_factory=list)
    message: str


class GetActivePromotionsInput(BaseModel):
    """Input parameters for querying active promotions."""

    model_config = ConfigDict(extra="forbid")

    category_or_product: str | None = Field(
        default=None,
        description="Filtro opcional por nome da categoria ou instrumento.",
    )


class PromotionDetail(BaseModel):
    """Detailed active promotion information."""

    promotion_id: int
    product_id: int
    product_name: str
    category_name: str
    original_price_brl: float
    discount_percent: int
    promotional_price_brl: float
    description: str
    is_active: bool
    stock_quantity: int
    status: str
    pix_price_brl: float
    non_cumulative_pix_note: str


class ActivePromotionsResult(BaseModel):
    """Result of active promotions lookup."""

    total_promotions: int
    promotions: list[PromotionDetail]
    message: str


# ============================================================================
# Helper Functions
# ============================================================================


def _resolve_category_id(category_name_or_term: str | None, db: DuckDBManager) -> int | None:
    """Resolve category term to category_id using alias dictionary or DuckDB query."""
    if not category_name_or_term:
        return None

    normalized = DuckDBManager.normalize_text(category_name_or_term)
    if normalized in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[normalized]

    # Check against category names in DuckDB
    sql = (
        "SELECT category_id FROM categories "
        "WHERE normalize_pt(name) = ? OR normalize_pt(name) LIKE ?"
    )
    rows = db.execute_query(sql, [normalized, f"%{normalized}%"])
    if rows:
        return int(rows[0]["category_id"])

    return None


def _check_accessory_inquiry(text: str) -> bool:
    """Check if query is asking for accessories that the store doesn't sell."""
    normalized = DuckDBManager.normalize_text(text)
    tokens = set(normalized.split())
    return any(kw in tokens or kw == normalized for kw in ACCESSORY_KEYWORDS)


def _safe_parse_specs(raw_specs: Any) -> dict[str, Any]:
    """Parse JSON specs field safely."""
    if isinstance(raw_specs, dict):
        return raw_specs
    if isinstance(raw_specs, str):
        try:
            return json.loads(raw_specs)
        except json.JSONDecodeError:
            return {}
    return {}


# ============================================================================
# Deterministic Tool Implementations
# ============================================================================


def search_products(
    query: str | None = None,
    category: str | None = None,
    max_price: float | None = None,
    in_stock_only: bool = True,
    *,
    db: DuckDBManager | None = None,
) -> CatalogSearchResult:
    """Search products in the Empório da Música catalog with deterministic filters.

    Args:
        query: Optional free-text search term (matches name, description, specs, category).
        category: Optional category name or family (e.g. 'Violões', 'Guitarras', 'Ukuleles').
        max_price: Optional maximum price in BRL.
        in_stock_only: If True, only returns active products with stock > 0.
        db: Optional DuckDBManager instance (defaults to singleton).

    Returns:
        CatalogSearchResult with found products and metadata.
    """
    try:
        active_db = db or get_duckdb_manager()
        conditions: list[str] = []
        params: list[Any] = []

        filters_applied: dict[str, Any] = {
            "query": query,
            "category": category,
            "max_price": max_price,
            "in_stock_only": in_stock_only,
        }

        # Check for accessory inquiry first
        if query and _check_accessory_inquiry(query):
            return CatalogSearchResult(
                total_found=0,
                items=[],
                filters_applied=filters_applied,
                message=(
                    "A Empório da Música comercializa exclusivamente instrumentos musicais "
                    "(violões, guitarras, baixos, baterias, teclados e ukuleles). "
                    "Não vendemos acessórios avulsos (cordas, palhetas, cabos, pedais, etc.)."
                ),
            )

        # Category resolution
        resolved_category_id: int | None = None
        if category:
            resolved_category_id = _resolve_category_id(category, active_db)
            if resolved_category_id is not None:
                conditions.append("p.category_id = ?")
                params.append(resolved_category_id)
            else:
                # Category name not found in categories table
                return CatalogSearchResult(
                    total_found=0,
                    items=[],
                    filters_applied=filters_applied,
                    message=f"Categoria '{category}' não encontrada no catálogo da loja.",
                )

        # Stock and status filter
        if in_stock_only:
            conditions.append("p.status = 'active' AND p.stock_quantity > 0")

        # Max price filter
        if max_price is not None:
            if max_price < 0:
                return CatalogSearchResult(
                    total_found=0,
                    items=[],
                    filters_applied=filters_applied,
                    message="O preço máximo informado não pode ser negativo.",
                )
            conditions.append("p.price_brl <= ?")
            params.append(float(max_price))

        # Query filter
        if query:
            norm_q = DuckDBManager.normalize_text(query)
            # If query corresponds to a category family and category was not explicitly specified
            inferred_cat_id = (
                _resolve_category_id(norm_q, active_db) if resolved_category_id is None else None
            )

            if inferred_cat_id is not None:
                # Scoped by inferred category from query
                conditions.append("p.category_id = ?")
                params.append(inferred_cat_id)
            else:
                like_pattern = f"%{norm_q}%"
                conditions.append(
                    """(
                        normalize_pt(p.name) LIKE ?
                        OR normalize_pt(p.description) LIKE ?
                        OR normalize_pt(c.name) LIKE ?
                        OR normalize_pt(p.specs) LIKE ?
                    )"""
                )
                params.extend([like_pattern, like_pattern, like_pattern, like_pattern])

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # Join products with categories and only active promotions (is_active = 1)
        sql = f"""
            SELECT
                p.product_id,
                p.price_brl,
                p.name,
                p.category_id,
                c.name AS category_name,
                p.description,
                p.stock_quantity,
                p.status,
                p.specs,
                pr.promotion_id,
                pr.discount_percent,
                pr.description AS promo_desc,
                pr.is_active AS promo_is_active
            FROM products p
            JOIN categories c ON p.category_id = c.category_id
            LEFT JOIN promotions pr ON p.product_id = pr.product_id AND pr.is_active = 1
            {where_clause}
            ORDER BY p.price_brl ASC, p.product_id ASC
        """

        rows = active_db.execute_query(sql, params)

        items: list[ProductSummary] = []
        for row in rows:
            price = float(row["price_brl"])
            discount = int(row["discount_percent"]) if row["discount_percent"] is not None else None
            promo_price = round(price * (1 - discount / 100), 2) if discount is not None else None
            stock = int(row["stock_quantity"])
            status = str(row["status"])
            is_available = status == "active" and stock > 0

            items.append(
                ProductSummary(
                    product_id=int(row["product_id"]),
                    name=str(row["name"]),
                    category_id=int(row["category_id"]),
                    category_name=str(row["category_name"]),
                    price_brl=price,
                    promotional_price_brl=promo_price,
                    discount_percent=discount,
                    stock_quantity=stock,
                    status=status,
                    is_available=is_available,
                    specs=_safe_parse_specs(row["specs"]),
                    description=str(row["description"]),
                )
            )

        # Craft explanatory message
        if not items:
            if resolved_category_id in (6, 7, 8):
                # Empty operational categories
                cat_row = active_db.execute_query(
                    "SELECT name FROM categories WHERE category_id = ?", [resolved_category_id]
                )
                cat_name = cat_row[0]["name"] if cat_row else category
                message = (
                    f"A categoria '{cat_name}' faz parte do catálogo institucional da loja, "
                    "mas no momento não há instrumentos cadastrados em estoque."
                )
            else:
                message = (
                    "Nenhum produto encontrado com os filtros informados. "
                    "Tente buscar com termos mais amplos ou verificar outras categorias."
                )
        else:
            message = (
                f"Encontrado(s) {len(items)} produto(s) correspondente(s) aos critérios de busca."
            )

        return CatalogSearchResult(
            total_found=len(items),
            items=items,
            filters_applied=filters_applied,
            message=message,
        )

    except Exception as exc:
        logger.exception("Error executing search_products: %s", exc)
        return CatalogSearchResult(
            total_found=0,
            items=[],
            filters_applied={
                "query": query,
                "category": category,
                "max_price": max_price,
                "in_stock_only": in_stock_only,
            },
            message="Ocorreu um erro ao consultar o catálogo. Por favor, tente novamente.",
        )


def get_product_details(
    product_name_or_id: int | str,
    *,
    db: DuckDBManager | None = None,
) -> ProductDetailsResult:
    """Retrieve full details of a specific instrument by ID or name, including promotion history.

    Args:
        product_name_or_id: Product ID (integer or string) or product name.
        db: Optional DuckDBManager instance (defaults to singleton).

    Returns:
        ProductDetailsResult containing product details, active promo, expired promos, and status.
    """
    try:
        active_db = db or get_duckdb_manager()

        # Check for accessory inquiry first
        str_input = str(product_name_or_id).strip()
        if _check_accessory_inquiry(str_input):
            return ProductDetailsResult(
                found=False,
                product=None,
                active_promotion=None,
                expired_promotions=[],
                message=(
                    "A Empório da Música comercializa exclusivamente instrumentos musicais. "
                    "Não vendemos acessórios avulsos (cordas, palhetas, cabos, pedais, etc.)."
                ),
            )

        # Parse ID if input is an integer or numeric string like "#95" or "95"
        clean_num = str_input.lstrip("#").strip()
        product_id_val: int | None = int(clean_num) if clean_num.isdigit() else None

        if product_id_val is not None:
            sql = """
                SELECT
                    p.product_id,
                    p.price_brl,
                    p.name,
                    p.category_id,
                    c.name AS category_name,
                    p.description,
                    p.stock_quantity,
                    p.status,
                    p.specs,
                    p.created_at
                FROM products p
                JOIN categories c ON p.category_id = c.category_id
                WHERE p.product_id = ?
            """
            rows = active_db.execute_query(sql, [product_id_val])
        else:
            norm_name = DuckDBManager.normalize_text(str_input)
            sql = """
                SELECT
                    p.product_id,
                    p.price_brl,
                    p.name,
                    p.category_id,
                    c.name AS category_name,
                    p.description,
                    p.stock_quantity,
                    p.status,
                    p.specs,
                    p.created_at
                FROM products p
                JOIN categories c ON p.category_id = c.category_id
                WHERE normalize_pt(p.name) = ? OR normalize_pt(p.name) LIKE ?
                ORDER BY
                    CASE WHEN normalize_pt(p.name) = ? THEN 0 ELSE 1 END,
                    p.product_id ASC
                LIMIT 1
            """
            rows = active_db.execute_query(sql, [norm_name, f"%{norm_name}%", norm_name])

        if not rows:
            return ProductDetailsResult(
                found=False,
                product=None,
                active_promotion=None,
                expired_promotions=[],
                message=f"Produto '{product_name_or_id}' não foi encontrado no catálogo.",
            )

        p_row = rows[0]
        prod_id = int(p_row["product_id"])
        price = float(p_row["price_brl"])
        stock = int(p_row["stock_quantity"])
        status = str(p_row["status"])
        is_available = status == "active" and stock > 0

        # Query all promotions for this product (active and expired)
        promo_rows = active_db.execute_query(
            """
            SELECT promotion_id, discount_percent, description, is_active
            FROM promotions
            WHERE product_id = ?
            ORDER BY is_active DESC, promotion_id ASC
            """,
            [prod_id],
        )

        active_promo: PromotionSummary | None = None
        expired_promos: list[PromotionSummary] = []

        for pr in promo_rows:
            p_summary = PromotionSummary(
                promotion_id=int(pr["promotion_id"]),
                discount_percent=int(pr["discount_percent"]),
                description=str(pr["description"]),
                is_active=bool(pr["is_active"] == 1),
            )
            if pr["is_active"] == 1 and active_promo is None:
                active_promo = p_summary
            else:
                expired_promos.append(p_summary)

        promotional_price: float | None = None
        discount_pct: int | None = None
        if active_promo is not None:
            discount_pct = active_promo.discount_percent
            promotional_price = round(price * (1 - discount_pct / 100), 2)

        product = ProductDetail(
            product_id=prod_id,
            name=str(p_row["name"]),
            category_id=int(p_row["category_id"]),
            category_name=str(p_row["category_name"]),
            price_brl=price,
            promotional_price_brl=promotional_price,
            discount_percent=discount_pct,
            stock_quantity=stock,
            status=status,
            is_available=is_available,
            specs=_safe_parse_specs(p_row["specs"]),
            description=str(p_row["description"]),
            created_at=str(p_row["created_at"]),
        )

        # Build consultative status message
        msg_parts = [
            f"{product.name} (ID #{product.product_id}): Preço regular R$ {product.price_brl:.2f}."
        ]
        if not is_available:
            if status == "discontinued":
                msg_parts.append("Status: Produto descontinuado.")
            elif status == "coming_soon":
                msg_parts.append("Status: Em breve no estoque.")
            elif stock == 0:
                msg_parts.append("Status: Temporariamente esgotado.")
        else:
            msg_parts.append(f"Em estoque ({product.stock_quantity} unidades disponíveis).")

        if active_promo is not None:
            msg_parts.append(
                f"Promoção ativa: '{active_promo.description}' "
                f"({active_promo.discount_percent}% de desconto)! "
                f"Preço promocional: R$ {promotional_price:.2f}."
            )
        elif expired_promos:
            exp = expired_promos[0]
            msg_parts.append(
                f"Atenção: A promoção '{exp.description}' ({exp.discount_percent}%) está expirada. "
                f"O preço vigente é o valor de tabela de R$ {product.price_brl:.2f}."
            )

        return ProductDetailsResult(
            found=True,
            product=product,
            active_promotion=active_promo,
            expired_promotions=expired_promos,
            message=" ".join(msg_parts),
        )

    except Exception as exc:
        logger.exception("Error executing get_product_details: %s", exc)
        return ProductDetailsResult(
            found=False,
            product=None,
            active_promotion=None,
            expired_promotions=[],
            message="Ocorreu um erro ao consultar os detalhes do produto. Tente novamente.",
        )


def get_active_promotions(
    category_or_product: str | None = None,
    *,
    db: DuckDBManager | None = None,
) -> ActivePromotionsResult:
    """Retrieve currently active promotions with calculated discount prices and policies.

    Args:
        category_or_product: Optional filter by category name or product name/ID.
        db: Optional DuckDBManager instance (defaults to singleton).

    Returns:
        ActivePromotionsResult with all matching active promotional campaigns.
    """
    try:
        active_db = db or get_duckdb_manager()

        sql = """
            SELECT
                pr.promotion_id,
                pr.product_id,
                p.name AS product_name,
                c.name AS category_name,
                p.category_id,
                p.price_brl,
                pr.discount_percent,
                pr.description AS promo_desc,
                pr.is_active,
                p.stock_quantity,
                p.status
            FROM promotions pr
            JOIN products p ON pr.product_id = p.product_id
            JOIN categories c ON p.category_id = c.category_id
            WHERE pr.is_active = 1
            ORDER BY pr.discount_percent DESC, p.price_brl ASC
        """

        rows = active_db.execute_query(sql)

        # Apply category or product filter if provided
        filtered_rows = rows
        expired_note: str | None = None

        if category_or_product:
            norm_filter = DuckDBManager.normalize_text(category_or_product)
            target_cat_id = _resolve_category_id(norm_filter, active_db)

            filtered: list[dict[str, Any]] = []
            for r in rows:
                p_name_norm = DuckDBManager.normalize_text(r["product_name"])
                c_name_norm = DuckDBManager.normalize_text(r["category_name"])
                if (
                    (target_cat_id is not None and r["category_id"] == target_cat_id)
                    or norm_filter in p_name_norm
                    or norm_filter in c_name_norm
                ):
                    filtered.append(r)

            # If no active promotions, check if there is an expired promotion
            if not filtered:
                clean_num = norm_filter.lstrip("#").strip()
                p_id = int(clean_num) if clean_num.isdigit() else None
                if p_id is not None:
                    exp_sql = (
                        "SELECT description, discount_percent FROM promotions "
                        "WHERE product_id = ? AND is_active = 0"
                    )
                    exp_rows = active_db.execute_query(exp_sql, [p_id])
                else:
                    exp_sql = """
                        SELECT pr.description, pr.discount_percent, p.name, p.price_brl
                        FROM promotions pr
                        JOIN products p ON pr.product_id = p.product_id
                        WHERE (normalize_pt(p.name) LIKE ? OR normalize_pt(p.name) = ?)
                          AND pr.is_active = 0
                    """
                    exp_rows = active_db.execute_query(exp_sql, [f"%{norm_filter}%", norm_filter])

                if exp_rows:
                    exp_r = exp_rows[0]
                    expired_note = (
                        f"O produto '{category_or_product}' não possui promoções ativas agora. "
                        f"A promoção anterior ('{exp_r['description']}' com "
                        f"{exp_r['discount_percent']}% de desconto) está expirada."
                    )

            filtered_rows = filtered

        promotions: list[PromotionDetail] = []
        for r in filtered_rows:
            price = float(r["price_brl"])
            discount = int(r["discount_percent"])
            promo_price = round(price * (1 - discount / 100), 2)

            promotions.append(
                PromotionDetail(
                    promotion_id=int(r["promotion_id"]),
                    product_id=int(r["product_id"]),
                    product_name=str(r["product_name"]),
                    category_name=str(r["category_name"]),
                    original_price_brl=price,
                    discount_percent=discount,
                    promotional_price_brl=promo_price,
                    description=str(r["promo_desc"]),
                    is_active=True,
                    stock_quantity=int(r["stock_quantity"]),
                    status=str(r["status"]),
                    pix_price_brl=promo_price,
                    non_cumulative_pix_note=(
                        "Conforme Seção 6.1 do regulamento, os descontos de campanhas promocionais "
                        "não são cumulativos com os 5% de desconto à vista no PIX."
                    ),
                )
            )

        if not promotions:
            msg = (
                expired_note
                or f"Não foram encontradas promoções ativas para '{category_or_product}'."
            )
        else:
            msg = (
                f"Encontrada(s) {len(promotions)} promoção(ões) ativa(s). "
                "Lembrete: descontos promocionais não acumulam com o desconto de 5% no PIX."
            )

        return ActivePromotionsResult(
            total_promotions=len(promotions),
            promotions=promotions,
            message=msg,
        )

    except Exception as exc:
        logger.exception("Error executing get_active_promotions: %s", exc)
        return ActivePromotionsResult(
            total_promotions=0,
            promotions=[],
            message="Ocorreu um erro ao consultar as promoções ativas. Tente novamente.",
        )


# ============================================================================
# LangChain StructuredTool Factory
# ============================================================================


def create_catalog_tools(db: DuckDBManager | None = None) -> list[StructuredTool]:
    """Create LangChain StructuredTool instances for catalog tools.

    Args:
        db: Optional DuckDBManager instance (defaults to singleton).

    Returns:
        List of StructuredTool instances ready for LangGraph agent binding.
    """
    active_db = db or get_duckdb_manager()

    def _search_products_tool(
        query: str | None = None,
        category: str | None = None,
        max_price: float | None = None,
        in_stock_only: bool = True,
    ) -> str:
        res = search_products(
            query=query,
            category=category,
            max_price=max_price,
            in_stock_only=in_stock_only,
            db=active_db,
        )
        return res.model_dump_json(indent=2)

    def _get_product_details_tool(product_name_or_id: int | str) -> str:
        res = get_product_details(product_name_or_id, db=active_db)
        return res.model_dump_json(indent=2)

    def _get_active_promotions_tool(category_or_product: str | None = None) -> str:
        res = get_active_promotions(category_or_product, db=active_db)
        return res.model_dump_json(indent=2)

    return [
        StructuredTool.from_function(
            func=_search_products_tool,
            name="search_products",
            description=(
                "Busca instrumentos musicais no catálogo da Empório da Música com filtros de "
                "termo de busca, categoria (Violões, Guitarras, Baixos, Baterias, Teclados), "
                "preço máximo e disponibilidade de estoque."
            ),
            args_schema=SearchProductsInput,
        ),
        StructuredTool.from_function(
            func=_get_product_details_tool,
            name="get_product_details",
            description=(
                "Obtém a ficha técnica completa, preço de tabela, estoque atual e histórico de "
                "promoções (ativas e expiradas) de um instrumento específico por nome ou ID."
            ),
            args_schema=GetProductDetailsInput,
        ),
        StructuredTool.from_function(
            func=_get_active_promotions_tool,
            name="get_active_promotions",
            description=(
                "Lista promoções e descontos atualmente ativos na loja, com preços calculados "
                "e aviso de não cumulatividade com PIX. Pode filtrar por categoria ou instrumento."
            ),
            args_schema=GetActivePromotionsInput,
        ),
    ]
