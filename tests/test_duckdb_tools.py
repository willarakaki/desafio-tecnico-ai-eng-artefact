"""Unit and integration tests for DuckDB ingestion and deterministic catalog and order tools."""

from __future__ import annotations

import json
from pathlib import Path

from src.database.duckdb_manager import (
    DuckDBManager,
    get_duckdb_manager,
    reset_duckdb_manager,
)
from src.tools.catalog_tools import (
    create_catalog_tools,
    get_active_promotions,
    get_product_details,
    search_products,
)
from src.tools.order_tools import (
    create_order_tools,
    get_order_status,
)


def setup_function() -> None:
    """Reset singleton state before each test."""
    reset_duckdb_manager()


def teardown_function() -> None:
    """Clean up singleton state after each test."""
    reset_duckdb_manager()


# ============================================================================
# DuckDB Ingestion & Database Tests
# ============================================================================


def test_duckdb_manager_ingestion_and_schemas() -> None:
    """Verify DuckDBManager loads all 6 operational CSVs with exact schema and row counts."""
    db = DuckDBManager()
    tables = db.get_table_names()
    assert tables == ["categories", "customers", "order_items", "orders", "products", "promotions"]

    # Exact ground truth row counts
    assert db.get_table_count("categories") == 9
    assert db.get_table_count("customers") == 50
    assert db.get_table_count("order_items") == 22
    assert db.get_table_count("orders") == 20
    assert db.get_table_count("products") == 65
    assert db.get_table_count("promotions") == 25

    # Check products schema types (INTEGER, DOUBLE, VARCHAR, JSON, DATE)
    prod_cols = {
        col["column_name"]: col["column_type"] for col in db.execute_query("DESCRIBE products")
    }
    assert prod_cols["product_id"] == "INTEGER"
    assert prod_cols["price_brl"] == "DOUBLE"
    assert prod_cols["name"] == "VARCHAR"
    assert prod_cols["category_id"] == "INTEGER"
    assert prod_cols["stock_quantity"] == "INTEGER"
    assert prod_cols["status"] == "VARCHAR"
    assert prod_cols["specs"] == "JSON"
    assert prod_cols["created_at"] == "DATE"

    # Check orders schema types (nullable columns tracking_code, estimated_delivery, notes)
    order_cols = {
        col["column_name"]: col["column_type"] for col in db.execute_query("DESCRIBE orders")
    }
    assert order_cols["order_id"] == "INTEGER"
    assert order_cols["customer_id"] == "INTEGER"
    assert order_cols["order_date"] == "DATE"
    assert order_cols["status"] == "VARCHAR"
    assert order_cols["total_brl"] == "DOUBLE"
    assert order_cols["tracking_code"] == "VARCHAR"
    assert order_cols["estimated_delivery"] == "DATE"
    assert order_cols["notes"] == "VARCHAR"


def test_duckdb_text_normalization_macro_and_python() -> None:
    """Ensure normalize_pt SQL macro and normalize_text Python function produce identical output."""
    db = DuckDBManager()
    sample = "  Violão   Clássico   Elétrico   São Paulo   Três Lagoas  "
    expected = "violao classico eletrico sao paulo tres lagoas"

    py_norm = DuckDBManager.normalize_text(sample)
    assert py_norm == expected

    res = db.execute_query("SELECT normalize_pt(?) AS val", [sample])
    assert res[0]["val"] == expected


def test_duckdb_custom_directory_and_whitespace_trimming(tmp_path: Path) -> None:
    """Verify loading from custom path and whitespace trimming on text columns."""
    custom_csv = tmp_path / "desafio_tecnico_ai_eng - categories.csv"
    custom_csv.write_text(
        "category_id,name,description\n"
        "1,  Guitarras  ,  Guitarras elétricas   \n"
        "2, Baixos , Baixos elétricos \n",
        encoding="utf-8",
    )

    custom_db = DuckDBManager(data_dir=tmp_path)
    rows = custom_db.execute_query(
        "SELECT category_id, name, description FROM categories ORDER BY category_id"
    )
    assert len(rows) == 2
    assert rows[0] == {"category_id": 1, "name": "Guitarras", "description": "Guitarras elétricas"}
    assert rows[1] == {"category_id": 2, "name": "Baixos", "description": "Baixos elétricos"}


# ============================================================================
# Catalog Tools Tests
# ============================================================================


def test_search_products_violoes_up_to_1000() -> None:
    """Ground truth query: acoustic guitars (violões) up to R$ 1.000,00 in stock."""
    db = get_duckdb_manager()

    res = search_products(
        category="violões",
        max_price=1000.0,
        in_stock_only=True,
        db=db,
    )

    assert res.total_found == 12
    assert len(res.items) == 12

    # Every item must be in category 5, price <= 1000, in stock, and active
    for item in res.items:
        assert item.category_id == 5
        assert item.price_brl <= 1000.0
        assert item.stock_quantity > 0
        assert item.status == "active"
        assert item.is_available is True

    # product_id=96 (Giannini GF-3D, active but stock=0) must NOT be in the results
    found_ids = {it.product_id for it in res.items}
    assert 96 not in found_ids
    assert 113 not in found_ids  # Discontinued (stock=0)
    assert 128 not in found_ids  # Ukulele mentioning violão in description


def test_search_products_stock_filter_edge_cases() -> None:
    """Verify in_stock_only=False includes out-of-stock, discontinued, and coming_soon items."""
    db = get_duckdb_manager()

    # Search for Giannini GF-3D (product_id=96, stock=0, active)
    in_stock_res = search_products(query="Giannini GF-3D", in_stock_only=True, db=db)
    assert in_stock_res.total_found == 0

    all_res = search_products(query="Giannini GF-3D", in_stock_only=False, db=db)
    assert all_res.total_found == 1
    p96 = all_res.items[0]
    assert p96.product_id == 96
    assert p96.stock_quantity == 0
    assert p96.status == "active"
    assert p96.is_available is False

    # Search for Ibanez Bass 2X (product_id=136, stock=13, discontinued)
    bass_in_stock = search_products(query="Ibanez Bass 2X", in_stock_only=True, db=db)
    assert bass_in_stock.total_found == 0

    bass_all = search_products(query="Ibanez Bass 2X", in_stock_only=False, db=db)
    assert bass_all.total_found == 1
    p136 = bass_all.items[0]
    assert p136.product_id == 136
    assert p136.status == "discontinued"
    assert p136.is_available is False


def test_search_products_empty_categories_and_synthetic_variance() -> None:
    """Verify handling of empty operational categories and synthetic attribute variance."""
    db = get_duckdb_manager()

    # Empty category: Instrumentos de Sopro (Madeiras)
    res_sopro = search_products(category="sopro", db=db)
    assert res_sopro.total_found == 0
    assert "não há instrumentos cadastrados em estoque" in res_sopro.message

    # Empty category: Cordas Orquestrais
    res_orquestral = search_products(category="Cordas Orquestrais", db=db)
    assert res_orquestral.total_found == 0

    # Synthetic variance: product_id=135 (name: Music Man Bass 1X, desc: Fender Jazz Bass)
    res_synth = search_products(query="Fender Jazz Bass", db=db)
    assert res_synth.total_found >= 1
    assert any(it.product_id == 135 for it in res_synth.items)


def test_search_products_accessory_guard() -> None:
    """Verify that asking for accessories (strings, picks, cables) returns structured refusal."""
    db = get_duckdb_manager()

    res_cordas = search_products(query="encordoamento para violão", db=db)
    assert res_cordas.total_found == 0
    assert "Não vendemos acessórios avulsos" in res_cordas.message

    res_palhetas = search_products(query="palhetas", db=db)
    assert res_palhetas.total_found == 0
    assert "Não vendemos acessórios avulsos" in res_palhetas.message


def test_search_products_invalid_price() -> None:
    """Verify negative max_price is rejected gracefully."""
    db = get_duckdb_manager()
    res = search_products(max_price=-100.0, db=db)
    assert res.total_found == 0
    assert "não pode ser negativo" in res.message


def test_get_product_details_takamine_gd20() -> None:
    """Verify details lookup for Takamine GD20 (product_id=95), including expired promotion."""
    db = get_duckdb_manager()

    # Lookup by name
    res_name = get_product_details("Takamine GD20", db=db)
    assert res_name.found is True
    assert res_name.product is not None
    p = res_name.product
    assert p.product_id == 95
    assert p.name == "Takamine GD20 Dreadnought Natural"
    assert p.price_brl == 2199.0
    assert p.stock_quantity == 5
    assert p.status == "active"
    assert p.is_available is True
    assert p.specs["top"] == "Cedro Sólido"

    # Takamine GD20 promotion in promotions.csv has is_active = 0 (EXPIRED)
    assert res_name.active_promotion is None
    assert len(res_name.expired_promotions) >= 1
    exp_promo = res_name.expired_promotions[0]
    assert exp_promo.discount_percent == 12
    assert exp_promo.description == "Oferta Especial"
    assert "expirada" in res_name.message
    assert "R$ 2199.00" in res_name.message or "2.199" in res_name.message

    # Lookup by numeric ID
    res_id = get_product_details(95, db=db)
    assert res_id.found is True
    assert res_id.product is not None
    assert res_id.product.product_id == 95


def test_get_product_details_active_promo() -> None:
    """Verify product details for Taylor 110e (product_id=94), which has an active 8% promo."""
    db = get_duckdb_manager()

    res = get_product_details(94, db=db)
    assert res.found is True
    assert res.product is not None
    p = res.product
    assert p.product_id == 94
    assert p.price_brl == 5999.0
    assert res.active_promotion is not None
    assert res.active_promotion.discount_percent == 8
    assert p.promotional_price_brl == 5519.08
    assert "Promoção ativa" in res.message


def test_get_product_details_nonexistent_and_accessory() -> None:
    """Verify product details error handling for nonexistent items and accessories."""
    db = get_duckdb_manager()

    res_missing = get_product_details("Instrumento Fantasma 999", db=db)
    assert res_missing.found is False
    assert res_missing.product is None

    res_acc = get_product_details("palheta fender", db=db)
    assert res_acc.found is False
    assert "Não vendemos acessórios avulsos" in res_acc.message


def test_get_active_promotions_all_and_filtered() -> None:
    """Verify active promotions lookup matches the 4 ground-truth active campaigns."""
    db = get_duckdb_manager()

    # Unfiltered call: exactly 4 active promotions in dataset
    all_promos = get_active_promotions(db=db)
    assert all_promos.total_promotions == 4
    active_prod_ids = {p.product_id for p in all_promos.promotions}
    assert active_prod_ids == {90, 94, 121, 127}

    for promo in all_promos.promotions:
        assert promo.is_active is True
        assert promo.promotional_price_brl < promo.original_price_brl
        assert (
            "não são cumulativos com os 5% de desconto à vista no PIX"
            in promo.non_cumulative_pix_note
        )

    # Filtered by category "Ukuleles" -> 2 active promotions (121 and 127)
    uku_promos = get_active_promotions("Ukuleles", db=db)
    assert uku_promos.total_promotions == 2
    assert {p.product_id for p in uku_promos.promotions} == {121, 127}

    # Filtered by Takamine GD20 (product has only expired promotion)
    gd20_promos = get_active_promotions("Takamine GD20", db=db)
    assert gd20_promos.total_promotions == 0
    assert "está expirada" in gd20_promos.message


# ============================================================================
# Order Tools Tests
# ============================================================================


def test_get_order_status_by_order_id() -> None:
    """Verify order lookup across statuses (delivered, shipped, pending, cancelled)."""
    db = get_duckdb_manager()

    # Order 1: delivered, tracking_code BRAB1234567BR
    res_1 = get_order_status(order_id=1, db=db)
    assert res_1.found is True
    assert len(res_1.orders) == 1
    o1 = res_1.orders[0]
    assert o1.order_id == 1
    assert o1.customer_id == 3
    assert o1.status == "delivered"
    assert o1.status_label == "Entregue"
    assert o1.tracking_code == "BRAB1234567BR"
    assert o1.estimated_delivery == "2025-10-25"
    assert len(o1.items) == 1
    assert o1.items[0].product_id == 93

    # Order 8: shipped, tracking_code BRJL5544332BR
    res_8 = get_order_status(order_id="#8", db=db)
    assert res_8.found is True
    o8 = res_8.orders[0]
    assert o8.status == "shipped"
    assert o8.tracking_code == "BRJL5544332BR"

    # Order 15: pending, multi-item (products 82 and 106)
    res_15 = get_order_status(order_id="Pedido 15", db=db)
    assert res_15.found is True
    o15 = res_15.orders[0]
    assert o15.status == "pending"
    assert len(o15.items) == 2
    assert {it.product_id for it in o15.items} == {82, 106}

    # Order 17: cancelled with cancellation notes
    res_17 = get_order_status(order_id=17, db=db)
    assert res_17.found is True
    o17 = res_17.orders[0]
    assert o17.status == "cancelled"
    assert o17.notes == "Pedido cancelado pelo cliente antes do processamento"


def test_get_order_status_by_customer_identifier() -> None:
    """Verify looking up orders by email, digits-normalized phone, and customer name."""
    db = get_duckdb_manager()

    # Lookup by email (pedro.oliveira@jmail.com has orders 1 and 19)
    res_email = get_order_status(customer_identifier="pedro.oliveira@jmail.com", db=db)
    assert res_email.found is True
    assert len(res_email.orders) == 2
    order_ids = {o.order_id for o in res_email.orders}
    assert order_ids == {1, 19}

    # Lookup by phone: customer 1 phone is (67) 99812-3456, has order 4
    res_phone = get_order_status(customer_identifier="67998123456", db=db)
    assert res_phone.found is True
    assert res_phone.orders[0].order_id == 4

    # Lookup by name: Leticia Gonçalves Rocha (customer 14) has order 7
    res_name = get_order_status(customer_identifier="Leticia Goncalves Rocha", db=db)
    assert res_name.found is True
    assert res_name.orders[0].order_id == 7

    # Duplicate email: bruno.martins@jmail.com (customers 11 & 43) -> order 10 belongs to 11
    res_dup = get_order_status(customer_identifier="bruno.martins@jmail.com", db=db)
    assert res_dup.found is True
    assert any(o.order_id == 10 for o in res_dup.orders)


def test_get_order_status_lgpd_and_cpf_and_sqli() -> None:
    """Verify LGPD customer cross-check, CPF guidance, and SQL injection safety."""
    db = get_duckdb_manager()

    # LGPD mismatch: order 1 belongs to customer 3, not customer 1 (lucas.mendes@jmail.com)
    mismatch = get_order_status(order_id=1, customer_identifier="lucas.mendes@jmail.com", db=db)
    assert mismatch.found is False
    assert "não coincidem com o titular" in mismatch.message

    # LGPD match: order 1 with matching email passes
    match = get_order_status(order_id=1, customer_identifier="pedro.oliveira@jmail.com", db=db)
    assert match.found is True

    # CPF input guidance
    res_cpf = get_order_status(customer_identifier="123.456.789-00", db=db)
    assert res_cpf.found is False
    assert "não armazena CPF" in res_cpf.message

    # SQL Injection attempt safety
    sqli_attempt = get_order_status(
        customer_identifier="1' OR '1'='1; DROP TABLE orders; --", db=db
    )
    assert sqli_attempt.found is False
    # Ensure orders table was not dropped!
    assert db.table_exists("orders")
    assert db.get_table_count("orders") == 20

    # No identifier provided
    res_none = get_order_status(db=db)
    assert res_none.found is False
    assert "informe o número do pedido" in res_none.message


# ============================================================================
# LangChain StructuredTools Parity Tests
# ============================================================================


def test_structured_tools_langchain_parity() -> None:
    """Verify create_catalog_tools and create_order_tools return callable StructuredTools (JSON)."""
    db = get_duckdb_manager()

    catalog_tools = create_catalog_tools(db=db)
    assert len(catalog_tools) == 3
    tool_names = {t.name for t in catalog_tools}
    assert tool_names == {"search_products", "get_product_details", "get_active_promotions"}

    # Invoke search_products via StructuredTool
    search_tool = next(t for t in catalog_tools if t.name == "search_products")
    raw_res = search_tool.invoke({"category": "Ukuleles", "max_price": 500.0})
    parsed = json.loads(raw_res)
    assert "total_found" in parsed
    assert parsed["total_found"] > 0

    order_tools = create_order_tools(db=db)
    assert len(order_tools) == 1
    order_tool = order_tools[0]
    assert order_tool.name == "get_order_status"

    # Invoke get_order_status via StructuredTool
    order_res = order_tool.invoke({"order_id": 1})
    parsed_order = json.loads(order_res)
    assert parsed_order["found"] is True
    assert parsed_order["orders"][0]["order_id"] == 1
