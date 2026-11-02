"""Task 0 smoke tests validating runtime setup and ground-truth dataset schemas."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import src

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

EXPECTED_CSV_SCHEMAS: dict[str, tuple[list[str], int]] = {
    "desafio_tecnico_ai_eng - categories.csv": (
        ["category_id", "name", "description"],
        9,
    ),
    "desafio_tecnico_ai_eng - customers.csv": (
        ["customer_id", "name", "phone", "email", "city"],
        50,
    ),
    "desafio_tecnico_ai_eng - orders.csv": (
        [
            "order_id",
            "customer_id",
            "order_date",
            "status",
            "total_brl",
            "payment_method",
            "tracking_code",
            "estimated_delivery",
            "notes",
        ],
        20,
    ),
    "desafio_tecnico_ai_eng - order_items.csv": (
        ["order_id", "quantity", "product_id"],
        22,
    ),
    "desafio_tecnico_ai_eng - products.csv": (
        [
            "product_id",
            "price_brl",
            "name",
            "category_id",
            "description",
            "stock_quantity",
            "status",
            "specs",
            "created_at",
        ],
        65,
    ),
    "desafio_tecnico_ai_eng - promotions.csv": (
        ["promotion_id", "product_id", "discount_percent", "description", "is_active"],
        25,
    ),
}


def test_python_runtime_and_package_import() -> None:
    """Ensure Python 3.11+ runtime and clean import of the root `src` package."""
    assert sys.version_info >= (3, 11)
    assert hasattr(src, "__all__")


@pytest.mark.parametrize(
    ("filename", "expected_columns", "expected_row_count"),
    [(name, cols, count) for name, (cols, count) in EXPECTED_CSV_SCHEMAS.items()],
)
def test_operational_csv_schemas_and_row_counts(
    filename: str,
    expected_columns: list[str],
    expected_row_count: int,
) -> None:
    """Verify every operational CSV exists with exact expected headers and row count."""
    csv_path = DATA_DIR / filename
    assert csv_path.is_file(), f"Missing dataset file: {csv_path}"

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == expected_columns
        rows = list(reader)
        assert len(rows) == expected_row_count


def test_store_policies_pdf_exists_and_is_valid() -> None:
    """Verify the store policies PDF exists and starts with a valid PDF header."""
    pdf_path = DATA_DIR / "políticas_da_loja.pdf"
    assert pdf_path.is_file(), f"Missing policy PDF: {pdf_path}"
    assert pdf_path.stat().st_size > 10_000
    with pdf_path.open("rb") as handle:
        assert handle.read(5) == b"%PDF-"
