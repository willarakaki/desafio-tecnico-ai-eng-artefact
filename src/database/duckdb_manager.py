"""DuckDB in-memory database manager with CSV ingestion and text normalization."""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb

from src.config import get_settings

logger = logging.getLogger(__name__)

# Pattern for safe SQL table/identifier names
SAFE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
CSV_PREFIX_PATTERN = re.compile(r"^desafio_tecnico_ai_eng\s*-\s*", re.IGNORECASE)


class DuckDBManager:
    """In-memory DuckDB manager for operational musical store data.

    Ingests operational CSV files from data/ into typed relational tables,
    registers SQL normalization macros, and provides thread-safe parameterized query execution.
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        """Initialize in-memory DuckDB connection and ingest operational CSVs.

        Args:
            data_dir: Path to directory containing operational CSV files.
                     Defaults to settings.data_dir.
        """
        self._lock = threading.RLock()
        self._data_dir = self._resolve_data_dir(data_dir)
        self._connection = duckdb.connect(":memory:")
        self._init_macros()
        self._ingest_csv_files()

    @staticmethod
    def _resolve_data_dir(data_dir: Path | str | None) -> Path:
        """Resolve valid data directory path."""
        if data_dir is not None:
            resolved = Path(data_dir).resolve()
            if not resolved.is_dir():
                raise FileNotFoundError(f"Specified data_dir does not exist: {resolved}")
            return resolved

        settings_path = get_settings().data_dir
        candidate = Path(settings_path)
        if candidate.is_dir():
            return candidate.resolve()

        # Fallback to project root / data
        project_root = Path(__file__).resolve().parent.parent.parent
        root_data = project_root / "data"
        if root_data.is_dir():
            return root_data.resolve()

        raise FileNotFoundError(
            f"Operational data directory not found at '{candidate}' or '{root_data}'"
        )

    def _init_macros(self) -> None:
        """Register text normalization macros in DuckDB."""
        with self._lock:
            # Macro to strip accents, lowercase, trim, and collapse multiple whitespace
            self._connection.execute(
                """
                CREATE OR REPLACE MACRO normalize_pt(val) AS
                trim(regexp_replace(
                    lower(strip_accents(coalesce(CAST(val AS VARCHAR), ''))),
                    '\\s+', ' ', 'g'
                ))
                """
            )

    def _ingest_csv_files(self) -> None:
        """Discover and ingest all CSV files into typed tables."""
        with self._lock:
            csv_files = sorted(self._data_dir.glob("*.csv"))
            if not csv_files:
                logger.warning("No CSV files found in %s", self._data_dir)
                return

            for file_path in csv_files:
                raw_stem = file_path.stem
                clean_name = CSV_PREFIX_PATTERN.sub("", raw_stem).strip()
                if not SAFE_IDENTIFIER_PATTERN.match(clean_name):
                    logger.warning("Skipping file with unsafe table name: %s", raw_stem)
                    continue

                self._load_table(clean_name, file_path)

    def _load_table(self, table_name: str, file_path: Path) -> None:
        """Load a CSV file into a typed DuckDB table."""
        str_path = str(file_path.resolve())

        # Exact schema casting and whitespace trimming for expected tables
        if table_name == "categories":
            self._connection.execute(
                """
                CREATE TABLE categories AS
                SELECT
                    CAST(category_id AS INTEGER) AS category_id,
                    TRIM(name) AS name,
                    TRIM(description) AS description
                FROM read_csv_auto(?, header=true, encoding='utf-8')
                """,
                [str_path],
            )
        elif table_name == "products":
            self._connection.execute(
                """
                CREATE TABLE products AS
                SELECT
                    CAST(product_id AS INTEGER) AS product_id,
                    CAST(price_brl AS DOUBLE) AS price_brl,
                    TRIM(name) AS name,
                    CAST(category_id AS INTEGER) AS category_id,
                    TRIM(description) AS description,
                    CAST(stock_quantity AS INTEGER) AS stock_quantity,
                    TRIM(status) AS status,
                    CAST(specs AS JSON) AS specs,
                    CAST(created_at AS DATE) AS created_at
                FROM read_csv_auto(?, header=true, encoding='utf-8')
                """,
                [str_path],
            )
        elif table_name == "promotions":
            self._connection.execute(
                """
                CREATE TABLE promotions AS
                SELECT
                    CAST(promotion_id AS INTEGER) AS promotion_id,
                    CAST(product_id AS INTEGER) AS product_id,
                    CAST(discount_percent AS INTEGER) AS discount_percent,
                    TRIM(description) AS description,
                    CAST(is_active AS INTEGER) AS is_active
                FROM read_csv_auto(?, header=true, encoding='utf-8')
                """,
                [str_path],
            )
        elif table_name == "customers":
            self._connection.execute(
                """
                CREATE TABLE customers AS
                SELECT
                    CAST(customer_id AS INTEGER) AS customer_id,
                    TRIM(name) AS name,
                    TRIM(phone) AS phone,
                    TRIM(email) AS email,
                    TRIM(city) AS city
                FROM read_csv_auto(?, header=true, encoding='utf-8')
                """,
                [str_path],
            )
        elif table_name == "orders":
            self._connection.execute(
                """
                CREATE TABLE orders AS
                SELECT
                    CAST(order_id AS INTEGER) AS order_id,
                    CAST(customer_id AS INTEGER) AS customer_id,
                    CAST(order_date AS DATE) AS order_date,
                    TRIM(status) AS status,
                    CAST(total_brl AS DOUBLE) AS total_brl,
                    TRIM(payment_method) AS payment_method,
                    NULLIF(TRIM(tracking_code), '') AS tracking_code,
                    CAST(estimated_delivery AS DATE) AS estimated_delivery,
                    NULLIF(TRIM(notes), '') AS notes
                FROM read_csv_auto(?, header=true, encoding='utf-8')
                """,
                [str_path],
            )
        elif table_name == "order_items":
            self._connection.execute(
                """
                CREATE TABLE order_items AS
                SELECT
                    CAST(order_id AS INTEGER) AS order_id,
                    CAST(quantity AS INTEGER) AS quantity,
                    CAST(product_id AS INTEGER) AS product_id
                FROM read_csv_auto(?, header=true, encoding='utf-8')
                """,
                [str_path],
            )
        else:
            # Fallback for custom or test tables
            query = (
                f"CREATE TABLE {table_name} AS "
                "SELECT * FROM read_csv_auto(?, header=true, encoding='utf-8')"
            )
            self._connection.execute(query, [str_path])

    @classmethod
    def normalize_text(cls, text: str | None) -> str:
        """Normalize Portuguese string by removing diacritics, lowercasing, and collapsing spaces.

        Args:
            text: Input string or None.

        Returns:
            Normalized lowercase ASCII-like string without accents.
        """
        if not text:
            return ""
        # Decompose unicode characters (NFKD) and strip non-spacing marks (Mn)
        nfkd = unicodedata.normalize("NFKD", str(text))
        without_accents = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
        # Lowercase, strip and collapse multiple whitespace
        return re.sub(r"\s+", " ", without_accents).strip().lower()

    def execute_query(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a parameterized SQL query thread-safely and return rows as dictionaries.

        Args:
            sql: Parameterized SQL string with '?' placeholders.
            params: Sequence of parameter values to bind.

        Returns:
            List of dictionaries mapping column name to value.
        """
        with self._lock:
            cur = self._connection.cursor()
            if params is not None:
                cur.execute(sql, list(params))
            else:
                cur.execute(sql)

            if not cur.description:
                return []

            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return [dict(zip(columns, row, strict=False)) for row in rows]

    def get_table_names(self) -> list[str]:
        """Return list of existing tables in DuckDB."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
            return sorted(r[0] for r in rows)

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        return table_name in self.get_table_names()

    def get_table_count(self, table_name: str) -> int:
        """Return total row count for given table."""
        if not SAFE_IDENTIFIER_PATTERN.match(table_name):
            raise ValueError(f"Invalid table name: {table_name}")
        with self._lock:
            res = self._connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            return res[0] if res else 0

    def close(self) -> None:
        """Close DuckDB connection."""
        with self._lock:
            self._connection.close()


_duckdb_manager_instance: DuckDBManager | None = None
_manager_lock = threading.RLock()


def get_duckdb_manager(data_dir: Path | str | None = None) -> DuckDBManager:
    """Return singleton instance of DuckDBManager or create new if custom data_dir provided.

    Args:
        data_dir: Optional custom data directory.

    Returns:
        DuckDBManager instance.
    """
    global _duckdb_manager_instance
    with _manager_lock:
        if data_dir is not None:
            return DuckDBManager(data_dir=data_dir)
        if _duckdb_manager_instance is None:
            _duckdb_manager_instance = DuckDBManager()
        return _duckdb_manager_instance


def reset_duckdb_manager() -> None:
    """Reset the global DuckDBManager singleton."""
    global _duckdb_manager_instance
    with _manager_lock:
        if _duckdb_manager_instance is not None:
            _duckdb_manager_instance.close()
            _duckdb_manager_instance = None
