"""Database management package for Empório da Música."""

from src.database.duckdb_manager import DuckDBManager, get_duckdb_manager, reset_duckdb_manager

__all__: list[str] = [
    "DuckDBManager",
    "get_duckdb_manager",
    "reset_duckdb_manager",
]
