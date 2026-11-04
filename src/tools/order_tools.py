"""Deterministic order status and customer lookup tools for Empório da Música."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from src.database.duckdb_manager import DuckDBManager, get_duckdb_manager

logger = logging.getLogger(__name__)

# Status translations for customer communication
STATUS_TRANSLATIONS: dict[str, str] = {
    "delivered": "Entregue",
    "shipped": "Enviado / Em transporte",
    "confirmed": "Confirmado / Em separação",
    "pending": "Pendente de pagamento",
    "cancelled": "Cancelado",
}

# Regex to detect Brazilian CPF format with standard punctuation (e.g. 123.456.789-00)
CPF_FORMATTED_REGEX = re.compile(r"^\d{3}\.\d{3}\.\d{3}-\d{2}$")


def _is_valid_cpf_digits(digits_str: str) -> bool:
    """Validate 11-digit string using the official Brazilian CPF modulo 11 checksum algorithm."""
    if len(digits_str) != 11 or not digits_str.isdigit():
        return False
    if len(set(digits_str)) == 1:
        return False  # Known invalid CPFs like 111.111.111-11

    d = [int(x) for x in digits_str]
    s1 = sum(d[i] * (10 - i) for i in range(9))
    d1 = (s1 * 10 % 11) % 10
    if d[9] != d1:
        return False

    s2 = sum(d[i] * (11 - i) for i in range(10))
    d2 = (s2 * 10 % 11) % 10
    return d[10] == d2


class GetOrderStatusInput(BaseModel):
    """Input parameters for querying order status."""

    model_config = ConfigDict(extra="forbid")

    order_id: int | str | None = Field(
        default=None,
        description="Número identificador do pedido (ex: 1, '#1', 'Pedido 4').",
    )
    customer_identifier: str | None = Field(
        default=None,
        description="Identificador: e-mail cadastrado, telefone com ou sem DDD, ou nome.",
    )


class OrderItemDetail(BaseModel):
    """Line item within an order."""

    product_id: int
    product_name: str
    quantity: int
    unit_price_brl: float
    total_price_brl: float


class OrderDetail(BaseModel):
    """Full detail of a customer order."""

    order_id: int
    customer_id: int
    customer_name: str
    customer_city: str
    order_date: str
    status: str
    status_label: str
    total_brl: float
    payment_method: str
    tracking_code: str | None = None
    estimated_delivery: str | None = None
    notes: str | None = None
    items: list[OrderItemDetail] = Field(default_factory=list)


class OrderStatusResult(BaseModel):
    """Result of order status lookup."""

    found: bool
    orders: list[OrderDetail] = Field(default_factory=list)
    message: str


# ============================================================================
# Helper Functions
# ============================================================================


def _extract_order_id(raw_order_id: int | str | None) -> int | None:
    """Extract numeric order_id safely from integer or string."""
    if raw_order_id is None:
        return None
    if isinstance(raw_order_id, int):
        return raw_order_id if raw_order_id > 0 else None

    # Handle string cases like "#4", "Pedido 4", " 4 "
    matches = re.findall(r"\d+", str(raw_order_id))
    if matches:
        return int(matches[0])
    return None


def _find_customers(
    customer_identifier: str,
    db: DuckDBManager,
) -> list[dict[str, Any]]:
    """Find matching customers by email, phone, name, or customer_id using parameterized SQL."""
    raw = customer_identifier.strip()
    digits = re.sub(r"\D", "", raw)

    # 1. Check for email
    if "@" in raw:
        return db.execute_query(
            "SELECT customer_id, name, city, email, phone FROM customers WHERE lower(email) = ?",
            [raw.lower()],
        )

    # 2. Check for phone digits (e.g., 67998123456 or 998123456)
    if digits and len(digits) >= 8:
        # Match against digits-normalized phone column in DuckDB
        return db.execute_query(
            """
            SELECT customer_id, name, city, email, phone
            FROM customers
            WHERE regexp_replace(phone, '[^0-9]', '', 'g') LIKE ?
            """,
            [f"%{digits}"],
        )

    # 3. Check for exact customer_id if numeric
    if raw.isdigit():
        num_id = int(raw)
        rows = db.execute_query(
            "SELECT customer_id, name, city, email, phone FROM customers WHERE customer_id = ?",
            [num_id],
        )
        if rows:
            return rows

    # 4. Check for name match (accent and case-insensitive)
    norm_name = DuckDBManager.normalize_text(raw)
    return db.execute_query(
        """
        SELECT customer_id, name, city, email, phone
        FROM customers
        WHERE normalize_pt(name) = ? OR normalize_pt(name) LIKE ?
        """,
        [norm_name, f"%{norm_name}%"],
    )


# ============================================================================
# Deterministic Tool Implementations
# ============================================================================


def get_order_status(
    order_id: int | str | None = None,
    customer_identifier: str | None = None,
    *,
    db: DuckDBManager | None = None,
) -> OrderStatusResult:
    """Retrieve order status, delivery tracking, and line items with LGPD privacy validation.

    Args:
        order_id: Optional order number (e.g. 1, '#1', 'Pedido 4').
        customer_identifier: Optional customer identifier (email, phone, name, or customer_id).
        db: Optional DuckDBManager instance (defaults to singleton).

    Returns:
        OrderStatusResult with list of orders, line items, and status summary.
    """
    try:
        active_db = db or get_duckdb_manager()

        # Validate that at least one identifier is provided
        if order_id is None and not customer_identifier:
            return OrderStatusResult(
                found=False,
                orders=[],
                message=(
                    "Para consultar o status do pedido, por favor informe o número do pedido "
                    "(ex: #1) ou um dado cadastral (e-mail, telefone ou nome completo)."
                ),
            )

        # Check for explicitly formatted CPF inputs (e.g. 123.456.789-00)
        if (customer_identifier and CPF_FORMATTED_REGEX.match(customer_identifier.strip())) or (
            order_id is not None
            and isinstance(order_id, str)
            and CPF_FORMATTED_REGEX.match(order_id.strip())
        ):
            return OrderStatusResult(
                found=False,
                orders=[],
                message=(
                    "Identificamos que você informou um CPF. O sistema da Empório da Música "
                    "não armazena CPF nos pedidos. Por favor, informe o número do pedido, "
                    "e-mail cadastrado ou telefone para localizarmos sua compra."
                ),
            )

        parsed_order_id = _extract_order_id(order_id)

        # Retrieve matched customers if identifier provided
        matched_customers: list[dict[str, Any]] = []
        if customer_identifier:
            matched_customers = _find_customers(customer_identifier, active_db)

        # If customer_identifier was provided but no customer found
        if customer_identifier and not matched_customers:
            # Check if input was an unformatted 11-digit CPF
            clean_digits = re.sub(r"\D", "", customer_identifier)
            if _is_valid_cpf_digits(clean_digits):
                return OrderStatusResult(
                    found=False,
                    orders=[],
                    message=(
                        "Identificamos que você informou um CPF. O sistema da Empório da Música "
                        "não armazena CPF nos pedidos. Por favor, informe o número do pedido, "
                        "e-mail cadastrado ou telefone para localizarmos sua compra."
                    ),
                )

            return OrderStatusResult(
                found=False,
                orders=[],
                message=(
                    f"Nenhum cliente cadastrado foi localizado com '{customer_identifier}'. "
                    "Por favor, verifique o e-mail ou telefone informado."
                ),
            )

        matched_cust_ids = [int(c["customer_id"]) for c in matched_customers]

        # Scenario 1: Both order_id and customer_identifier are provided -> LGPD Cross-Check
        if parsed_order_id is not None and customer_identifier:
            order_rows = active_db.execute_query(
                """
                SELECT
                    o.order_id,
                    o.customer_id,
                    c.name AS customer_name,
                    c.city AS customer_city,
                    o.order_date,
                    o.status,
                    o.total_brl,
                    o.payment_method,
                    o.tracking_code,
                    o.estimated_delivery,
                    o.notes
                FROM orders o
                JOIN customers c ON o.customer_id = c.customer_id
                WHERE o.order_id = ?
                """,
                [parsed_order_id],
            )

            if not order_rows:
                return OrderStatusResult(
                    found=False,
                    orders=[],
                    message=f"Pedido #{parsed_order_id} não foi encontrado no sistema.",
                )

            order_data = order_rows[0]
            actual_cust_id = int(order_data["customer_id"])

            # Verify that the order actually belongs to this customer
            if actual_cust_id not in matched_cust_ids:
                logger.warning(
                    "LGPD mismatch: order %s customer %s does not match requested %s",
                    parsed_order_id,
                    actual_cust_id,
                    matched_cust_ids,
                )
                return OrderStatusResult(
                    found=False,
                    orders=[],
                    message=(
                        f"Os dados não coincidem com o titular do pedido #{parsed_order_id}. "
                        "Por conformidade com a LGPD e segurança, verifique os dados informados."
                    ),
                )

            target_orders = order_rows

        # Scenario 2: Only order_id is provided
        elif parsed_order_id is not None:
            target_orders = active_db.execute_query(
                """
                SELECT
                    o.order_id,
                    o.customer_id,
                    c.name AS customer_name,
                    c.city AS customer_city,
                    o.order_date,
                    o.status,
                    o.total_brl,
                    o.payment_method,
                    o.tracking_code,
                    o.estimated_delivery,
                    o.notes
                FROM orders o
                JOIN customers c ON o.customer_id = c.customer_id
                WHERE o.order_id = ?
                """,
                [parsed_order_id],
            )
            if not target_orders:
                return OrderStatusResult(
                    found=False,
                    orders=[],
                    message=f"Pedido #{parsed_order_id} não foi encontrado no sistema.",
                )

        # Scenario 3: Only customer_identifier is provided
        else:
            placeholders = ",".join(["?"] * len(matched_cust_ids))
            target_orders = active_db.execute_query(
                f"""
                SELECT
                    o.order_id,
                    o.customer_id,
                    c.name AS customer_name,
                    c.city AS customer_city,
                    o.order_date,
                    o.status,
                    o.total_brl,
                    o.payment_method,
                    o.tracking_code,
                    o.estimated_delivery,
                    o.notes
                FROM orders o
                JOIN customers c ON o.customer_id = c.customer_id
                WHERE o.customer_id IN ({placeholders})
                ORDER BY o.order_date DESC, o.order_id DESC
                """,
                matched_cust_ids,
            )
            if not target_orders:
                cust_name = matched_customers[0]["name"]
                return OrderStatusResult(
                    found=False,
                    orders=[],
                    message=f"O cliente {cust_name} não possui pedidos registrados no momento.",
                )

        # Build order objects with line items
        orders: list[OrderDetail] = []
        for o_row in target_orders:
            curr_oid = int(o_row["order_id"])

            item_rows = active_db.execute_query(
                """
                SELECT
                    oi.product_id,
                    p.name AS product_name,
                    oi.quantity,
                    p.price_brl AS unit_price_brl,
                    (oi.quantity * p.price_brl) AS total_price_brl
                FROM order_items oi
                JOIN products p ON oi.product_id = p.product_id
                WHERE oi.order_id = ?
                ORDER BY oi.product_id ASC
                """,
                [curr_oid],
            )

            items = [
                OrderItemDetail(
                    product_id=int(item["product_id"]),
                    product_name=str(item["product_name"]),
                    quantity=int(item["quantity"]),
                    unit_price_brl=float(item["unit_price_brl"]),
                    total_price_brl=float(item["total_price_brl"]),
                )
                for item in item_rows
            ]

            raw_status = str(o_row["status"])
            status_label = STATUS_TRANSLATIONS.get(raw_status, raw_status.title())

            orders.append(
                OrderDetail(
                    order_id=curr_oid,
                    customer_id=int(o_row["customer_id"]),
                    customer_name=str(o_row["customer_name"]),
                    customer_city=str(o_row["customer_city"]),
                    order_date=str(o_row["order_date"]),
                    status=raw_status,
                    status_label=status_label,
                    total_brl=float(o_row["total_brl"]),
                    payment_method=str(o_row["payment_method"]),
                    tracking_code=(
                        str(o_row["tracking_code"]) if o_row["tracking_code"] is not None else None
                    ),
                    estimated_delivery=(
                        str(o_row["estimated_delivery"])
                        if o_row["estimated_delivery"] is not None
                        else None
                    ),
                    notes=str(o_row["notes"]) if o_row["notes"] is not None else None,
                    items=items,
                )
            )

        # Craft human-readable status message for the agent
        first_order = orders[0]
        if len(orders) == 1:
            if not customer_identifier:
                # Strict LGPD / Anti-IDOR barrier when querying without customer verification
                msg_parts = [
                    f"Pedido #{first_order.order_id} em {first_order.order_date}: "
                    f"Status '{first_order.status_label}'. "
                    f"[AVISO DE SEGURANÇA E PRIVACIDADE LGPD: O usuário consultou apenas o número do pedido "
                    f"sem fornecer e-mail ou telefone cadastrado para autenticação. NÃO divulgue o nome completo "
                    f"do titular ({first_order.customer_name}), cidade ({first_order.customer_city}) ou código "
                    f"de rastreamento ({first_order.tracking_code or 'N/A'}) a terceiros. Solicite educadamente "
                    f"que o cliente confirme o e-mail ou o telefone cadastrado no pedido antes de revelar esses detalhes.]"
                ]
            else:
                msg_parts = [
                    f"Pedido #{first_order.order_id} ({first_order.customer_name}, "
                    f"{first_order.customer_city}) em {first_order.order_date}: "
                    f"Status '{first_order.status_label}', total R$ {first_order.total_brl:.2f} "
                    f"via {first_order.payment_method}."
                ]
                if first_order.tracking_code:
                    msg_parts.append(f"Código de rastreamento: {first_order.tracking_code}.")
                if first_order.estimated_delivery:
                    msg_parts.append(
                        f"Previsão estimada de entrega: {first_order.estimated_delivery}."
                    )
                if first_order.notes:
                    msg_parts.append(f"Observações: '{first_order.notes}'.")

                if first_order.items:
                    item_descs = [f"{it.quantity}x {it.product_name}" for it in first_order.items]
                    msg_parts.append(f"Itens do pedido: {', '.join(item_descs)}.")

            message = " ".join(msg_parts)
        else:
            message = (
                f"Localizados {len(orders)} pedidos para {first_order.customer_name}. "
                f"Mais recente: Pedido #{first_order.order_id} ({first_order.status_label})."
            )

        return OrderStatusResult(
            found=True,
            orders=orders,
            message=message,
        )

    except Exception as exc:
        logger.exception("Error executing get_order_status: %s", exc)
        return OrderStatusResult(
            found=False,
            orders=[],
            message="Ocorreu um erro ao consultar o status do pedido. Por favor, tente novamente.",
        )


# ============================================================================
# LangChain StructuredTool Factory
# ============================================================================


def create_order_tools(db: DuckDBManager | None = None) -> list[StructuredTool]:
    """Create LangChain StructuredTool instances for order tools.

    Args:
        db: Optional DuckDBManager instance (defaults to singleton).

    Returns:
        List of StructuredTool instances ready for LangGraph agent binding.
    """
    active_db = db or get_duckdb_manager()

    def _get_order_status_tool(
        order_id: int | str | None = None,
        customer_identifier: str | None = None,
    ) -> str:
        res = get_order_status(
            order_id=order_id,
            customer_identifier=customer_identifier,
            db=active_db,
        )
        return res.model_dump_json(indent=2)

    return [
        StructuredTool.from_function(
            func=_get_order_status_tool,
            name="get_order_status",
            description=(
                "Consulta o status em tempo real de um pedido na Empório da Música, "
                "incluindo código de rastreamento, previsão de entrega, itens comprados "
                "e notas de cancelamento (se houver), garantindo privacidade e LGPD."
            ),
            args_schema=GetOrderStatusInput,
        ),
    ]
