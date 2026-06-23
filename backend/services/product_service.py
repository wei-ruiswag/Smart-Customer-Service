# backend/services/product_service.py
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from utils.db import mysql_cursor


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value


def _json_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_safe(value) for key, value in row.items()}

def search_products(
    keyword: str | None = None,
    category: str | None = None,
    max_price: float | None = None,
    min_price: float | None = None,
    only_in_stock: bool = True,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    查询商品信息。
    适合处理：商品名称、分类、价格、库存、描述、商品售后字段等结构化查询。
    """

    sql = """
        SELECT
            id,
            product_name,
            category,
            price,
            stock,
            description,
            after_sale_policy
        FROM products
        WHERE 1 = 1
    """   # brand,

    params: list[Any] = []

    if keyword:
        sql += """
            AND (
                product_name LIKE %s
                OR category LIKE %s       
                OR description LIKE %s
            )
        """             # OR brand LIKE %s
        like_kw = f"%{keyword}%"
        params.extend([like_kw, like_kw, like_kw])

    if category:
        sql += " AND category LIKE %s"
        params.append(f"%{category}%")

    if min_price is not None:
        sql += " AND price >= %s"
        params.append(min_price)

    if max_price is not None:
        sql += " AND price <= %s"
        params.append(max_price)

    if only_in_stock:
        sql += " AND stock > 0"

    sql += " ORDER BY price ASC LIMIT %s"
    params.append(limit)

    with mysql_cursor() as cursor:
        cursor.execute(sql, params)
        rows = cursor.fetchall()

    return [_json_safe_row(row) for row in rows]