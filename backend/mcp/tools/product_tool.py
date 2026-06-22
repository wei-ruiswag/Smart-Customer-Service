# backend/mcp/tools/product_tool.py

import asyncio
from typing import Any

from services.product_service import search_products


def _to_float(value):
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value, default: int = 5):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


async def product_query(
    keyword: str = "",
    category: str = "",
    max_price: float | str | None = None,
    min_price: float | str | None = None,
    only_in_stock: bool = True,
    limit: int = 5,
) -> dict[str, Any]:
    """
    MCP 商品查询工具。
    负责接收工具参数，调用 product_service 查询 MySQL。
    """

    max_price_value = _to_float(max_price)
    min_price_value = _to_float(min_price)
    limit_value = _to_int(limit, default=5)

    products = await asyncio.to_thread(
        search_products,
        keyword=keyword or "",
        category=category or "",
        max_price=max_price_value,
        min_price=min_price_value,
        only_in_stock=only_in_stock,
        limit=limit_value,
    )

    if not products:
        return {
            "found": False,
            "message": "暂未查询到符合条件的商品。",
            "products": [],
        }

    return {
        "found": True,
        "message": f"共查询到 {len(products)} 个符合条件的商品。",
        "products": products,
    }