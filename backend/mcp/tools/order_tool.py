# backend/mcp/tools/order_tool.py
from __future__ import annotations

import asyncio
from typing import Any

from services.order_service import get_order_by_no, list_recent_orders


def _to_int(value: Any, default: int = 5) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


async def order_query(
    order_no: str = "",
    user_id: str = "",
    limit: int = 5,
) -> dict[str, Any]:
    """
    MCP 订单查询工具。
    - 有 order_no：查询当前用户指定订单。
    - 无 order_no：查询当前用户最近订单。
    """
    if not user_id:
        return {
            "found": False,
            "message": "缺少 user_id，无法查询订单。",
            "order": None,
            "orders": [],
        }

    if order_no:
        order = await asyncio.to_thread(get_order_by_no, order_no, user_id)
        if not order:
            return {
                "found": False,
                "message": f"订单 {order_no} 不存在或不属于当前用户。",
                "order": None,
                "orders": [],
            }

        return {
            "found": True,
            "message": "订单查询成功。",
            "order": order,
            "orders": [],
        }

    limit_value = _to_int(limit, default=5)
    orders = await asyncio.to_thread(list_recent_orders, user_id, limit_value)

    if not orders:
        return {
            "found": False,
            "message": "暂未查询到当前用户的订单。",
            "order": None,
            "orders": [],
        }

    return {
        "found": True,
        "message": f"共查询到 {len(orders)} 条最近订单。",
        "order": None,
        "orders": orders,
    }