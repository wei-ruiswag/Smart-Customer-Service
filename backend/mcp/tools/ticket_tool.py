# backend/mcp/tools/ticket_tool.py

from __future__ import annotations

import asyncio
from typing import Any

from services.ticket_service import (
    create_ticket as create_ticket_db,
    query_tickets as query_tickets_db,
    update_ticket_status as update_ticket_status_db,
)


async def ticket_create(
    user_id: str,
    order_no: str,
    ticket_type: str,
    priority: str = "中",
    description: str = "",
) -> dict[str, Any]:
    """
    MCP 工具：创建工单。
    """
    ticket = await asyncio.to_thread(
        create_ticket_db,
        user_id=user_id,
        order_no=order_no,
        ticket_type=ticket_type,
        priority=priority,
        description=description,
    )

    return {
        "found": True,
        "message": "工单创建成功。",
        "ticket": ticket,
    }


async def ticket_query(
    ticket_no: str = "",
    user_id: str = "",
    order_no: str = "",
    limit: int = 5,
) -> dict[str, Any]:
    """
    MCP 工具：查询工单。
    """
    tickets = await asyncio.to_thread(
        query_tickets_db,
        ticket_no=ticket_no,
        user_id=user_id,
        order_no=order_no,
        limit=limit,
    )

    if not tickets:
        return {
            "found": False,
            "message": "未查询到符合条件的工单。",
            "tickets": [],
        }

    return {
        "found": True,
        "message": f"共查询到 {len(tickets)} 条工单。",
        "tickets": tickets,
    }


async def ticket_update(
    ticket_no: str,
    status: str,
) -> dict[str, Any]:
    """
    MCP 工具：更新工单状态。
    """
    ticket = await asyncio.to_thread(
        update_ticket_status_db,
        ticket_no=ticket_no,
        status=status,
    )

    if not ticket:
        return {
            "found": False,
            "message": f"未找到工单号 {ticket_no}。",
            "ticket": None,
        }

    return {
        "found": True,
        "message": "工单状态已更新。",
        "ticket": ticket,
    }