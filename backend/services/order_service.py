# backend/services/order_service.py
from __future__ import annotations

import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pymysql
from dotenv import load_dotenv

from utils.db import mysql_cursor





def _mask_phone(phone: str | None) -> str:
    if not phone:
        return ""
    phone = str(phone)
    if len(phone) < 7:
        return phone[:2] + "***"
    return phone[:3] + "****" + phone[-4:]


def _mask_addr(addr: str | None) -> str:
    if not addr:
        return ""
    addr = str(addr)
    if len(addr) <= 8:
        return addr[:2] + "***"
    return addr[:6] + "***" + addr[-3:]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value


def _build_refund_window(signed_at: Any) -> dict[str, Any]:
    """
    七天无理由起算窗口。
    本项目定义：signed_at 是签收时间，七天无理由以 signed_at 为准。
    """
    result = {
        "basis": "signed_at",
        "signed_at": None,
        "deadline": None,
        "in_window": None,
        "message": "订单暂无签收时间，无法计算七天无理由截止时间。",
    }

    if not isinstance(signed_at, datetime):
        return result

    deadline = signed_at + timedelta(days=7)
    now = datetime.now()

    result.update(
        {
            "signed_at": signed_at.strftime("%Y-%m-%d %H:%M:%S"),
            "deadline": deadline.strftime("%Y-%m-%d %H:%M:%S"),
            "in_window": now <= deadline,
            "message": (
                "当前仍在七天无理由申请期内。"
                if now <= deadline
                else "当前已超过七天无理由申请期。"
            ),
        }
    )

    return result


def _enrich_order(row: dict[str, Any]) -> dict[str, Any]:
    """
    对订单结果做：
    1. JSON 安全转换；
    2. 手机号、地址脱敏；
    3. 补充七天无理由时间窗口。
    """
    signed_at = row.get("signed_at")

    safe_row = {key: _json_safe(value) for key, value in row.items()}
    safe_row["receiver_phone"] = _mask_phone(row.get("receiver_phone"))
    safe_row["receiver_addr"] = _mask_addr(row.get("receiver_addr"))
    safe_row["seven_day_refund"] = _build_refund_window(signed_at)

    return safe_row


def get_order_by_no(order_no: str, user_id: str) -> dict[str, Any] | None:
    """
    按订单号查询当前用户订单。
    必须同时使用 order_no + user_id，避免泄露其他用户订单。
    """
    sql = """
    SELECT
        o.id,
        o.order_no,
        o.user_id,
        o.product_id,
        o.amount,
        o.status,
        o.payment_status,
        o.logistics_status,
        o.receiver_name,
        o.receiver_phone,
        o.receiver_addr,
        o.created_at,
        o.updated_at,
        o.tracking_no,
        o.signed_at,
        o.logistics_company,

        p.product_name,
        p.category AS product_category,
        p.price AS product_price,
        p.stock AS product_stock,
        p.description AS product_description,
        p.after_sale_policy AS product_after_sale_policy
    FROM orders o
    LEFT JOIN products p ON o.product_id = p.id
    WHERE o.order_no = %s AND o.user_id = %s
    LIMIT 1
    """

    with mysql_cursor() as cursor:
        cursor.execute(sql, (order_no, user_id))
        row = cursor.fetchone()

    return _enrich_order(row) if row else None


def list_recent_orders(user_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """
    查询当前用户最近订单。
    用户只说“我的订单”但没给订单号时使用。
    """
    sql = """
    SELECT
        o.id,
        o.order_no,
        o.user_id,
        o.product_id,
        o.amount,
        o.status,
        o.payment_status,
        o.logistics_status,
        o.receiver_name,
        o.receiver_phone,
        o.receiver_addr,
        o.created_at,
        o.updated_at,
        o.tracking_no,
        o.signed_at,
        o.logistics_company,

        p.product_name,
        p.category AS product_category,
        p.price AS product_price,
        p.stock AS product_stock,
        p.description AS product_description,
        p.after_sale_policy AS product_after_sale_policy
    FROM orders o
    LEFT JOIN products p ON o.product_id = p.id
    WHERE o.user_id = %s
    ORDER BY o.created_at DESC
    LIMIT %s
    """

    with mysql_cursor() as cursor:
        cursor.execute(sql, (user_id, limit))
        rows = cursor.fetchall()

    return [_enrich_order(row) for row in rows]