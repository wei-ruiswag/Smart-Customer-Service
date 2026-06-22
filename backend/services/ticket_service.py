# backend/services/ticket_service.py

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any

import pymysql


def get_mysql_connection():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "smart_customer_service"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def generate_ticket_no() -> str:
    return f"TK{datetime.now().strftime('%Y%m%d')}{uuid.uuid4().hex[:6].upper()}"


def order_exists(order_no: str, user_id: str | None = None) -> bool:
    """
    可选校验：创建工单前确认订单是否存在。
    如果你的 orders 表字段名不同，需要对应修改。
    """
    if not order_no:
        return False

    sql = "SELECT COUNT(*) AS cnt FROM orders WHERE order_no = %s"
    params: list[Any] = [order_no]

    if user_id:
        sql += " AND user_id = %s"
        params.append(user_id)

    conn = get_mysql_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            row = cursor.fetchone()
            return bool(row and row["cnt"] > 0)
    finally:
        conn.close()


def create_ticket(
    user_id: str,
    order_no: str,
    ticket_type: str,
    priority: str,
    description: str,
    status: str = "待处理",
    validate_order: bool = True,
) -> dict[str, Any]:
    """
    创建工单并写入 MySQL tickets 表。
    """

    if not user_id:
        raise ValueError("user_id 不能为空")

    if not order_no:
        raise ValueError("创建工单需要提供订单号")

    if validate_order and not order_exists(order_no=order_no, user_id=user_id):
        raise ValueError(f"订单 {order_no} 不存在或不属于当前用户")

    ticket_no = generate_ticket_no()

    sql = """
        INSERT INTO tickets
            (ticket_no, user_id, order_no, ticket_type, priority, status, description, created_at, updated_at)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    """

    conn = get_mysql_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                sql,
                (
                    ticket_no,
                    user_id,
                    order_no,
                    ticket_type,
                    priority,
                    status,
                    description,
                ),
            )
        conn.commit()
    finally:
        conn.close()

    return get_ticket_by_no(ticket_no)


def get_ticket_by_no(ticket_no: str) -> dict[str, Any] | None:
    sql = """
        SELECT
            id,
            ticket_no,
            user_id,
            order_no,
            ticket_type,
            priority,
            status,
            description,
            created_at,
            updated_at
        FROM tickets
        WHERE ticket_no = %s
        LIMIT 1
    """

    conn = get_mysql_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, (ticket_no,))
            return cursor.fetchone()
    finally:
        conn.close()


def query_tickets(
    ticket_no: str = "",
    user_id: str = "",
    order_no: str = "",
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    查询工单。
    优先按 ticket_no 精确查；否则按 user_id/order_no 查询最近工单。
    """
    sql = """
        SELECT
            id,
            ticket_no,
            user_id,
            order_no,
            ticket_type,
            priority,
            status,
            description,
            created_at,
            updated_at
        FROM tickets
        WHERE 1 = 1
    """
    params: list[Any] = []

    if ticket_no:
        sql += " AND ticket_no = %s"
        params.append(ticket_no)

    if user_id:
        sql += " AND user_id = %s"
        params.append(user_id)

    if order_no:
        sql += " AND order_no = %s"
        params.append(order_no)

    sql += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)

    conn = get_mysql_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()
    finally:
        conn.close()


def update_ticket_status(ticket_no: str, status: str) -> dict[str, Any] | None:
    sql = """
        UPDATE tickets
        SET status = %s, updated_at = NOW()
        WHERE ticket_no = %s
    """

    conn = get_mysql_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, (status, ticket_no))
        conn.commit()
    finally:
        conn.close()

    return get_ticket_by_no(ticket_no)