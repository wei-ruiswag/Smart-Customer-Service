# backend/utils/db.py
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import pymysql
from dotenv import load_dotenv
from pymysql.connections import Connection
from pymysql.cursors import DictCursor

load_dotenv()


def get_mysql_connection() -> Connection:
    """
    获取 MySQL 连接。
    统一从环境变量读取数据库配置，避免在多个 service 文件中重复编写连接代码。
    """
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "smart_cs"),
        charset="utf8mb4",
        cursorclass=DictCursor,
    )


@contextmanager
def mysql_cursor(commit: bool = False) -> Iterator[DictCursor]:
    """
    MySQL cursor 上下文管理器。

    commit=False：适合 SELECT 查询。
    commit=True：适合 INSERT / UPDATE / DELETE。
    """
    conn = get_mysql_connection()
    try:
        with conn.cursor() as cursor:
            yield cursor

        if commit:
            conn.commit()

    except Exception:
        if commit:
            conn.rollback()
        raise

    finally:
        conn.close()