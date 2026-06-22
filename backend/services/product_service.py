# backend/services/product_service.py

import os
import pymysql
from typing import Any
from dotenv import load_dotenv

# 在读取环境变量之前，先加载 .env 文件
load_dotenv()

def get_mysql_connection():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "smart_cs"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


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
    适合处理：商品名称、分类、价格、库存、描述等结构化查询。
    """

    sql = """
        SELECT
            id,
            product_name,
            category,
            # brand,
            price,
            stock,
            description,
            after_sale_policy
        FROM products
        WHERE 1 = 1
    """

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

    conn = get_mysql_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()
    finally:
        conn.close()