"""
知识库检索模块。

职责：
1. 根据用户问题检索 Chroma；
2. 支持按 category 过滤；
3. 将检索结果组织成 RAG 上下文。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from rag.chroma_store import get_knowledge_collection


def infer_category(query: str) -> Optional[str]:
    """
    根据关键词粗略推断知识类别。
    后续可以替换成 IntentRouterAgent 的输出。
    """
    q = query.lower()

    if any(w in q for w in ["退款", "退钱", "到账", "原路退回"]):
        return "refund_policy"

    if any(w in q for w in ["售后", "退货", "换货", "维修", "质量问题", "七天无理由"]):
        return "after_sale_policy"

    if any(w in q for w in ["物流", "快递", "配送", "发货", "签收", "运费", "到哪"]):
        return "logistics_policy"

    if any(w in q for w in ["优惠券", "券", "满减", "折扣", "不能用"]):
        return "coupon_policy"

    if any(w in q for w in ["会员", "vip", "等级", "权益", "积分"]):
        return "membership_policy"

    if any(w in q for w in ["发票", "抬头", "税号", "电子发票", "纸质发票"]):
        return "invoice_policy"

    return None


def search_knowledge(
    query: str,
    top_k: int = 5,
    category: Optional[str] = None,
) -> List[Dict]:
    """
    检索知识库。

    Args:
        query: 用户问题
        top_k: 返回数量
        category: 可选类别过滤

    Returns:
        检索结果列表
    """
    collection = get_knowledge_collection()

    if category is None:
        category = infer_category(query)

    query_args = {
        "query_texts": [query],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    if category:
        query_args["where"] = {"category": category}

    result = collection.query(**query_args)

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    hits = []

    for document, metadata, distance in zip(documents, metadatas, distances):
        hits.append({
            "content": document,
            "title": metadata.get("title", ""),
            "source": metadata.get("source", ""),
            "source_path": metadata.get("source_path", ""),
            "source_ext": metadata.get("source_ext", ""),
            "category": metadata.get("category", ""),
            "file_md5": metadata.get("file_md5", ""),
            "chunk_md5": metadata.get("chunk_md5", ""),
            "chunk_index": metadata.get("chunk_index", -1),
            "distance": distance,
        })

    return hits


def build_rag_context(results: List[Dict]) -> str:
    """
    将检索结果拼成 LLM 可用上下文。
    """
    if not results:
        return ""

    parts = []

    for idx, item in enumerate(results, start=1):
        parts.append(
            f"【知识片段{idx}】\n"
            f"标题：{item.get('title', '')}\n"
            f"来源：{item.get('source', '')}\n"
            f"类别：{item.get('category', '')}\n"
            f"内容：{item.get('content', '')}"
        )

    return "\n\n".join(parts)


if __name__ == "__main__":
    query = "换货怎么换？"
    results = search_knowledge(query, top_k=3)
    print(build_rag_context(results))