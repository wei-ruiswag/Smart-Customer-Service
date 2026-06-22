"""
知识检索服务层。

该文件主要给 MCP 工具或 Agent 调用。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from rag.knowledge_search import search_knowledge


class KnowledgeService:
    @staticmethod
    def search(
        query: str,
        top_k: int = 3,
        category: Optional[str] = None,
    ) -> List[Dict]:
        results = search_knowledge(
            query=query,
            top_k=top_k,
            category=category,
        )

        output = []

        for item in results:
            distance = item.get("distance")

            if distance is None:
                score = None
            else:
                score = round(1 - float(distance), 4)

            output.append({
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "source": item.get("source", ""),
                "source_path": item.get("source_path", ""),
                "category": item.get("category", ""),
                "score": score,
            })

        return output