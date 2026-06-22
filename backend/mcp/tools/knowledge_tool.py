# backend/mcp/tools/knowledge_tool.py

import asyncio
from typing import Any

from services.knowledge_service import KnowledgeService


async def knowledge_search(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """
    MCP 知识库检索工具。
    负责调用 KnowledgeService 搜索知识库。
    """

    return await asyncio.to_thread(
        KnowledgeService.search,
        query=query,
        top_k=top_k,
    )