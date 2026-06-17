"""
记忆管理器：统一协调工作记忆、短期记忆和长期记忆。
"""

from __future__ import annotations

from typing import Any

from memory.short_term import ShortTermMemory
from memory.working import WorkingMemory
from memory.long_term import LongTermMemory


class MemoryManager:
    def __init__(
        self,
        short_term: ShortTermMemory | None = None,
        working: WorkingMemory | None = None,
        long_term: LongTermMemory | None = None,
    ):
        self.short_term = short_term or ShortTermMemory()
        self.working = working or WorkingMemory()
        self.long_term = long_term or LongTermMemory()

    async def build_context(
        self,
        session_id: str,
        user_id: str,
        query: str,
        short_max_tokens: int = 3000,
        long_top_k: int = 5,
    ) -> dict[str, Any]:
        """
        构建一次 Agent 调用需要的记忆上下文。
        """
        short_context = await self.short_term.get_context_window(
            session_id=session_id,
            max_tokens=short_max_tokens,
        )

        long_memories = self.long_term.search(
            query=query,
            user_id=user_id,
            top_k=long_top_k,
        )

        self.working.update(
            session_id,
            {
                "query": query,
                "user_id": user_id,
                "short_context": short_context,
                "long_memories": long_memories,
            },
        )

        return {
            "short_context": short_context,
            "long_memories": long_memories,
            "working_context": self.working.get_context(session_id),
        }

    async def save_turn(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """
        保存一轮对话到短期记忆。
        """
        await self.short_term.add_message(
            session_id=session_id,
            role="user",
            content=user_message,
        )

        await self.short_term.add_message(
            session_id=session_id,
            role="assistant",
            content=assistant_message,
        )

        self.working.update(
            session_id,
            {
                "last_user_message": user_message,
                "last_assistant_message": assistant_message,
            },
        )

    def save_long_term_if_needed(
        self,
        session_id: str,
        user_id: str,
        content: str,
        memory_type: str = "conversation_summary",
        source: str = "chat",
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """
        将有长期价值的信息写入长期记忆。

        注意：这里不建议直接保存整轮原始对话。
        最好保存经过抽取或总结后的内容。
        """
        content = content.strip()

        if not content:
            return None

        return self.long_term.add_memory(
            content=content,
            user_id=user_id,
            memory_type=memory_type,
            source=source,
            session_id=session_id,
            metadata=metadata or {},
        )

    def update_working(self, session_id: str, data: dict[str, Any]) -> None:
        """
        更新当前请求的工作记忆。
        """
        self.working.update(session_id, data)

    def get_working_context(self, session_id: str) -> dict[str, Any]:
        return self.working.get_context(session_id)

    def clear_working(self, session_id: str) -> None:
        """
        请求结束后清理工作记忆。
        """
        self.working.clear(session_id)