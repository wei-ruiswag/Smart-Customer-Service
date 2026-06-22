"""
长期记忆 — 基于 Chroma 的持久化记忆

用于存储用户画像、历史问题、历史工单、对话摘要、产品反馈等长期信息。

注意：
- 知识库文档仍然由 rag/init_chroma.py 导入 ecommerce_knowledge collection。
- 长期记忆单独使用 long_term_memory collection。
- 二者共用 data/chroma_db 这个 Chroma 持久化目录，但 collection 不同。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from utils.path_utils import get_abs_path


class LongTermMemory:
    """
    长期记忆：基于 Chroma 的向量检索。

    适合存储：
    - 用户画像 user_profile
    - 对话摘要 conversation_summary
    - 商品咨询 product_question
    - 售后问题 after_sale_issue
    - 工单记录 ticket
    - 其他长期可复用信息

    不建议：
    - 直接存每一句聊天原文
    - 和知识库文档混在同一个 collection
    """

    def __init__(
        self,
        persist_dir: str = "data/chroma_db",
        collection_name: str = "long_term_memory",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
    ):
        self.persist_dir = get_abs_path(persist_dir)
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={
                "description": "Long-term user memory for customer services system"
            },
        )

    @staticmethod
    def _make_id(content: str, user_id: str = "", memory_type: str = "") -> str:
        """
        根据内容、用户、记忆类型生成稳定 id。
        同一用户、同一类型、同一内容会得到同一个 id。
        """
        raw = f"{user_id}|{memory_type}|{content}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Chroma metadata 只适合保存 str / int / float / bool / None。
        对复杂对象转成 json 字符串。
        """
        clean = {}

        for key, value in metadata.items():
            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            else:
                clean[key] = json.dumps(value, ensure_ascii=False)

        return clean

    def add_memory(
        self,
        content: str,
        user_id: str = "default",
        memory_type: str = "general",
        source: str = "",
        session_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        添加一条长期记忆。

        参数：
        - content: 要保存的记忆内容
        - user_id: 用户 id
        - memory_type: 记忆类型，例如 user_profile / product_question / ticket
        - source: 来源，例如 chat / order / ticket / manual
        - session_id: 当前会话 id
        - metadata: 额外信息
        """
        content = content.strip()

        if not content:
            raise ValueError("长期记忆内容不能为空")

        memory_id = self._make_id(
            content=content,
            user_id=user_id,
            memory_type=memory_type,
        )

        now = int(time.time())

        base_metadata = {
            "user_id": user_id,
            "memory_type": memory_type,
            "source": source,
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
        }

        if metadata:
            base_metadata.update(metadata)

        base_metadata = self._clean_metadata(base_metadata)

        self.collection.upsert(
            ids=[memory_id],
            documents=[content],
            metadatas=[base_metadata],
        )

        return memory_id

    def add_document(
        self,
        content: str,
        source: str = "",
        metadata: dict | None = None,
    ) -> str:
        """
        兼容旧接口。

        原来的 LongTermMemory.add_document() 可以继续用。
        但这里默认把它作为 general 类型长期记忆保存。
        """
        metadata = metadata or {}

        return self.add_memory(
            content=content,
            user_id=metadata.get("user_id", "default"),
            memory_type=metadata.get("memory_type", "general"),
            source=source,
            session_id=metadata.get("session_id", ""),
            metadata=metadata,
        )

    def add_documents_batch(self, documents: list[dict]) -> list[str]:
        """
        批量添加长期记忆。
        """
        doc_ids = []

        for doc in documents:
            doc_id = self.add_document(
                content=doc.get("content", ""),
                source=doc.get("source", ""),
                metadata=doc.get("metadata", {}),
            )
            doc_ids.append(doc_id)

        return doc_ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
        memory_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        检索长期记忆。

        支持按 user_id 和 memory_type 过滤。
        """
        query = query.strip()

        if not query:
            return []

        where = {}

        if user_id:
            where["user_id"] = user_id

        if memory_type:
            where["memory_type"] = memory_type

        if len(where) == 0:
            where = None
        elif len(where) == 1:
            pass
        else:
            where = {"$and": [{k: v} for k, v in where.items()]}

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        output = []

        for doc_id, content, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            score = 1 / (1 + float(distance))

            output.append(
                {
                    "id": doc_id,
                    "content": content,
                    "source": metadata.get("source", ""),
                    "metadata": metadata,
                    "score": score,
                    "distance": float(distance),
                }
            )

        return output

    def get_user_memories(
        self,
        user_id: str,
        memory_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        获取某个用户的长期记忆。
        不做语义检索，只按 metadata 查询。
        """
        if memory_type:
            where = {
                "$and": [
                    {"user_id": user_id},
                    {"memory_type": memory_type},
                ]
            }
        else:
            where = {"user_id": user_id}

        results = self.collection.get(
            where=where,
            limit=limit,
        )

        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        output = []

        for doc_id, content, metadata in zip(ids, documents, metadatas):
            output.append(
                {
                    "id": doc_id,
                    "content": content,
                    "source": metadata.get("source", ""),
                    "metadata": metadata,
                }
            )

        return output

    def delete_memory(self, memory_id: str) -> None:
        """
        删除某条长期记忆。
        """
        self.collection.delete(ids=[memory_id])

    def clear_user_memory(self, user_id: str) -> None:
        """
        删除某个用户的所有长期记忆。
        """
        self.collection.delete(where={"user_id": user_id})

    def save(self):
        """
        兼容旧接口。

        Chroma PersistentClient 会自动持久化，不需要手动 save。
        """
        pass

    def load_knowledge_base(self, kb_dir: str) -> int:
        """
        兼容旧接口。

        不建议再通过 LongTermMemory 加载知识库文档。
        知识库文档请放入 data/knowledge 后运行：

            python -m rag.init_chroma

        这个函数保留是为了避免旧代码调用时报错。
        """
        raise NotImplementedError(
            "知识库文档请使用 rag.init_chroma 导入，不建议通过 LongTermMemory.load_knowledge_base 导入。"
        )