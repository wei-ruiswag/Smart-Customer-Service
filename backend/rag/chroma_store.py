"""
Chroma 数据库连接层。

职责：
1. 创建 Chroma 本地持久化客户端；
2. 获取或创建知识库 collection；
3. 删除并重建知识库 collection。
"""

import os
from typing import List

import chromadb
from chromadb.utils import embedding_functions
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from openai import OpenAI

from utils.path_utils import ensure_dir, get_abs_path


COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "ecommerce_knowledge")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "data/chroma_db")


# 本地 HuggingFace 模型
EMBEDDING_MODEL = os.getenv("CHROMA_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")  # BAAI/bge-small-zh-v1.5


def get_embedding_function():
    """
    获取 Chroma 使用的 embedding 函数。

    默认使用中文 embedding 模型。
    如果本地暂时无法下载模型，可以设置环境变量：
    CHROMA_USE_DEFAULT_EMBEDDING=1
    """
    use_default = os.getenv("CHROMA_USE_DEFAULT_EMBEDDING", "0") == "1"

    if use_default:
        return None

    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def get_chroma_client():
    """
    获取 Chroma 本地持久化客户端。
    """
    ensure_dir(CHROMA_DB_DIR)
    chroma_path = get_abs_path(CHROMA_DB_DIR)
    return chromadb.PersistentClient(path=str(chroma_path))


def get_knowledge_collection():
    """
    获取或创建知识库 collection。
    """
    client = get_chroma_client()
    embedding_function = get_embedding_function()

    if embedding_function is None:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "ecommerce customer services knowledge base"},
        )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"description": "ecommerce customer services knowledge base"},
    )


def reset_knowledge_collection():
    """
    删除并重建知识库 collection。
    """
    client = get_chroma_client()

    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"已删除旧 collection: {COLLECTION_NAME}")
    except Exception:
        print(f"collection 不存在，无需删除: {COLLECTION_NAME}")

    return get_knowledge_collection()