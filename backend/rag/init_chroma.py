"""
知识库初始化与同步脚本。

运行方式：
python -m rag.init_chroma --reset
python -m rag.init_chroma
python -m rag.init_chroma --force

参数说明：
--reset  清空 Chroma collection 和 manifest 后重新导入
--force  不判断 md5，强制重新导入所有本地知识文件
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from rag.chroma_store import get_knowledge_collection, reset_knowledge_collection
from rag.document_loader import (
    build_chroma_documents_from_file,
    calc_file_md5,
    infer_category,
    list_knowledge_files,
)
from utils.path_utils import ensure_dir, get_abs_path, get_relative_path


MANIFEST_RELATIVE_PATH = "data/chroma_manifest.json"


def get_manifest_path() -> Path:
    """
    获取 manifest 文件路径。
    """
    ensure_dir("data")
    return get_abs_path(MANIFEST_RELATIVE_PATH)


def empty_manifest() -> Dict:
    """
    创建空 manifest。
    """
    return {
        "version": 1,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "files": {},
    }


def load_manifest() -> Dict:
    """
    读取 manifest。
    """
    manifest_path = get_manifest_path()

    if not manifest_path.exists():
        return empty_manifest()

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if "files" not in data or not isinstance(data["files"], dict):
            return empty_manifest()

        return data

    except Exception:
        return empty_manifest()


def save_manifest(manifest: Dict) -> None:
    """
    保存 manifest。
    """
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    manifest_path = get_manifest_path()

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def delete_chunks_by_ids(collection, chunk_ids: List[str]) -> None:
    """
    根据 chunk id 删除 Chroma 中的旧片段。
    """
    if not chunk_ids:
        return

    try:
        collection.delete(ids=chunk_ids)
    except Exception as exc:
        print(f"按 ids 删除旧片段失败: {exc}")


def delete_chunks_by_source_path(collection, source_path: str) -> None:
    """
    根据 source_path 删除 Chroma 中的旧片段。
    用于 manifest 缺失 chunk_ids 或异常恢复。
    """
    try:
        collection.delete(where={"source_path": source_path})
    except Exception:
        pass


def chunks_exist(collection, chunk_ids: List[str]) -> bool:
    """
    检查 manifest 中记录的 chunk 是否仍然存在于 Chroma。

    只检查第一个 chunk，主要防止 Chroma 被清空但 manifest 还在。
    """
    if not chunk_ids:
        return False

    try:
        result = collection.get(
            ids=[chunk_ids[0]],
            include=["metadatas"],
        )
        return len(result.get("ids", [])) > 0
    except Exception:
        return False


def build_file_record(file_path: Path, docs: List[Dict], file_md5: str) -> Dict:
    """
    构建 manifest 中单个文件的记录。
    """
    source_path = get_relative_path(file_path)
    chunk_ids = [doc["id"] for doc in docs]

    return {
        "source": file_path.name,
        "source_path": source_path,
        "source_ext": file_path.suffix.lower(),
        "category": infer_category(file_path),
        "file_md5": file_md5,
        "chunk_count": len(chunk_ids),
        "chunk_ids": chunk_ids,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def sync_deleted_files(collection, manifest: Dict, current_source_paths: set[str]) -> int:
    """
    如果本地文件已经删除，则同步删除 Chroma 中对应片段和 manifest 记录。
    """
    deleted_count = 0

    old_source_paths = list(manifest.get("files", {}).keys())

    for source_path in old_source_paths:
        if source_path in current_source_paths:
            continue

        record = manifest["files"].get(source_path, {})
        chunk_ids = record.get("chunk_ids", [])

        delete_chunks_by_ids(collection, chunk_ids)
        delete_chunks_by_source_path(collection, source_path)

        manifest["files"].pop(source_path, None)

        print(f"已删除失效知识文件对应片段: {source_path}")
        deleted_count += 1

    return deleted_count


def import_docs_to_chroma(reset: bool = False, force: bool = False) -> None:
    """
    同步知识目录到 Chroma。
    """
    if reset:
        collection = reset_knowledge_collection()
        manifest = empty_manifest()
    else:
        collection = get_knowledge_collection()
        manifest = load_manifest()

    files = list_knowledge_files("data/knowledge")
    current_source_paths = {get_relative_path(file_path) for file_path in files}

    deleted_files = sync_deleted_files(
        collection=collection,
        manifest=manifest,
        current_source_paths=current_source_paths,
    )

    scanned_files = len(files)
    imported_files = 0
    skipped_files = 0
    imported_chunks = 0

    for file_path in files:
        source_path = get_relative_path(file_path)
        current_file_md5 = calc_file_md5(file_path)

        old_record = manifest.get("files", {}).get(source_path)

        if (
            not force
            and old_record
            and old_record.get("file_md5") == current_file_md5
            and chunks_exist(collection, old_record.get("chunk_ids", []))
        ):
            print(f"跳过未变化文件: {source_path}")
            skipped_files += 1
            continue

        if old_record:
            delete_chunks_by_ids(collection, old_record.get("chunk_ids", []))
            delete_chunks_by_source_path(collection, source_path)

        docs = build_chroma_documents_from_file(file_path)

        if not docs:
            print(f"文件无有效文本，跳过: {source_path}")
            continue

        collection.upsert(
            ids=[doc["id"] for doc in docs],
            documents=[doc["document"] for doc in docs],
            metadatas=[doc["metadata"] for doc in docs],
        )

        manifest["files"][source_path] = build_file_record(
            file_path=file_path,
            docs=docs,
            file_md5=current_file_md5,
        )

        imported_files += 1
        imported_chunks += len(docs)

        print(f"已导入/更新文件: {source_path}，片段数: {len(docs)}")

    save_manifest(manifest)

    print("-" * 60)
    print(f"扫描文件数: {scanned_files}")
    print(f"删除失效文件数: {deleted_files}")
    print(f"导入/更新文件数: {imported_files}")
    print(f"跳过未变化文件数: {skipped_files}")
    print(f"本次写入片段数: {imported_chunks}")
    print(f"collection 总片段数: {collection.count()}")
    print(f"manifest 路径: {get_manifest_path()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="清空 collection 和 manifest 后重新导入")
    parser.add_argument("--force", action="store_true", help="强制重新导入所有本地知识文件")
    args = parser.parse_args()

    import_docs_to_chroma(reset=args.reset, force=args.force)