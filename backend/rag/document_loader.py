"""
知识文件读取、文本切分和 md5 计算工具。

职责：
1. 扫描知识目录中的文件；
2. 读取文件内容；
3. 计算文件级 md5；
4. 切分为知识片段；
5. 计算片段级 md5；
6. 组装为可写入 Chroma 的数据结构。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List

from utils.path_utils import get_abs_path, get_relative_path


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


CATEGORY_MAP = {
    "售后政策": "after_sale_policy",
    "售后": "after_sale_policy",
    "after_sale": "after_sale_policy",

    "退款规则": "refund_policy",
    "退款": "refund_policy",
    "refund": "refund_policy",

    "物流说明": "logistics_policy",
    "物流": "logistics_policy",
    "logistics": "logistics_policy",

    "优惠券规则": "coupon_policy",
    "优惠券": "coupon_policy",
    "coupon": "coupon_policy",

    "会员规则": "membership_policy",
    "会员": "membership_policy",
    "membership": "membership_policy",

    "发票规则": "invoice_policy",
    "发票": "invoice_policy",
    "invoice": "invoice_policy",
}


def calc_file_md5(file_path: Path) -> str:
    """
    计算文件 md5。
    """
    md5 = hashlib.md5()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)

    return md5.hexdigest()


def calc_text_md5(text: str) -> str:
    """
    计算文本 md5。
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def infer_category(file_path: Path) -> str:
    """
    根据文件名推断知识类别。
    """
    file_name = file_path.stem.lower()

    for key, category in CATEGORY_MAP.items():
        if key.lower() in file_name:
            return category

    return "general_policy"


def read_text_file(file_path: Path) -> str:
    """
    读取文本类文件。
    """
    return file_path.read_text(encoding="utf-8")


def read_pdf_file(file_path: Path) -> str:
    """
    读取 PDF 文本。

    依赖：
    pip install pypdf
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("读取 PDF 需要安装 pypdf：pip install pypdf") from exc

    reader = PdfReader(str(file_path))
    texts = []

    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            texts.append(text)

    return "\n\n".join(texts)


def read_knowledge_file(file_path: Path) -> str:
    """
    根据文件扩展名读取知识文件。
    """
    suffix = file_path.suffix.lower()

    if suffix in {".md", ".txt"}:
        return read_text_file(file_path)

    if suffix == ".pdf":
        return read_pdf_file(file_path)

    raise ValueError(f"不支持的文件类型: {file_path}")


def normalize_text(text: str) -> str:
    """
    清理多余空白。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_long_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    """
    将较长文本切分为多个片段。
    """
    text = text.strip()

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def split_by_headings_or_paragraphs(
    text: str,
    default_title: str,
    max_chars: int = 800,
    overlap: int = 100,
) -> List[Dict[str, str]]:
    """
    优先按标题结构切分。
    如果没有明显标题，则按长度进行切分。
    """
    text = normalize_text(text)
    lines = text.splitlines()

    sections: List[Dict[str, str]] = []
    current_title = default_title
    buffer: List[str] = []

    heading_pattern = re.compile(r"^\s{0,3}#{1,6}\s+(.+)$")

    def flush_section():
        nonlocal buffer, current_title

        content = "\n".join(buffer).strip()
        buffer = []

        if not content:
            return

        sub_chunks = split_long_text(
            text=content,
            max_chars=max_chars,
            overlap=overlap,
        )

        for idx, sub_chunk in enumerate(sub_chunks):
            title = current_title

            if len(sub_chunks) > 1:
                title = f"{current_title}-片段{idx + 1}"

            sections.append({
                "title": title,
                "content": sub_chunk,
            })

    for line in lines:
        stripped = line.strip()
        match = heading_pattern.match(stripped)

        if match:
            flush_section()
            current_title = match.group(1).strip() or default_title
        else:
            if stripped:
                buffer.append(line)

    flush_section()

    if not sections:
        chunks = split_long_text(
            text=text,
            max_chars=max_chars,
            overlap=overlap,
        )

        for idx, chunk in enumerate(chunks):
            sections.append({
                "title": f"{default_title}-片段{idx + 1}",
                "content": chunk,
            })

    return sections


def list_knowledge_files(relative_dir: str = "data/knowledge") -> List[Path]:
    """
    获取知识目录下所有支持的文件。
    """
    knowledge_dir = get_abs_path(relative_dir)

    if not knowledge_dir.exists():
        raise FileNotFoundError(f"知识目录不存在: {knowledge_dir}")

    files = []

    for file_path in knowledge_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(file_path)

    return sorted(files)


def build_chroma_documents_from_file(file_path: Path) -> List[Dict]:
    """
    将单个知识文件转换为 Chroma 可写入的数据。
    """
    file_md5 = calc_file_md5(file_path)
    source_path = get_relative_path(file_path)
    source = file_path.name
    source_ext = file_path.suffix.lower()
    category = infer_category(file_path)

    raw_text = read_knowledge_file(file_path)

    sections = split_by_headings_or_paragraphs(
        text=raw_text,
        default_title=file_path.stem,
        max_chars=800,
        overlap=100,
    )

    source_key = calc_text_md5(source_path)[:10]

    docs = []

    for idx, section in enumerate(sections):
        content = section["content"].strip()
        title = section["title"].strip()

        if not content:
            continue

        chunk_md5 = calc_text_md5(content)

        doc_id = f"{source_key}_{idx}_{chunk_md5[:10]}"

        docs.append({
            "id": doc_id,
            "document": content,
            "metadata": {
                "title": title,
                "source": source,
                "source_path": source_path,
                "source_ext": source_ext,
                "category": category,
                "file_md5": file_md5,
                "chunk_md5": chunk_md5,
                "chunk_index": idx,
            },
        })

    return docs