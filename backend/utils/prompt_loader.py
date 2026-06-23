# backend/utils/prompt_loader.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from utils.path_utils import get_abs_path


def get_prompt_dir() -> Path:
    """
    prompts 目录。

    默认假设 get_abs_path("prompts") 指向 backend/prompts。
    如果你的 get_abs_path 是以仓库根目录为基准，
    则这里改为：Path(get_abs_path("backend/prompts"))
    """
    return Path(get_abs_path("prompts"))


@lru_cache(maxsize=32)
def load_prompt_file(agent_name: str) -> dict[str, Any]:
    prompt_path = get_prompt_dir() / f"{agent_name}.yml"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt 文件不存在: {prompt_path}")

    with prompt_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Prompt 文件格式错误，应为 YAML dict: {prompt_path}")

    return data


def get_prompt(agent_name: str, key: str, default: str = "") -> str:
    data = load_prompt_file(agent_name)
    value = data.get(key, default)
    return str(value)