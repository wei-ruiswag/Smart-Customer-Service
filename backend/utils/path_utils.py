"""
为整个 backend 工程提供统一的绝对路径工具。

注意：
get_project_root() 不是根据“在哪个文件里调用”来判断根目录，
而是根据 path_utils.py 自己的位置来判断。

当前文件位置：
backend/utils/path_utils.py

所以：
Path(__file__).resolve().parents[1] = backend 目录
"""

from pathlib import Path


def get_project_root() -> Path:
    """
    获取 backend 工程根目录。

    Returns:
        Path: backend 目录的绝对路径
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[1]
    return project_root


def get_abs_path(relative_path: str) -> Path:
    """
    根据相对于 backend 根目录的路径，返回绝对路径。

    Args:
        relative_path: 相对 backend 的路径，例如 "data/knowledge"

    Returns:
        Path: 绝对路径
    """
    return get_project_root() / relative_path


def ensure_dir(relative_path: str) -> Path:
    """
    确保某个目录存在，如果不存在则自动创建。

    Args:
        relative_path: 相对 backend 的目录路径

    Returns:
        Path: 该目录的绝对路径
    """
    abs_path = get_abs_path(relative_path)
    abs_path.mkdir(parents=True, exist_ok=True)
    return abs_path


if __name__ == "__main__":
    print("项目根目录:", get_project_root())
    print("知识库目录:", get_abs_path("data/knowledge"))
    # print("Chroma目录:", ensure_dir("data/chroma_db"))