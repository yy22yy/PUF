"""数组落盘与摘要：全项目共用的 ``.npy`` + 同名 ``.csv`` 写入约定。

约定（README「数据说明」）：保存 ``name.npy`` 时必须同时写同名 ``name.csv``，
便于人工查看与交换；精确类型与形状以 ``.npy`` 为准。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def array_sha256(array: np.ndarray) -> str:
    """Return a content hash for a NumPy array's stored values."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return a content hash for a file on disk."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_array_with_csv(
    path: str | Path, array: np.ndarray, fmt: str = "%d",
) -> tuple[Path, Path]:
    """保存 ``.npy`` 并写出同名 ``.csv``，返回两个路径。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, array)
    csv_path = output.with_suffix(".csv")
    np.savetxt(csv_path, array, delimiter=",", fmt=fmt)
    return output, csv_path