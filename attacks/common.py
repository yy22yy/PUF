"""Shared CRP loading, validation, splitting, and evaluation helpers."""

from pathlib import Path
import hashlib
import io

import numpy as np


def _validate_crps(value: np.ndarray) -> np.ndarray:
    crps = np.asarray(value)
    if crps.ndim != 2 or crps.shape[1] < 2:
        raise ValueError("CRP 必须是二维数组，且每行至少包含 1 位 Challenge 和 1 位 Response")
    if not np.all((crps == 0) | (crps == 1)):
        raise ValueError("CRP 只能包含 0 和 1")
    return crps


def load_crps(path: str | Path) -> np.ndarray:
    """Load a CRP matrix whose last column contains the 0/1 Response."""
    value = np.load(Path(path), allow_pickle=False)
    if not isinstance(value, np.ndarray):
        value.close()
        raise ValueError("CRP 文件必须是 NPY 矩阵")
    return _validate_crps(value)


def load_crps_snapshot(path: str | Path) -> tuple[np.ndarray, str]:
    """Read and hash the same immutable byte snapshot used to parse a CRP."""
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    if not isinstance(value, np.ndarray):
        value.close()
        raise ValueError("CRP 文件必须是 NPY 矩阵")
    return _validate_crps(value), digest


def split_by_response(
    challenges: np.ndarray,
    responses: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make a deterministic stratified split, preserving both Response classes."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size 必须大于 0 且小于 1")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for response in (0, 1):
        indices = np.flatnonzero(responses == response)
        if indices.size < 2:
            raise ValueError("Response 的 0 和 1 各自都至少需要 2 条样本")
        rng.shuffle(indices)
        test_count = max(
            1,
            min(indices.size - 1, int(np.ceil(indices.size * test_size))),
        )
        test_indices.extend(indices[:test_count].tolist())
        train_indices.extend(indices[test_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    train_indices_array = np.asarray(train_indices)
    test_indices_array = np.asarray(test_indices)
    return (
        challenges[train_indices_array],
        challenges[test_indices_array],
        responses[train_indices_array],
        responses[test_indices_array],
    )


def accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Return the fraction of predictions that match the labels."""
    return float(np.mean(labels == predictions))


def majority_baseline(labels: np.ndarray) -> float:
    """Return the accuracy of always guessing the most common class."""
    return float(max(np.mean(labels == 0), np.mean(labels == 1)))
