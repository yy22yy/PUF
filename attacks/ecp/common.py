"""Framework-independent ECP parameters, data protocol, and record helpers."""

from __future__ import annotations

import hashlib
import importlib
import json
import csv
from pathlib import Path

import numpy as np

from attacks.common import split_by_response


PARAMETER_NAMES = ("factors", "coefficients", "bias")


def validate_parameters(
    parameters: dict[str, np.ndarray], feature_count: int | None = None,
) -> dict[str, np.ndarray]:
    """Validate the portable single-APUF parameter layout and copy as float64."""
    if set(parameters) != set(PARAMETER_NAMES):
        raise ValueError(f"模型参数必须恰好包含 {PARAMETER_NAMES}")
    arrays = {name: np.asarray(parameters[name], dtype=np.float64) for name in PARAMETER_NAMES}
    factors = arrays["factors"]
    if factors.ndim != 3 or factors.shape[0] < 1 or factors.shape[1] != 1 or factors.shape[2] < 1:
        raise ValueError("单 APUF 的 factors 形状必须为 (rank, 1, feature_count)")
    if arrays["coefficients"].shape != (factors.shape[0],) or arrays["bias"].shape != (1,):
        raise ValueError("coefficients 必须为 (rank,)，bias 必须为 (1,)")
    if feature_count is not None and factors.shape[2] != feature_count:
        raise ValueError("模型特征维度与 Challenge 特征维度不一致")
    if not all(np.all(np.isfinite(array)) for array in arrays.values()):
        raise ValueError("模型参数必须是有限数值")
    return {name: array.copy() for name, array in arrays.items()}


def initialize_parameters(feature_count: int, rank: int, seed: int) -> dict[str, np.ndarray]:
    """Use one NumPy initializer for both frameworks (TENAX uniform bounds)."""
    for name, value in (("feature_count", feature_count), ("rank", rank)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} 必须为正整数")
    rng = np.random.default_rng(seed)
    bound = np.sqrt(3.0 / feature_count)
    return {
        "factors": rng.uniform(-bound, bound, (rank, 1, feature_count)),
        "coefficients": rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), rank),
        "bias": np.zeros(1, dtype=np.float64),
    }


def numpy_logits(features: np.ndarray, parameters: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate ECP without a training framework, including all rank terms."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("features 必须为二维矩阵")
    parameters = validate_parameters(parameters, features.shape[1])
    projections = np.einsum("bd,rkd->brk", features, parameters["factors"])
    return np.prod(projections, axis=2) @ parameters["coefficients"] + parameters["bias"][0]


def probabilities(logits: np.ndarray) -> np.ndarray:
    """Stable sigmoid, without overflow at either end."""
    return np.exp(-np.logaddexp(0.0, -np.asarray(logits, dtype=np.float64)))


def binary_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels)
    logits = np.asarray(logits, dtype=np.float64)
    if labels.ndim != 1 or labels.size == 0 or labels.shape != logits.shape:
        raise ValueError("labels 和 logits 必须是长度相同的非空一维数组")
    if not np.all((labels == 0) | (labels == 1)) or not np.all(np.isfinite(logits)):
        raise ValueError("labels 必须为 0/1，logits 必须为有限数值")
    signed_logits = np.where(labels == 1, -logits, logits)
    return {
        "accuracy": float(np.mean((logits >= 0) == labels)),
        "bce": float(np.mean(np.logaddexp(0.0, signed_logits))),
    }


def split_indices(responses: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Preserve the existing LR/MLP/DNN split exactly, saving original row IDs."""
    responses = np.asarray(responses)
    if responses.ndim != 1 or not np.all((responses == 0) | (responses == 1)):
        raise ValueError("responses 必须是一维 0/1 数组")
    train, test, _, _ = split_by_response(np.arange(len(responses)), responses, test_size, seed)
    return train, test


def validate_split(train: np.ndarray, test: np.ndarray, size: int) -> None:
    """Reject changed, overlapping, missing, or out-of-range saved row indices."""
    for indices in (train, test):
        if indices.ndim != 1 or len(indices) == 0 or indices.dtype.kind not in "iu":
            raise ValueError("划分索引必须为非空一维整数数组")
    if not np.array_equal(np.sort(np.concatenate((train, test))), np.arange(size)):
        raise ValueError("训练与测试索引必须互斥且恰好覆盖 CRP 的所有行")


def save_parameters(path: str | Path, parameters: dict[str, np.ndarray]) -> None:
    save_npz_with_csv(path, **validate_parameters(parameters))


def save_npz_with_csv(path: str | Path, **arrays: np.ndarray) -> None:
    """Save an NPZ and readable CSV sidecars (one archive index and one per array)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    with path.open("wb") as handle:
        np.savez(handle, **normalized)
    write_npz_csv_sidecars(path, normalized)


def write_npz_csv_sidecars(path: str | Path, arrays: dict[str, np.ndarray] | None = None) -> None:
    """Write CSV companions for an existing NPZ without changing the archive."""
    path = Path(path)
    if arrays is None:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    # The matching archive CSV inventories member shapes and their CSV files.
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("array", "shape", "dtype", "csv_file"))
        for name, array in arrays.items():
            member_csv = f"{path.stem}__{name}.csv"
            writer.writerow((name, "x".join(map(str, array.shape)), str(array.dtype), member_csv))
            flat = array.reshape(array.shape[0], -1) if array.ndim else array.reshape(1, 1)
            with (path.parent / member_csv).open("w", newline="", encoding="utf-8") as member:
                out = csv.writer(member)
                out.writerow(["index", *[f"value_{i}" for i in range(flat.shape[1])]])
                for index, row in enumerate(flat):
                    out.writerow([index, *row.tolist()])


def load_parameters(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return validate_parameters({name: archive[name] for name in archive.files})


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: dict) -> None:
    """Atomically update one record; interrupted writes cannot truncate it."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_backend(name: str, parameters: dict[str, np.ndarray], learning_rate: float = 0.001):
    if name not in ("torch", "tensorflow"):
        raise ValueError(f"未知 backend：{name}")
    try:
        module = importlib.import_module(f"attacks.ecp.{name}_backend")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"无法加载 {name} 后端（缺少 {error.name}）；请按 attacks/ecp/README.md 配置对应环境"
        ) from error
    return module.Backend(validate_parameters(parameters), learning_rate=learning_rate)
