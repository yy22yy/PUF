"""不同 CRP 训练规模下的单 APUF ECP 攻击效果（可复现实验脚本）。

对应归档产物：``archive/2026-10-ecp-train-sizes/``。

实验协议（与归档 manifest 的 ``protocol`` 字段一致）：

1. 先做一次固定划分，得到一份固定的测试集（``test_size``，``split_seed``）；
2. 把训练集打乱一次，之后所有规模都取这份打乱结果的**前 n 行**（嵌套前缀），
   因此不同 n_train 之间只差训练样本数量，不差样本内容；
3. 每个规模用 3 个不同的初始化种子各训练一次，种子只影响参数初始化；
4. 全批量（batch = 全部训练行）Adam，``epochs`` 轮，测试集只在训练结束后评估。

两种数据来源：

* ``--datasets DIR``：直接读取归档的固定划分（``train_<n>.npy`` + ``fixed_test.npy``）。
  这是复现归档数字的方式，结果应与 ``primary/metrics.csv`` 完全一致。
* ``--crp FILE``：从一份 CRP 文件现场构建嵌套前缀（用于新数据）。

用法::

    # 复现归档结果（需要 PyTorch 环境）
    .venv-torch/bin/python -m experiments.ecp_train_sizes \
        --datasets archive/2026-10-ecp-train-sizes/datasets \
        --output runs/ecp-train-sizes-repro

    # 用自己的 CRP 文件快速试跑
    .venv-torch/bin/python -m experiments.ecp_train_sizes \
        --crp data/apuf/n64_w42_c42_m100_run01/crps.npy \
        --train-sizes 50 100 200 --epochs 100

脚本只在 ``--output`` 指定的**新目录**中写入，绝不覆盖已有运行或输入数据。
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from attacks.common import majority_baseline
from attacks.ecp.common import (
    binary_metrics,
    file_sha256,
    initialize_parameters,
    load_backend,
    save_npz_with_csv,
    write_json,
)
from attacks.ecp.common import split_indices
from puf.features import ChallengeFeatureTransformer

METRIC_FIELDS = (
    "n_train", "n_test", "seed", "parameter_count",
    "train_accuracy", "test_accuracy", "test_errors",
    "majority_baseline", "majority_response",
    "train_label_one_rate", "test_label_one_rate", "fit_seconds",
)

DEFAULT_RANK = 4
DEFAULT_EPOCHS = 500
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_TEST_SIZE = 0.25
DEFAULT_SPLIT_SEED = 42
DEFAULT_MODEL_SEEDS = (42, 43, 44)
DEFAULT_TRAIN_SIZES = (1000, 5000, 10000, 20000, 50000, 74998)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_archived_split(directory: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """读取归档划分：返回 (固定测试集, {n_train: 训练前缀})。"""
    test_path = directory / "fixed_test.npy"
    if not test_path.is_file():
        raise FileNotFoundError(f"缺少固定测试集：{test_path}")
    prefixes: dict[int, np.ndarray] = {}
    for path in sorted(directory.glob("train_*.npy")):
        try:
            size = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        prefixes[size] = np.load(path, allow_pickle=False)
    if not prefixes:
        raise FileNotFoundError(f"目录中没有 train_<n>.npy：{directory}")
    return np.load(test_path, allow_pickle=False), prefixes


def build_prefixes(
    crps: np.ndarray, test_size: float, split_seed: int, train_sizes: tuple[int, ...],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """按归档协议从一份 CRP 文件构建固定测试集与嵌套训练前缀。"""
    crps = np.asarray(crps)
    if crps.ndim != 2 or crps.shape[1] < 2:
        raise ValueError("CRP 必须是二维数组，最后一列为 Response")
    labels = crps[:, -1].astype(np.uint8)
    train_indices, test_indices = split_indices(labels, test_size, split_seed)
    shuffled = np.random.default_rng(split_seed).permutation(train_indices)
    prefixes = {}
    for size in train_sizes:
        if size < 1:
            raise ValueError("train-sizes 必须为正整数")
        if size > len(shuffled):
            raise ValueError(
                f"训练行数不足：需要 {size} 行，实际划分后只有 {len(shuffled)} 行"
            )
        prefixes[size] = crps[shuffled[:size]]
    return crps[test_indices], prefixes


def train_one(
    train_crps: np.ndarray,
    test_crps: np.ndarray,
    features_all: tuple[np.ndarray, np.ndarray] | None = None,
    *,
    rank: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """训练一次并返回 (指标记录, 测试预测, 测试 logits)。"""
    transformer = ChallengeFeatureTransformer()
    if features_all is None:
        train_x = transformer.transform(train_crps[:, :-1])
        test_x = transformer.transform(test_crps[:, :-1])
    else:
        train_x, test_x = features_all
    train_y = train_crps[:, -1].astype(np.uint8)
    test_y = test_crps[:, -1].astype(np.uint8)

    initial = initialize_parameters(train_x.shape[1], rank, seed)
    backend = load_backend("torch", initial, learning_rate)

    started = time.perf_counter()
    for _ in range(epochs):
        loss = backend.train_batch(train_x, train_y)
        if not np.isfinite(loss):
            raise FloatingPointError(f"seed={seed} n_train={len(train_y)} 出现非有限损失")
    fit_seconds = time.perf_counter() - started

    train_accuracy = binary_metrics(train_y, backend.logits(train_x))["accuracy"]
    logits = backend.logits(test_x)
    test_accuracy = binary_metrics(test_y, logits)["accuracy"]
    record = {
        "n_train": len(train_y),
        "n_test": len(test_y),
        "seed": seed,
        "parameter_count": int(sum(value.size for value in initial.values())),
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "test_errors": int(round((1.0 - test_accuracy) * len(test_y))),
        "majority_baseline": majority_baseline(test_y),
        "majority_response": int(np.bincount(test_y).argmax()),
        "train_label_one_rate": float(train_y.mean()),
        "test_label_one_rate": float(test_y.mean()),
        "fit_seconds": fit_seconds,
    }
    return record, (logits >= 0).astype(np.uint8), logits


def summarize(records: list[dict[str, float]]) -> list[dict[str, float]]:
    """按 n_train 汇总均值与标准差（与归档 summary.json 字段一致）。"""
    sizes = sorted({int(record["n_train"]) for record in records})
    summary = []
    for size in sizes:
        group = [record for record in records if int(record["n_train"]) == size]
        test_accuracy = np.array([record["test_accuracy"] for record in group], dtype=np.float64)
        summary.append({
            "n_train": size,
            "n_test": int(group[0]["n_test"]),
            "runs": len(group),
            "train_accuracy_mean": float(np.mean([r["train_accuracy"] for r in group])),
            "test_accuracy_mean": float(np.mean(test_accuracy)),
            "test_accuracy_std": float(np.std(test_accuracy)),
            "test_errors_mean": float(np.mean([r["test_errors"] for r in group])),
            "fit_seconds_mean": float(np.mean([r["fit_seconds"] for r in group])),
        })
    return summary


def write_metrics(path: Path, records: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record[name] for name in METRIC_FIELDS})


def write_summary(path: Path, summary: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def source_record(project_root: Path) -> dict:
    paths = [Path(__file__).resolve()]
    paths.extend(
        project_root / name
        for name in ("attacks/common.py", "attacks/ecp/common.py", "puf/features.py", "puf.toml")
    )
    record = {
        "sha256": {
            str(path.relative_to(project_root)): file_sha256(path)
            for path in paths if path.exists()
        }
    }
    try:
        record["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        record["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=project_root, text=True,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        record["git_commit"] = None
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="ECP 训练规模扫描（嵌套训练前缀 + 固定测试集）",
        epilog=(
            "示例：\n"
            "  .venv-torch/bin/python -m experiments.ecp_train_sizes "
            "--datasets archive/2026-10-ecp-train-sizes/datasets "
            "--output runs/ecp-train-sizes-repro"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--datasets", help="归档划分目录（含 train_<n>.npy 与 fixed_test.npy）")
    parser.add_argument("--crp", help="CRP NPY 文件；与 --datasets 二选一")
    parser.add_argument("--train-sizes", type=int, nargs="+", default=list(DEFAULT_TRAIN_SIZES),
                        help=f"训练规模列表；默认 {list(DEFAULT_TRAIN_SIZES)}")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_MODEL_SEEDS),
                        help=f"初始化种子列表；默认 {list(DEFAULT_MODEL_SEEDS)}")
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--output", required=True, help="新的运行目录（必须不存在）")
    parser.add_argument("--dataset-id", default=None,
                        help="数据标识，写入 manifest；默认取目录名或 CRP 文件名")
    args = parser.parse_args(argv)

    if bool(args.datasets) == bool(args.crp):
        parser.error("--datasets 与 --crp 必须且只能指定一个")
    if args.rank < 1 or args.epochs < 1:
        parser.error("--rank 与 --epochs 必须为正整数")

    project_root = Path(__file__).resolve().parents[1]
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = (project_root / output).resolve()
    if output.exists():
        parser.error(f"输出目录已存在：{output}；请换一个新目录，脚本不会覆盖已有运行")

    if args.datasets:
        data_dir = Path(args.datasets).expanduser()
        if not data_dir.is_absolute():
            data_dir = (project_root / data_dir).resolve()
        test_crps, prefixes = load_archived_split(data_dir)
        dataset_id = args.dataset_id or data_dir.name
        data_description = {"kind": "archived_split", "path": str(data_dir),
                            "sha256": {p.name: file_sha256(p) for p in sorted(data_dir.glob("*.npy"))}}
    else:
        crp_path = Path(args.crp).expanduser()
        if not crp_path.is_absolute():
            crp_path = (project_root / crp_path).resolve()
        crps = np.load(crp_path, allow_pickle=False)
        test_crps, prefixes = build_prefixes(crps, args.test_size, args.split_seed, tuple(args.train_sizes))
        dataset_id = args.dataset_id or crp_path.stem
        data_description = {"kind": "crp_file", "path": str(crp_path), "sha256": file_sha256(crp_path)}

    missing = [size for size in args.train_sizes if size not in prefixes]
    if missing:
        parser.error(f"以下规模没有对应数据：{missing}；归档目录中可用：{sorted(prefixes)}")

    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "record_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "experiment": "ecp_train_sizes",
        "dataset_id": dataset_id,
        "data": data_description,
        "protocol": "nested prefixes of one shuffled training set; fixed held-out test; "
                    "seeds change initialization only; full-batch training",
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "torch": __import__("torch").__version__,
        },
        "source": source_record(project_root),
    }
    config = {
        "train_sizes": list(args.train_sizes), "model_seeds": list(args.seeds),
        "rank": args.rank, "order": 1, "epochs": args.epochs,
        "learning_rate": args.learning_rate, "batch_size": "full",
        "test_size": args.test_size, "split_seed": args.split_seed,
        "optimizer": {"name": "Adam", "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
        "prediction_rule": "logit >= 0 -> 1",
    }

    try:
        write_json(output / "config.json", config)
        write_json(output / "manifest.json", manifest)
        save_npz_with_csv(
            output / "fixed_test.npz",
            crps=test_crps,
        )
        transformer = ChallengeFeatureTransformer()
        test_features = transformer.transform(test_crps[:, :-1])
        records: list[dict[str, float]] = []
        for size in args.train_sizes:
            train_crps = prefixes[size]
            train_x = transformer.transform(train_crps[:, :-1])
            for seed in args.seeds:
                record, predictions, logits = train_one(
                    train_crps, test_crps, (train_x, test_features),
                    rank=args.rank, epochs=args.epochs,
                    learning_rate=args.learning_rate, seed=seed,
                )
                records.append(record)
                print(
                    f"n_train={size:>6} seed={seed}: "
                    f"train={record['train_accuracy']:.4f} test={record['test_accuracy']:.4f} "
                    f"({record['fit_seconds']:.2f}s)",
                    flush=True,
                )
                save_npz_with_csv(
                    output / f"predictions_n{size}_seed{seed}.npz",
                    rows=np.arange(len(test_crps)), labels=test_crps[:, -1].astype(np.uint8),
                    logits=logits, predictions=predictions,
                )
        write_metrics(output / "metrics.csv", records)
        summary = summarize(records)
        write_summary(output / "summary.csv", summary)
        write_json(output / "summary.json", summary)
        manifest.update(
            status="completed", finished_at=utc_now(), completed_runs=len(records),
            mean_test_accuracy=float(np.mean([r["test_accuracy"] for r in records])),
        )
        manifest["artifact_sha256"] = {
            path.name: file_sha256(path) for path in sorted(output.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        write_json(output / "manifest.json", manifest)
    except BaseException as error:
        manifest.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            finished_at=utc_now(), error=f"{type(error).__name__}: {error}",
        )
        write_json(output / "manifest.json", manifest)
        raise

    best = summary[-1]
    print(f"最大规模 n_train={best['n_train']}："
          f"测试准确率 {best['test_accuracy_mean']:.4f} ± {best['test_accuracy_std']:.4f}")
    print(f"实验目录：{output}")


if __name__ == "__main__":
    main()