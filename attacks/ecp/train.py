"""Train one ECP backend and save a self-describing, portable experiment."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import platform
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np

from attacks.apuf_lr import fit_logistic_regression
from attacks.common import load_crps_snapshot, majority_baseline
from puf.features import ChallengeFeatureTransformer
from .common import (
    binary_metrics, file_sha256, initialize_parameters, load_backend,
    probabilities, save_npz_with_csv, save_parameters, split_indices, write_json,
)
from project_config import load_project_config, resolve_project_path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_record(project_root: Path) -> dict:
    paths = [*Path(__file__).parent.glob("*.py")]
    paths.extend(project_root / name for name in (
        "attacks/common.py", "attacks/apuf_lr.py", "puf/features.py",
        "project_config.py", "puf.toml",
    ))
    result = {"sha256": {str(p.relative_to(project_root)): file_sha256(p) for p in sorted(paths)}}
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        result["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=project_root, text=True,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    return result


def run(args: argparse.Namespace) -> Path:
    project_root, paths = load_project_config(args.config)
    if args.rank < 1 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("rank、epochs、batch-size 必须为正整数")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate 必须为有限正数")
    if args.seed < 0:
        raise ValueError("seed 必须为非负整数")

    crp_path = resolve_project_path(args.crp, project_root) if args.crp else paths["default_crp"]
    crps, crp_sha256 = load_crps_snapshot(crp_path)
    labels = crps[:, -1].astype(np.uint8)
    train_indices, test_indices = split_indices(labels, args.test_size, args.seed)
    features = ChallengeFeatureTransformer().transform(crps[:, :-1])
    initial = initialize_parameters(features.shape[1], args.rank, args.seed)
    backend = load_backend(args.backend, initial, args.learning_rate)

    output = resolve_project_path(args.output, project_root) if args.output else paths["runs_dir"] / (
        f"ecp-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{args.backend}-{uuid.uuid4().hex[:8]}"
    )
    # Creating a fresh directory is mandatory: no run or input file is overwritten.
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "record_version": 1, "model": "single_apuf_ecp", "backend": args.backend,
        "rank": args.rank, "branches": 1, "feature_count": features.shape[1],
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "test_size": args.test_size,
        "seeds": {"split": args.seed, "initialization": args.seed, "shuffle": args.seed + 1},
        "dtype": "float64", "device": "cpu",
        "optimizer": {"name": "Adam", "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
        "feature_transform": "APUF suffix products of (1 - 2*c), followed by bias 1",
        "prediction_rule": "logit >= 0 -> 1",
        "selection": "final epoch; test set evaluated only after training",
    }
    manifest = {
        "record_version": 1, "status": "running", "started_at": utc_now(),
        "data": {"path": str(crp_path), "sha256": crp_sha256,
                 "shape": list(crps.shape), "dtype": str(crps.dtype),
                 "provenance": "existing CRP; PUF weights and generation seeds are not inferred"},
        "environment": {"python": sys.version, "numpy": np.__version__,
                        args.backend: backend.version, "platform": platform.platform()},
        "source": source_record(project_root),
    }
    started = time.perf_counter()
    try:
        write_json(output / "config.json", config)
        write_json(output / "manifest.json", manifest)
        save_npz_with_csv(output / "split.npz", train=train_indices, test=test_indices)
        save_parameters(output / "initial.npz", initial)
        train_x, train_y = features[train_indices], labels[train_indices]

        # A fixed initial probe supports cross-framework checks before optimization.
        probe_rows = train_indices[:min(8, len(train_indices))]
        probe_loss, probe_gradients = backend.loss_and_gradients(features[probe_rows], labels[probe_rows])
        save_npz_with_csv(output / "initial_check.npz", rows=probe_rows,
                 logits=backend.logits(features[probe_rows]), loss=np.asarray(probe_loss),
                 **{f"grad_{name}": value for name, value in probe_gradients.items()})

        shuffle_rng = np.random.default_rng(args.seed + 1)
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("epoch", "train_bce", "train_accuracy"))
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                order = shuffle_rng.permutation(len(train_indices))
                for start in range(0, len(order), args.batch_size):
                    batch = order[start:start + args.batch_size]
                    loss = backend.train_batch(train_x[batch], train_y[batch])
                    if not np.isfinite(loss):
                        raise FloatingPointError(f"epoch {epoch} 出现非有限损失")
                metrics = binary_metrics(train_y, backend.logits(train_x))
                writer.writerow({"epoch": epoch, "train_bce": metrics["bce"], "train_accuracy": metrics["accuracy"]})
                handle.flush()
                if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                    print(f"{args.backend} epoch {epoch}/{args.epochs}: "
                          f"train BCE={metrics['bce']:.6f}, accuracy={metrics['accuracy']:.2%}", flush=True)

        final_parameters = backend.export_parameters()
        save_parameters(output / "model.npz", final_parameters)
        logits = backend.logits(features)
        lr_weights = fit_logistic_regression(train_x, train_y)
        lr_logits = features @ lr_weights
        save_npz_with_csv(output / "lr_model.npz", weights=lr_weights)
        metrics = {
            "train_count": len(train_indices), "test_count": len(test_indices),
            "ecp": {"train": binary_metrics(train_y, logits[train_indices]),
                    "test": binary_metrics(labels[test_indices], logits[test_indices])},
            "lr": {"train": binary_metrics(train_y, lr_logits[train_indices]),
                   "test": binary_metrics(labels[test_indices], lr_logits[test_indices]),
                   "config": {"max_iter": 5000, "learning_rate": 0.1, "regularization": 0.001}},
            "test_majority_baseline": majority_baseline(labels[test_indices]),
        }
        save_npz_with_csv(output / "predictions.npz", rows=test_indices, labels=labels[test_indices],
                 logits=logits[test_indices], probabilities=probabilities(logits[test_indices]),
                 predictions=(logits[test_indices] >= 0).astype(np.uint8))
        write_json(output / "metrics.json", metrics)
        manifest.update(status="completed", finished_at=utc_now(), elapsed_seconds=time.perf_counter() - started)
        manifest["artifact_sha256"] = {
            path.name: file_sha256(path) for path in sorted(output.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        write_json(output / "manifest.json", manifest)
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        finished_at=utc_now(), error=f"{type(error).__name__}: {error}")
        write_json(output / "manifest.json", manifest)
        raise

    print(f"测试准确率：{metrics['ecp']['test']['accuracy']:.2%}；"
          f"LR：{metrics['lr']['test']['accuracy']:.2%}；"
          f"多数类基线：{metrics['test_majority_baseline']:.2%}")
    print(f"实验目录：{output}")
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="单 APUF ECP 攻击及实验记录")
    parser.add_argument("--backend", required=True, choices=("torch", "tensorflow"))
    parser.add_argument("--config", help="项目 TOML 配置；也可通过 PUF_CONFIG 指定")
    parser.add_argument("--crp", help="现有 CRP NPY 文件；默认读取配置")
    parser.add_argument("--output", help="新建的运行目录；省略时在 runs/ 自动命名")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"错误：{error}\n")


if __name__ == "__main__":
    main()
