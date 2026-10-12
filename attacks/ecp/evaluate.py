"""Reload a portable ECP model and verify its recorded evaluation, without training."""

import argparse
import json
from pathlib import Path

import numpy as np

from attacks.common import load_crps_snapshot
from puf.features import ChallengeFeatureTransformer
from .common import (
    binary_metrics, load_backend, load_parameters,
    numpy_logits, probabilities, validate_parameters, validate_split,
)
from .records import load_record


def evaluate(directory: str | Path, crp: str | Path | None = None, backend: str = "numpy") -> dict:
    directory, manifest, config, saved_metrics = load_record(directory)
    crp_path = Path(crp) if crp is not None else Path(manifest["data"]["path"])
    crps, crp_sha256 = load_crps_snapshot(crp_path)
    if crp_sha256 != manifest["data"]["sha256"]:
        raise ValueError("CRP 文件摘要与训练记录不同，拒绝复算")
    features = ChallengeFeatureTransformer().transform(crps[:, :-1])
    labels = crps[:, -1]
    parameters = validate_parameters(load_parameters(directory / "model.npz"), features.shape[1])
    if parameters["factors"].shape != (config["rank"], 1, config["feature_count"]):
        raise ValueError("模型形状与保存的配置不一致")
    with np.load(directory / "split.npz", allow_pickle=False) as archive:
        train, test = archive["train"], archive["test"]
    validate_split(train, test, len(crps))
    if backend == "numpy":
        logits = numpy_logits(features, parameters)
    else:
        logits = load_backend(backend, parameters).logits(features)
    recomputed = {
        "train": binary_metrics(labels[train], logits[train]),
        "test": binary_metrics(labels[test], logits[test]),
    }
    for partition in ("train", "test"):
        for name, value in recomputed[partition].items():
            if not np.isclose(value, saved_metrics["ecp"][partition][name], rtol=1e-9, atol=1e-10):
                raise ValueError(f"重载后的 {partition} {name} 与保存结果不同")
    with np.load(directory / "predictions.npz", allow_pickle=False) as predictions:
        for name, expected in (("rows", test), ("labels", labels[test]),
                               ("predictions", (logits[test] >= 0).astype(np.uint8))):
            if not np.array_equal(predictions[name], expected):
                raise ValueError(f"重载后的预测 {name} 与保存结果不同")
        for name, expected in (("logits", logits[test]), ("probabilities", probabilities(logits[test]))):
            if not np.allclose(predictions[name], expected, rtol=1e-9, atol=1e-10):
                raise ValueError(f"重载后的 {name} 与保存结果不同")
    return {"backend": backend, "run": str(directory), "matches_saved_metrics": True,
            "ecp": recomputed}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="重载 ECP 模型并复算已保存的评估结果")
    parser.add_argument("--run", required=True, help="已完成的实验目录")
    parser.add_argument("--crp", help="数据移动后可指定新路径；文件内容必须与原记录相同")
    parser.add_argument("--backend", choices=("numpy", "torch", "tensorflow"), default="numpy")
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.run, args.crp, args.backend)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f"错误：{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
