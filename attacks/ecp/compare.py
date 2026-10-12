"""Compare two backend runs through portable records; no framework is needed."""

import argparse
import json
from pathlib import Path

import numpy as np

from .common import load_parameters
from .records import load_record


def compare(first: str | Path, second: str | Path) -> dict:
    left, left_manifest, left_config, left_metrics = load_record(first)
    right, right_manifest, right_config, right_metrics = load_record(second)
    if {left_config["backend"], right_config["backend"]} != {"torch", "tensorflow"}:
        raise ValueError("必须提供一份 PyTorch 和一份 TensorFlow 实验")
    if left_manifest["data"]["sha256"] != right_manifest["data"]["sha256"]:
        raise ValueError("两次运行使用的 CRP 文件不同")
    left_protocol = {key: value for key, value in left_config.items() if key != "backend"}
    right_protocol = {key: value for key, value in right_config.items() if key != "backend"}
    if left_protocol != right_protocol:
        raise ValueError("两次运行的配置不同，不能作为相同协议下的对照")
    left_initial = load_parameters(left / "initial.npz")
    right_initial = load_parameters(right / "initial.npz")
    for name in left_initial:
        if not np.array_equal(left_initial[name], right_initial[name]):
            raise ValueError(f"初始参数不同：{name}")
    with np.load(left / "split.npz", allow_pickle=False) as a, np.load(right / "split.npz", allow_pickle=False) as b:
        if any(not np.array_equal(a[key], b[key]) for key in ("train", "test")):
            raise ValueError("两次运行的训练/测试索引不同")
    differences = {}
    with np.load(left / "initial_check.npz", allow_pickle=False) as a, np.load(right / "initial_check.npz", allow_pickle=False) as b:
        if not np.array_equal(a["rows"], b["rows"]):
            raise ValueError("初始数值检查使用了不同样本")
        for name in ("logits", "loss", "grad_factors", "grad_coefficients", "grad_bias"):
            if not np.allclose(a[name], b[name], rtol=1e-7, atol=1e-9):
                raise ValueError(f"初始 {name} 未通过跨框架数值对照")
            differences[name] = float(np.max(np.abs(a[name] - b[name])))
    return {
        "same_data_protocol_and_initialization": True,
        "initial_max_absolute_differences": differences,
        "test_accuracy": {left_config["backend"]: left_metrics["ecp"]["test"]["accuracy"],
                          right_config["backend"]: right_metrics["ecp"]["test"]["accuracy"]},
        "note": "两框架的原生 Adam 数值细节可能不同；最终模型参数无需逐位相等。",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="对照 PyTorch 与 TensorFlow 的 ECP 实验")
    parser.add_argument("--runs", nargs=2, required=True, metavar=("RUN_A", "RUN_B"))
    args = parser.parse_args(argv)
    try:
        result = compare(*args.runs)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f"错误：{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
