"""Load saved Challenges, simulate APUF or XOR-APUF Responses, and save them."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

from challenge_data import load_challenges
from puf import APUF, XORAPUF
from project_config import load_project_config, resolve_project_path


def array_sha256(array: np.ndarray) -> str:
    """Return a content hash for a NumPy array's stored values."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    """Return a content hash for a source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从已保存的 Challenge 生成并保存 APUF 或 XOR-APUF Response"
    )
    parser.add_argument(
        "--challenge-path",
        default=None,
        help="输入的 Challenge .npy 文件",
    )
    parser.add_argument(
        "--response-path",
        default=None,
        help="输出的 Response .npy 文件",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="模型权重随机种子；固定后可复现相同仿真实体",
    )
    parser.add_argument(
        "--challenge-seed",
        type=int,
        default=None,
        help="生成 Challenge 时使用的种子（可选，用于实验记录）",
    )
    parser.add_argument("--config", help="项目 TOML 配置；也可通过 PUF_CONFIG 指定")
    parser.add_argument("--puf-type", choices=("apuf", "xor-apuf"), default="apuf", help="PUF 类型，默认 apuf")
    parser.add_argument("--xor-count", type=int, help="XOR-APUF 子模型数量，选择 xor-apuf 时必须指定")
    args = parser.parse_args()

    if args.puf_type == "xor-apuf" and (args.xor_count is None or args.xor_count < 1):
        parser.error("--puf-type xor-apuf 需要指定正整数 --xor-count")
    if args.puf_type == "apuf" and args.xor_count is not None:
        parser.error("--xor-count 仅用于 --puf-type xor-apuf")

    project_root, paths = load_project_config(args.config)
    challenge_path = resolve_project_path(args.challenge_path, project_root) if args.challenge_path else paths["default_challenges"]
    response_path = resolve_project_path(args.response_path, project_root) if args.response_path else paths["default_responses"]

    # 先读取之前生成并保存的 Challenge。
    challenges = load_challenges(challenge_path)

    # Challenge 的列数就是每组 Challenge 的位数 n。
    try:
        if args.puf_type == "xor-apuf":
            puf = XORAPUF(n=challenges.shape[1], xor_count=args.xor_count, seed=args.seed)
            weights = np.stack([component.weights for component in puf.components])
        else:
            puf = APUF(n=challenges.shape[1], seed=args.seed)
            weights = puf.weights
    except ValueError as error:
        parser.error(str(error))
    responses = puf.response(challenges)

    # 确保输出目录存在，再保存 NPY 和同名 CSV。
    response_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(response_path, responses)
    response_csv_path = response_path.with_suffix(".csv")
    np.savetxt(response_csv_path, responses, delimiter=",", fmt="%d")

    # 保存 PUF 真值权重和配置；这些文件不属于攻击器可见的 CRP 数据。
    weights_path = response_path.with_name("puf_weights.npy")
    weights_csv_path = weights_path.with_suffix(".csv")
    np.save(weights_path, weights)
    np.savetxt(weights_csv_path, weights, delimiter=",", fmt="%.17g")

    metadata_path = response_path.with_name(f"{response_path.stem}_metadata.json")
    source_files = (
        project_root / "generate_crp.py",
        project_root / "puf" / "apuf.py",
        project_root / "puf" / "features.py",
    )
    if args.puf_type == "xor-apuf":
        source_files += (project_root / "puf" / "xor_apuf.py",)
    metadata = {
        "record_version": 2,
        "puf": {
            "type": "XORAPUF" if args.puf_type == "xor-apuf" else "APUF",
            "challenge_bits": puf.n,
            "xor_count": args.xor_count if args.puf_type == "xor-apuf" else 1,
            "weight_seed": args.seed,
            "weight_count": int(weights.size),
            "weight_shape": list(weights.shape),
            "component_weight_count": puf.n + 1,
            "weight_distribution": {"name": "normal", "loc": 0.0, "scale": 1.0},
            "response_rule": (
                "XOR of component responses (delay >= 0 -> 1; delay < 0 -> 0)"
                if args.puf_type == "xor-apuf"
                else "delay >= 0 -> 1; delay < 0 -> 0"
            ),
            "weights_file": weights_path.name,
            "weights_sha256": array_sha256(weights),
        },
        "challenge": {
            "file": str(challenge_path),
            "shape": list(challenges.shape),
            "generation_seed": args.challenge_seed,
            "values_sha256": array_sha256(challenges),
        },
        "response": {
            "file": response_path.name,
            "shape": list(responses.shape),
            "values_sha256": array_sha256(responses),
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "source_sha256": {
            str(path.relative_to(project_root)): file_sha256(path)
            for path in source_files
        },
    }
    if args.puf_type == "xor-apuf":
        metadata["puf"]["component_seeds"] = [component.seed for component in puf.components]
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"PUF 类型：{metadata['puf']['type']}；Challenge 位数：{puf.n}；子模型数：{metadata['puf']['xor_count']}")
    print(f"Challenge 数组形状：{challenges.shape}")
    print(f"Response 数组形状：{responses.shape}")
    print(f"Response NPY 已保存到：{response_path}")
    print(f"Response CSV 已保存到：{response_csv_path}")
    print(f"PUF 权重 NPY 已保存到：{weights_path}")
    print(f"PUF 权重 CSV 已保存到：{weights_csv_path}")
    print(f"实验元数据已保存到：{metadata_path}")


if __name__ == "__main__":
    main()
