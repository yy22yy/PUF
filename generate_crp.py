"""Load saved Challenges, simulate APUF Responses, and save them."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

from challenge_data import load_challenges
from puf import APUF


def array_sha256(array: np.ndarray) -> str:
    """Return a content hash for a NumPy array's stored values."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    """Return a content hash for a source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从已保存的 Challenge 生成并保存 APUF Response"
    )
    parser.add_argument(
        "--challenge-path",
        default="data/challenges.npy",
        help="输入的 Challenge .npy 文件",
    )
    parser.add_argument(
        "--response-path",
        default="data/responses.npy",
        help="输出的 Response .npy 文件",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="APUF 权重随机种子；固定后可复现相同仿真实体",
    )
    parser.add_argument(
        "--challenge-seed",
        type=int,
        default=None,
        help="生成 Challenge 时使用的种子（可选，用于实验记录）",
    )
    args = parser.parse_args()

    # 先读取之前生成并保存的 Challenge。
    challenges = load_challenges(args.challenge_path)

    # Challenge 的列数就是每组 Challenge 的位数 n。
    puf = APUF(n=challenges.shape[1], seed=args.seed)
    responses = puf.response(challenges)

    # 确保输出目录存在，再保存 NPY 和同名 CSV。
    response_path = Path(args.response_path)
    response_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(response_path, responses)
    response_csv_path = response_path.with_suffix(".csv")
    np.savetxt(response_csv_path, responses, delimiter=",", fmt="%d")

    # 保存 PUF 真值权重和配置；这些文件不属于攻击器可见的 CRP 数据。
    weights_path = response_path.with_name(f"{response_path.stem}_puf_weights.npy")
    weights_csv_path = weights_path.with_suffix(".csv")
    np.save(weights_path, puf.weights)
    np.savetxt(weights_csv_path, puf.weights, delimiter=",", fmt="%.17g")

    metadata_path = response_path.with_name(f"{response_path.stem}_metadata.json")
    project_root = Path(__file__).resolve().parent
    source_files = (
        project_root / "generate_crp.py",
        project_root / "puf" / "apuf.py",
        project_root / "puf" / "features.py",
    )
    metadata = {
        "record_version": 1,
        "puf": {
            "type": "APUF",
            "challenge_bits": puf.n,
            "weight_seed": args.seed,
            "weight_count": int(puf.weights.size),
            "weight_distribution": {"name": "normal", "loc": 0.0, "scale": 1.0},
            "response_rule": "delay >= 0 -> 1; delay < 0 -> 0",
            "weights_file": weights_path.name,
            "weights_sha256": array_sha256(puf.weights),
        },
        "challenge": {
            "file": str(Path(args.challenge_path)),
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
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Challenge 数组形状：{challenges.shape}")
    print(f"Response 数组形状：{responses.shape}")
    print(f"Response NPY 已保存到：{response_path}")
    print(f"Response CSV 已保存到：{response_csv_path}")
    print(f"PUF 权重 NPY 已保存到：{weights_path}")
    print(f"PUF 权重 CSV 已保存到：{weights_csv_path}")
    print(f"实验元数据已保存到：{metadata_path}")


if __name__ == "__main__":
    main()
