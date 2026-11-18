"""CRP 数据流程：仿真 Response，并把 Challenge 与 Response 合并成 CRP。

合并了原先的 ``generate_crp.py`` 与 ``merge_crp.py``，命令行用子命令区分：

* ``python -m pufkit.crp generate ...``  读取 Challenge，仿真 PUF，保存 Response/权重/元数据；
* ``python -m pufkit.crp merge ...``     按行配对 Challenge 与 Response，保存 CRP/元数据。

两处元数据都记录内容摘要与源文件摘要，使数据来源可核对。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np

from puf import APUF, XORAPUF
from pufkit.challenge import load_challenges
from pufkit.config import load_project_config, resolve_project_path
from pufkit.io import array_sha256, file_sha256, save_array_with_csv


def generate_responses(
    challenge_path: str | Path,
    response_path: str | Path,
    *,
    puf_type: str = "apuf",
    xor_count: int | None = None,
    seed: int = 42,
    challenge_seed: int | None = None,
    project_root: Path | None = None,
) -> dict:
    """仿真 PUF Response 并保存 Response、真值权重与元数据，返回元数据字典。"""
    challenge_path = Path(challenge_path)
    response_path = Path(response_path)
    project_root = Path(project_root) if project_root is not None else challenge_path.parent

    challenges = load_challenges(challenge_path)
    # Challenge 的列数就是每组 Challenge 的位数 n。
    if puf_type == "xor-apuf":
        puf = XORAPUF(n=challenges.shape[1], xor_count=xor_count, seed=seed)
        weights = np.stack([component.weights for component in puf.components])
    elif puf_type == "apuf":
        puf = APUF(n=challenges.shape[1], seed=seed)
        weights = puf.weights
    else:
        raise ValueError(f"未知 puf_type：{puf_type}")
    responses = puf.response(challenges)

    # 保存 Response 与同名 CSV。
    save_array_with_csv(response_path, responses, fmt="%d")

    # 保存 PUF 真值权重；这些文件不属于攻击器可见的 CRP 数据。
    weights_path = response_path.with_name("puf_weights.npy")
    save_array_with_csv(weights_path, weights, fmt="%.17g")

    metadata_path = response_path.with_name(f"{response_path.stem}_metadata.json")
    source_files = [
        project_root / "pufkit" / "crp.py",
        project_root / "pufkit" / "io.py",
        project_root / "puf" / "apuf.py",
        project_root / "puf" / "features.py",
    ]
    if puf_type == "xor-apuf":
        source_files.append(project_root / "puf" / "xor_apuf.py")
    metadata = {
        "record_version": 2,
        "puf": {
            "type": "XORAPUF" if puf_type == "xor-apuf" else "APUF",
            "challenge_bits": puf.n,
            "xor_count": xor_count if puf_type == "xor-apuf" else 1,
            "weight_seed": seed,
            "weight_count": int(weights.size),
            "weight_shape": list(weights.shape),
            "component_weight_count": puf.n + 1,
            "weight_distribution": {"name": "normal", "loc": 0.0, "scale": 1.0},
            "response_rule": (
                "XOR of component responses (delay >= 0 -> 1; delay < 0 -> 0)"
                if puf_type == "xor-apuf"
                else "delay >= 0 -> 1; delay < 0 -> 0"
            ),
            "weights_file": weights_path.name,
            "weights_sha256": array_sha256(weights),
        },
        "challenge": {
            "file": str(challenge_path),
            "shape": list(challenges.shape),
            "generation_seed": challenge_seed,
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
            for path in source_files if path.is_file()
        },
    }
    if puf_type == "xor-apuf":
        metadata["puf"]["component_seeds"] = [component.seed for component in puf.components]
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return metadata


def merge_crp(
    challenge_path: str | Path,
    response_path: str | Path,
    output_path: str | Path,
) -> np.ndarray:
    """Save a CRP matrix with Challenge bits first and Response last."""
    challenges = load_challenges(challenge_path)
    responses = np.asarray(np.load(Path(response_path), allow_pickle=False))
    response_hash = array_sha256(responses)

    # Accept either (samples,) or (samples, 1) for the one-bit Response.
    if responses.ndim == 2 and responses.shape[1] == 1:
        responses = responses[:, 0]
    if responses.ndim != 1:
        raise ValueError("Response 必须是一维数组，或形状为 (样本数, 1) 的二维数组")
    if not np.all((responses == 0) | (responses == 1)):
        raise ValueError("Response 只能包含 0 和 1")
    if challenges.shape[0] != responses.shape[0]:
        raise ValueError(
            f"Challenge 和 Response 数量不一致："
            f"{challenges.shape[0]} != {responses.shape[0]}"
        )

    # Use the response record to identify the PUF and prevent mixing input batches.
    response_file = Path(response_path)
    source_record_path = response_file.with_name(f"{response_file.stem}_metadata.json")
    source_record = None
    source_record_hash = None
    if source_record_path.is_file():
        source_record_bytes = source_record_path.read_bytes()
        source_record = json.loads(source_record_bytes)
        if source_record["challenge"]["values_sha256"] != array_sha256(challenges):
            raise ValueError("Challenge 与 Response 元数据不匹配，不能合并不同批次的数据")
        if source_record["response"]["values_sha256"] != response_hash:
            raise ValueError("Response 与其元数据不匹配，请检查输入文件")
        if source_record["puf"]["challenge_bits"] != challenges.shape[1]:
            raise ValueError("PUF 元数据中的 Challenge 位数与输入不一致")
        source_record_hash = file_sha256(source_record_path)

    # Each output row is [challenge bits..., response bit].
    crps = np.column_stack((challenges.astype(np.uint8), responses.astype(np.uint8)))
    output = Path(output_path)
    save_array_with_csv(output, crps, fmt="%d")
    metadata_path = output.with_name(f"{output.stem}_metadata.json")
    metadata = {
        "record_version": 1,
        "puf": source_record["puf"] if source_record is not None else {"type": "UNKNOWN"},
        "sample_count": int(crps.shape[0]),
        "challenge_bits": int(challenges.shape[1]),
        "response_bits": 1,
        "crp": {
            "file": str(output.resolve()),
            "shape": list(crps.shape),
            "dtype": str(crps.dtype),
            "row_format": "challenge bits first, one response bit last",
            "values_sha256": array_sha256(crps),
        },
        "challenge": {
            "file": str(Path(challenge_path).resolve()),
            "values_sha256": array_sha256(challenges),
            "generation_seed": source_record["challenge"].get("generation_seed") if source_record is not None else None,
        },
        "response": {"file": str(response_file.resolve()), "values_sha256": response_hash},
        "response_metadata": {
            "file": str(source_record_path.resolve()),
            "sha256": source_record_hash,
        } if source_record is not None else None,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return crps


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="项目 TOML 配置；也可通过 PUF_CONFIG 指定")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="CRP 数据流程：仿真 Response（generate）或合并 CRP（merge）",
        epilog=(
            "示例：\n"
            "  python -m pufkit.crp generate --puf-type apuf \\\n"
            "      --challenge-path data/apuf/<run>/challenges.npy \\\n"
            "      --response-path data/apuf/<run>/responses.npy \\\n"
            "      --challenge-seed 42 --seed 42\n"
            "  python -m pufkit.crp merge \\\n"
            "      --challenge-path data/apuf/<run>/challenges.npy \\\n"
            "      --response-path data/apuf/<run>/responses.npy \\\n"
            "      --output-path data/apuf/<run>/crps.npy"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="从已保存的 Challenge 仿真并保存 Response")
    generate.add_argument("--challenge-path", help="输入的 Challenge .npy 文件；默认读取配置")
    generate.add_argument("--response-path", help="输出的 Response .npy 文件；默认读取配置")
    generate.add_argument("--seed", type=int, default=42,
                          help="模型权重随机种子；固定后可复现相同仿真实体")
    generate.add_argument("--challenge-seed", type=int, default=None,
                          help="生成 Challenge 时使用的种子（可选，用于实验记录）")
    generate.add_argument("--puf-type", choices=("apuf", "xor-apuf"), default="apuf",
                          help="PUF 类型，默认 apuf")
    generate.add_argument("--xor-count", type=int, help="XOR-APUF 子模型数量，选择 xor-apuf 时必须指定")
    _add_common(generate)

    merge = subparsers.add_parser("merge", help="按行配对 Challenge 与 Response，保存 CRP")
    merge.add_argument("--challenge-path", help="输入 Challenge .npy 文件；默认读取项目配置")
    merge.add_argument("--response-path", help="输入 Response .npy 文件；默认读取项目配置")
    merge.add_argument("--output-path", help="输出 CRP .npy 文件；默认读取项目配置")
    _add_common(merge)

    args = parser.parse_args(argv)

    if args.command == "generate":
        if args.puf_type == "xor-apuf" and (args.xor_count is None or args.xor_count < 1):
            parser.error("--puf-type xor-apuf 需要指定正整数 --xor-count")
        if args.puf_type == "apuf" and args.xor_count is not None:
            parser.error("--xor-count 仅用于 --puf-type xor-apuf")

        project_root, paths = load_project_config(args.config)
        challenge_path = resolve_project_path(args.challenge_path, project_root) if args.challenge_path else paths["default_challenges"]
        response_path = resolve_project_path(args.response_path, project_root) if args.response_path else paths["default_responses"]
        try:
            metadata = generate_responses(
                challenge_path, response_path,
                puf_type=args.puf_type, xor_count=args.xor_count,
                seed=args.seed, challenge_seed=args.challenge_seed,
                project_root=project_root,
            )
        except ValueError as error:
            parser.exit(2, f"错误：{error}\n")

        weights_path = response_path.with_name("puf_weights.npy")
        print(f"PUF 类型：{metadata['puf']['type']}；Challenge 位数：{metadata['puf']['challenge_bits']}；"
              f"子模型数：{metadata['puf']['xor_count']}")
        print("Challenge 数组形状：", tuple(metadata["challenge"]["shape"]))
        print("Response 数组形状：", tuple(metadata["response"]["shape"]))
        print("Response NPY 已保存到：", response_path)
        print("Response CSV 已保存到：", response_path.with_suffix(".csv"))
        print("PUF 权重 NPY 已保存到：", weights_path)
        print("PUF 权重 CSV 已保存到：", weights_path.with_suffix(".csv"))
        print("实验元数据已保存到：", response_path.with_name(f"{response_path.stem}_metadata.json"))
        return

    project_root, paths = load_project_config(args.config)
    challenge_path = resolve_project_path(args.challenge_path, project_root) if args.challenge_path else paths["default_challenges"]
    response_path = resolve_project_path(args.response_path, project_root) if args.response_path else paths["default_responses"]
    output_path = resolve_project_path(args.output_path, project_root) if args.output_path else paths["default_crp"]

    try:
        crps = merge_crp(challenge_path, response_path, output_path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"错误：{error}\n")
    print(f"CRP 数组形状：{crps.shape}")
    print(f"每行格式：{crps.shape[1] - 1} 位 Challenge + 1 位 Response")
    print(f"CRP NPY 已保存到：{output_path}")
    print(f"CRP CSV 已保存到：{output_path.with_suffix('.csv')}")
    print(f"CRP 元数据已保存到：{output_path.with_name(f'{output_path.stem}_metadata.json')}")


if __name__ == "__main__":
    main()