"""Combine Challenge and Response arrays into one CRP matrix."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from challenge_data import load_challenges
from project_config import load_project_config, resolve_project_path


def array_sha256(array: np.ndarray) -> str:
    """Hash the stored array values to identify the input data."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


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
        source_record_hash = hashlib.sha256(source_record_bytes).hexdigest()

    # Each output row is [challenge bits..., response bit].
    crps = np.column_stack(
        (challenges.astype(np.uint8), responses.astype(np.uint8))
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, crps)
    csv_output = output.with_suffix(".csv")
    np.savetxt(csv_output, crps, delimiter=",", fmt="%d")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 Challenge 和 Response 合并成一个 CRP NPY 文件"
    )
    parser.add_argument("--challenge-path", help="输入 Challenge .npy 文件；默认读取项目配置")
    parser.add_argument("--response-path", help="输入 Response .npy 文件；默认读取项目配置")
    parser.add_argument("--output-path", help="输出 CRP .npy 文件；默认读取项目配置")
    parser.add_argument("--config", help="项目 TOML 配置；也可通过 PUF_CONFIG 指定")
    args = parser.parse_args()

    project_root, paths = load_project_config(args.config)
    challenge_path = resolve_project_path(args.challenge_path, project_root) if args.challenge_path else paths["default_challenges"]
    response_path = resolve_project_path(args.response_path, project_root) if args.response_path else paths["default_responses"]
    output_path = resolve_project_path(args.output_path, project_root) if args.output_path else paths["default_crp"]

    try:
        crps = merge_crp(challenge_path, response_path, output_path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"CRP 数组形状：{crps.shape}")
    print(f"每行格式：{crps.shape[1] - 1} 位 Challenge + 1 位 Response")
    print(f"CRP NPY 已保存到：{output_path}")
    print(f"CRP CSV 已保存到：{output_path.with_suffix('.csv')}")
    print(f"CRP 元数据已保存到：{output_path.with_name(f'{output_path.stem}_metadata.json')}")


if __name__ == "__main__":
    main()
