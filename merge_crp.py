"""Combine Challenge and Response arrays into one CRP matrix."""

import argparse
from pathlib import Path

import numpy as np

from challenge_data import load_challenges


def merge_crp(
    challenge_path: str | Path,
    response_path: str | Path,
    output_path: str | Path,
) -> np.ndarray:
    """Save a CRP matrix with Challenge bits first and Response last."""
    challenges = load_challenges(challenge_path)
    responses = np.asarray(np.load(Path(response_path), allow_pickle=False))

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

    # Each output row is [challenge bits..., response bit].
    crps = np.column_stack(
        (challenges.astype(np.uint8), responses.astype(np.uint8))
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, crps)
    csv_output = output.with_suffix(".csv")
    np.savetxt(csv_output, crps, delimiter=",", fmt="%d")
    return crps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 Challenge 和 Response 合并成一个 CRP NPY 文件"
    )
    parser.add_argument("--challenge-path", required=True, help="输入 Challenge .npy 文件")
    parser.add_argument("--response-path", required=True, help="输入 Response .npy 文件")
    parser.add_argument("--output-path", default="data/crps.npy", help="输出 CRP .npy 文件")
    args = parser.parse_args()

    crps = merge_crp(
        args.challenge_path,
        args.response_path,
        args.output_path,
    )
    print(f"CRP 数组形状：{crps.shape}")
    print(f"每行格式：{crps.shape[1] - 1} 位 Challenge + 1 位 Response")
    print(f"CRP NPY 已保存到：{args.output_path}")
    print(f"CRP CSV 已保存到：{Path(args.output_path).with_suffix('.csv')}")


if __name__ == "__main__":
    main()
