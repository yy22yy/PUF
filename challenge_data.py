"""Challenge 的生成、保存和读取示例。

这个文件负责数据持久化：

* ``save_challenges``：把内存中的 Challenge 保存到 ``.npy`` 文件。
* ``load_challenges``：从 ``.npy`` 文件读取之前保存的 Challenge。
* ``generate_and_save_challenges``：生成一批 Challenge 并立即保存。
"""

import argparse
from pathlib import Path

import numpy as np

from challenge import generate_challenges


def validate_challenges(challenges: np.ndarray) -> np.ndarray:
    """检查并返回合法的二维 Challenge 数组。"""
    array = np.asarray(challenges)
    if array.ndim != 2:
        raise ValueError("challenges 必须是二维数组")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("challenges 只能包含 0 和 1")
    return array


def prepare_output_path(path: str | Path) -> Path:
    """转换输出路径，并创建所需的父目录。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def save_challenges(challenges: np.ndarray, path: str | Path) -> None:
    """保存 Challenge 数组。

    参数：
        challenges: 只包含 0 和 1 的二维 Challenge 数组。
        path: 输出文件路径，例如 ``data/challenges.npy``。
    """
    array = validate_challenges(challenges)
    output_path = prepare_output_path(path)

    np.save(output_path, array.astype(np.uint8))


def save_challenges_csv(challenges: np.ndarray, path: str | Path) -> None:
    """将 Challenge 数组保存为方便人工查看的 CSV 文件。"""
    array = validate_challenges(challenges)
    output_path = prepare_output_path(path)

    np.savetxt(output_path, array.astype(np.uint8), delimiter=",", fmt="%d")


def load_challenges(path: str | Path) -> np.ndarray:
    """从 ``.npy`` 文件读取 Challenge 数组。"""
    array = validate_challenges(np.load(Path(path), allow_pickle=False))
    return array.astype(np.uint8)


def generate_and_save_challenges(
    path: str | Path,
    num_challenges: int = 100,
    challenge_length: int = 64,
    seed: int | None = 42,
) -> np.ndarray:
    """生成 Challenge、保存到文件，并返回生成的数组。"""
    challenges = generate_challenges(num_challenges, challenge_length, seed)
    save_challenges(challenges, path)
    save_challenges_csv(challenges, Path(path).with_suffix(".csv"))
    return challenges


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "生成、保存并读取 Challenge 数组；\n"
            "每次同时生成 .npy 文件和同名 .csv 文件。"
        ),
        epilog=(
            "示例：\n"
            "  python challenge_data.py --num 3 --c_bit 4 --seed 42 --overwrite\n"
            "  表示生成 3 组、每组 4 位的 Challenge，并覆盖已有文件。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--num",
        type=int,
        default=100,
        metavar="N",
        help="生成的 Challenge 组数；默认值：100",
    )
    parser.add_argument(
        "--c_bit",
        type=int,
        default=64,
        metavar="BITS",
        help="每组 Challenge 的位数；默认值：64",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="随机种子；相同种子会生成相同结果；默认值：42",
    )
    parser.add_argument(
        "--path",
        default="data/challenges.npy",
        help=(
            "NPY 文件保存路径；程序会在相同目录自动生成同名 CSV；"
            "默认值：data/challenges.npy"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已经存在的 .npy 文件",
    )
    args = parser.parse_args()

    if args.num < 1:
        parser.error("--num 必须是正整数")
    if args.c_bit < 1:
        parser.error("--c_bit 必须是正整数")

    file_path = Path(args.path)
    csv_path = file_path.with_suffix(".csv")
    existing_paths = [path for path in (file_path, csv_path) if path.exists()]
    if existing_paths and not args.overwrite:
        parser.error(
            f"文件已存在：{', '.join(map(str, existing_paths))}；"
            "如需覆盖，请增加 --overwrite"
        )

    challenges = generate_challenges(
        num_challenges=args.num,
        challenge_length=args.c_bit,
        seed=args.seed,
    )
    save_challenges(challenges, file_path)
    save_challenges_csv(challenges, csv_path)
    loaded = load_challenges(file_path)
    print("已保存并重新读取 Challenge：", file_path)
    print("已生成 CSV 文件：", csv_path)
    print("数组形状：", loaded.shape)
