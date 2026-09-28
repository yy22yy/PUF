"""使用面向过程方式生成 PUF Challenge。"""

import numpy as np


def generate_challenges(
    num_challenges: int = 100,
    challenge_length: int = 64,
    seed: int | None = 42,
) -> np.ndarray:
    """生成由 0 和 1 组成的 Challenge 矩阵。

    参数：
        num_challenges: 要生成的 Challenge 数量，即矩阵的行数。
        challenge_length: 每组 Challenge 的位数，即矩阵的列数。
        seed: 随机种子。固定为同一个整数时，结果可复现；设为 None
            时不固定结果。

    返回：
        形状为 ``(num_challenges, challenge_length)`` 的 NumPy 数组。
    """
    # 使用局部随机数生成器，避免修改 NumPy 的全局随机状态。
    rng = np.random.default_rng(seed)

    # 下界 0 包含在结果中，上界 2 不包含在结果中，所以元素只有 0 和 1。
    return rng.integers(
        0,
        2,
        size=(num_challenges, challenge_length),
    )


if __name__ == "__main__":
    # 直接运行本文件时生成默认规模的 Challenge。
    challenges = generate_challenges()
    print(challenges)
