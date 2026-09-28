"""使用面向对象方式生成 PUF Challenge。"""

import numpy as np


class ChallengeGenerator:
    """封装 Challenge 生成所需的配置和随机数生成器。"""

    def __init__(
        self,
        challenge_length: int = 64,
        seed: int | None = 42,
    ) -> None:
        """初始化生成器。

        参数：
            challenge_length: 每组 Challenge 的位数。
            seed: 随机种子。固定种子可复现实验；设为 None 时不固定结果。
        """
        self.challenge_length = challenge_length

        # 将随机数生成器保存为对象属性，后续可以连续生成多批数据，
        # 并保持同一个对象内部的随机状态。
        self.rng = np.random.default_rng(seed)

    def generate(self, num_challenges: int = 100) -> np.ndarray:
        """生成一批由 0 和 1 组成的 Challenge。

        参数：
            num_challenges: 要生成的 Challenge 数量，即矩阵的行数。

        返回：
            形状为 ``(num_challenges, challenge_length)`` 的 NumPy 数组。
        """
        # 上界 2 不包含在结果中，因此生成值只可能是 0 或 1。
        return self.rng.integers(
            0,
            2,
            size=(num_challenges, self.challenge_length),
        )


if __name__ == "__main__":
    # 直接运行本文件时，先创建对象，再调用对象的方法生成数据。
    generator = ChallengeGenerator(challenge_length=64, seed=42)
    challenges = generator.generate(num_challenges=100)
    print(challenges)
