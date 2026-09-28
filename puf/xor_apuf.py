"""由多个 APUF 组合而成的 XOR-APUF。"""

import numpy as np

from .apuf import APUF
from .base import BasePUF


class XORAPUF(BasePUF):
    """将多个独立 APUF 的 Response 进行逐位 XOR。"""

    def __init__(
        self,
        n: int,
        xor_count: int = 2,
        seed: int = 42,
    ) -> None:
        """创建一个 XOR-APUF。

        参数：
            n: 所有内部 APUF 使用的 Challenge 位数。
            xor_count: 内部 APUF 的数量。
            seed: 用于生成各个内部 APUF 权重的总随机种子。
        """
        super().__init__(n)

        if not isinstance(xor_count, int) or isinstance(xor_count, bool):
            raise ValueError("xor_count 必须是正整数")
        if xor_count < 1:
            raise ValueError("xor_count 必须是正整数")

        self.xor_count = xor_count
        self.seed = seed

        # 先生成不同的子种子，再创建多个独立的 APUF。
        seed_generator = np.random.default_rng(seed)
        component_seeds = seed_generator.integers(
            0,
            np.iinfo(np.uint32).max,
            size=xor_count,
            dtype=np.uint32,
        )

        self.components = tuple(
            APUF(n=self.n, seed=int(component_seed))
            for component_seed in component_seeds
        )

    def response(self, challenges: np.ndarray) -> np.ndarray:
        """输入 Challenge，返回多个 APUF Response 的 XOR 结果。"""
        # 先由基类统一检查输入格式。
        binary_challenges = self.validate_challenges(challenges)

        # 每一行是一个内部 APUF 对所有 Challenge 的 Response。
        component_responses = np.stack(
            [component.response(binary_challenges) for component in self.components],
            axis=0,
        )

        # 沿内部 APUF 这一维进行逐位 XOR。
        return np.bitwise_xor.reduce(component_responses, axis=0)
