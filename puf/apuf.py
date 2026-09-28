"""Arbiter PUF（APUF）的基础软件模型。"""

import numpy as np

from .base import BasePUF
from .features import ChallengeFeatureTransformer


class APUF(BasePUF):
    """使用线性加性延迟模型模拟一个固定的 APUF 实体。"""

    def __init__(self, n: int, seed: int = 42) -> None:
        """创建一个 APUF。

        参数：
            n: Challenge 位数；APUF 会生成 n+1 个内部权重。
            seed: 权重随机种子；相同参数会得到相同的 PUF 实体。
        """
        # 调用 BasePUF 的初始化逻辑，检查并保存 self.n。
        super().__init__(n)

        self.seed = seed
        self.feature_transformer = ChallengeFeatureTransformer()

        # 权重只在创建 PUF 实体时生成一次，后续 response 会重复使用。
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(
            loc=0.0,
            scale=1.0,
            size=self.n + 1,
        )

    def response(self, challenges: np.ndarray) -> np.ndarray:
        """输入 Challenge 矩阵，返回对应的 0/1 Response。"""
        # 使用基类统一检查 Challenge 的形状和取值。
        binary_challenges = self.validate_challenges(challenges)

        # Challenge -> 特征矩阵 phi，形状为 (样本数, n+1)。
        features = self.feature_transformer.transform(binary_challenges)

        # 线性加性延迟模型：delay = phi · w。
        delays = features @ self.weights

        # 延迟非负记为 1，否则记为 0。
        return (delays >= 0.0).astype(np.uint8)
