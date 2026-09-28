"""Challenge 到 APUF 特征向量的转换。"""

import numpy as np


class ChallengeFeatureTransformer:
    """将 n-bit Challenge 转换为 APUF 使用的特征向量。

    对于 n-bit Challenge，输出 n+1 维特征向量：

        phi = [phi_1, phi_2, ..., phi_n, 1]

    最后一个 1 是偏置项，因此 APUF 需要 n+1 个权重。
    """

    def transform(self, challenges: np.ndarray) -> np.ndarray:
        """将 0/1 Challenge 矩阵转换为特征矩阵。"""
        array = np.asarray(challenges)

        if array.ndim != 2:
            raise ValueError("challenges 必须是二维数组")

        if array.shape[1] < 1:
            raise ValueError("Challenge 至少需要包含 1 位")

        if not np.all((array == 0) | (array == 1)):
            raise ValueError("Challenge 只能包含 0 和 1")

        # 将 Challenge 从 0/1 转为 +1/-1：
        # 0 -> +1，1 -> -1。
        signs = 1 - 2 * array.astype(np.int8)

        # 从右向左计算累积乘积，得到前 n 个特征。
        stage_features = np.cumprod(
            signs[:, ::-1],
            axis=1,
        )[:, ::-1]

        # 增加偏置项，使特征维度从 n 变成 n+1。
        bias = np.ones(
            (array.shape[0], 1),
            dtype=np.int8,
        )

        return np.concatenate(
            [stage_features, bias],
            axis=1,
        ).astype(np.float64)
