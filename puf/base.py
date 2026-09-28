"""所有 PUF 模型共用的基础类。"""

from abc import ABC, abstractmethod

import numpy as np


class BasePUF(ABC):
    """定义 PUF 模型的公共属性和公共接口。"""

    def __init__(self, n: int) -> None:
        """初始化 PUF。

        参数：
            n: Challenge 的位数。一个 PUF 对象只接受 n-bit Challenge。
        """
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError("n 必须是正整数")

        # 保存 Challenge 位数，子类可以通过 self.n 使用它。
        self.n = n

    def validate_challenges(self, challenges: np.ndarray) -> np.ndarray:
        """检查并统一 Challenge 的格式。

        返回形状为 (样本数, n)、元素只包含 0/1 的 uint8 数组。
        """
        array = np.asarray(challenges)

        if array.ndim != 2:
            raise ValueError("challenges 必须是二维数组")

        if array.shape[1] != self.n:
            raise ValueError(
                f"Challenge 必须包含 {self.n} 位，实际为 {array.shape[1]} 位"
            )

        if not np.all((array == 0) | (array == 1)):
            raise ValueError("Challenge 只能包含 0 和 1")

        return array.astype(np.uint8, copy=False)

    @abstractmethod
    def response(self, challenges: np.ndarray) -> np.ndarray:
        """根据 Challenge 生成 Response；具体 PUF 子类必须实现。"""
        raise NotImplementedError
