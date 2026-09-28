"""生成用于 PUF 实验的 Challenge（挑战）数据。"""

# NumPy 用于高效处理数组，并提供向量、矩阵和批量数值计算功能。
# 本项目后续会进行 PUF 的向量化计算，因此使用 NumPy 比逐个处理 Python
# 列表更合适。Python 标准库中的 random 模块也可以生成随机数，但更适合
# 一般性的单个数值或简单随机操作，并不是所有场景都必须使用 NumPy。
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
        seed: 随机种子。固定为同一个整数时，每次生成的结果相同，
            便于实验复现和调试；设为 None 时，不固定生成结果。

    返回：
        形状为 ``(num_challenges, challenge_length)`` 的 NumPy 数组，
        每个元素都是 0 或 1。
    """
    # default_rng 是 NumPy 推荐的新式随机数生成器接口。
    # 它将随机状态封装在 rng 对象中，避免依赖全局随机状态。
    rng = np.random.default_rng(seed)

    # integers 的下界 0 包含在结果中，上界 2 不包含在结果中，
    # 因此生成的元素只有 0 和 1；size 指定输出矩阵的行数和列数。
    return rng.integers(
        0,
        2,
        size=(num_challenges, challenge_length),
    )


# 只有直接运行本文件时才执行下面的示例代码。
# 当其他文件 import 本模块时，不会自动生成数据，便于复用函数。
if __name__ == "__main__":
    challenges = generate_challenges()
    print(challenges)
