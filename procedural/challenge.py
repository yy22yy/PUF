"""使用面向过程方式生成 PUF Challenge。"""

# NumPy 用于向量、矩阵和批量数值计算。
import numpy as np

# 42 是随机种子：固定种子可以让每次运行生成相同的数据，便于复现和调试。
rng = np.random.default_rng(42)

# 生成 100 组、每组 64 位的二进制 Challenge。
# 0 是包含在结果中的下界，2 是不包含在结果中的上界，因此元素只有 0 和 1。
challenges = rng.integers(
    0,
    2,
    size=(100, 64),
)
