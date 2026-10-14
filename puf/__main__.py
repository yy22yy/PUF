"""创建 PUF 模型并在终端显示权重。"""

import argparse

import numpy as np

from .apuf import APUF
from .xor_apuf import XORAPUF


def main() -> None:
    parser = argparse.ArgumentParser(description="初始化 APUF 或 XOR-APUF 并显示权重")
    parser.add_argument("--n", type=int, default=32, help="Challenge 位数，默认 32")
    parser.add_argument("--seed", type=int, default=42, help="模型权重种子，默认 42")
    parser.add_argument("--xor-count", type=int, help="XOR-APUF 子模型数量；省略时创建单 APUF")
    args = parser.parse_args()

    try:
        if args.xor_count is None:
            model = APUF(n=args.n, seed=args.seed)
            weights = model.weights
            name = "APUF"
        else:
            model = XORAPUF(n=args.n, xor_count=args.xor_count, seed=args.seed)
            weights = np.stack([component.weights for component in model.components])
            name = f"{args.xor_count}-XOR APUF"
    except ValueError as error:
        parser.error(str(error))

    print(f"模型：{args.n} 位 {name}；权重种子：{args.seed}")
    print("权重：")
    print(weights)
    print(f"权重形状：{weights.shape}")
    print("模型已初始化；尚未输入 Challenge 或计算 Response，未保存文件。")


if __name__ == "__main__":
    main()
