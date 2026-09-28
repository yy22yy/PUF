"""分步骤展示 APUF 从 Challenge 到 Response 的计算过程。"""

import argparse

import numpy as np

from .apuf import APUF


def main() -> None:
    """读取命令行参数，并逐步展示 APUF 计算过程。"""
    parser = argparse.ArgumentParser(
        description="逐步展示 APUF 的 Challenge 到 Response 过程"
    )
    parser.add_argument("--n", type=int, default=4, help="Challenge 位数；默认值：4")
    parser.add_argument("--num", type=int, default=2, help="Challenge 数量；默认值：2")
    parser.add_argument("--seed", type=int, default=42, help="PUF 权重种子；默认值：42")
    parser.add_argument(
        "--challenge-seed",
        type=int,
        default=7,
        help="Challenge 随机种子；默认值：7",
    )
    args = parser.parse_args()

    if args.n < 1 or args.num < 1:
        parser.error("--n 和 --num 必须是正整数")

    # 使用固定种子生成一批 Challenge，便于重复观察同样的过程。
    challenge_rng = np.random.default_rng(args.challenge_seed)
    challenges = challenge_rng.integers(
        0,
        2,
        size=(args.num, args.n),
        dtype=np.uint8,
    )

    # 创建一个固定的 APUF 实体，并生成一次内部权重。
    puf = APUF(n=args.n, seed=args.seed)
    features = puf.feature_transformer.transform(challenges)
    delays = features @ puf.weights
    responses = puf.response(challenges)

    print("步骤 1：输入 Challenge")
    print(challenges)
    print("形状：", challenges.shape)

    print("\n步骤 2：生成的 PUF 权重")
    print(puf.weights)
    print("形状：", puf.weights.shape)

    print("\n步骤 3：Challenge 转换后的特征矩阵 phi")
    print(features)
    print("形状：", features.shape)

    print("\n步骤 4：计算延迟 delay = phi @ weights")
    print(delays)

    print("\n步骤 5：根据延迟生成 Response")
    print("delay >= 0 -> 1；delay < 0 -> 0")
    print(responses)


if __name__ == "__main__":
    main()
