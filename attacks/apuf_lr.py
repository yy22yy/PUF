"""Train and evaluate a logistic-regression model on single-APUF CRPs."""

import argparse

import numpy as np

from attacks.common import accuracy, load_crps, majority_baseline, split_by_response
from puf.features import ChallengeFeatureTransformer


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Compute sigmoid values while avoiding exponential overflow."""
    clipped = np.clip(values, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 5000,
    learning_rate: float = 0.1,
    regularization: float = 1e-3,
) -> np.ndarray:
    """Fit binary logistic regression by batch gradient descent."""
    weights = np.zeros(features.shape[1], dtype=np.float64)
    penalty = np.ones_like(weights)
    penalty[-1] = 0.0  # The last feature is the explicit bias column.

    for _ in range(max_iter):
        probabilities = sigmoid(features @ weights)
        gradient = features.T @ (probabilities - labels) / labels.size
        gradient += regularization * penalty * weights
        update = learning_rate * gradient
        weights -= update
        if np.linalg.norm(update) < 1e-8:
            break

    return weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 Logistic Regression 对单个 APUF 进行建模攻击"
    )
    parser.add_argument(
        "--crp-path",
        default="data/crps.npy",
        help="输入的 CRP .npy 文件，每行最后一列为 Response",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="测试集比例，默认 0.25",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="训练/测试划分随机种子",
    )
    args = parser.parse_args()

    # CRP 每行格式为 [Challenge 的各位..., Response]。
    try:
        crps = load_crps(args.crp_path)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not 0.0 < args.test_size < 1.0:
        parser.error("--test-size 必须大于 0 且小于 1")

    challenges = crps[:, :-1]
    responses = crps[:, -1].astype(np.uint8)
    labels, counts = np.unique(responses, return_counts=True)
    if labels.size != 2 or np.min(counts) < 2:
        parser.error("CRP 中的 Response 必须同时包含 0 和 1，且每类至少有 2 条样本")

    # 保留一部分从未参与训练的 CRP，用于评估模型泛化能力。
    train_challenges, test_challenges, y_train, y_test = split_by_response(
        challenges,
        responses,
        args.test_size,
        args.seed,
    )

    # 使用 APUF 的阶段特征，而不是直接把原始 Challenge 位交给模型。
    transformer = ChallengeFeatureTransformer()
    x_train = transformer.transform(train_challenges)
    x_test = transformer.transform(test_challenges)

    # 特征末尾已经有偏置列，因此优化器不再额外添加 intercept。
    weights = fit_logistic_regression(x_train, y_train)
    train_predictions = (sigmoid(x_train @ weights) >= 0.5).astype(np.uint8)
    test_predictions = (sigmoid(x_test @ weights) >= 0.5).astype(np.uint8)
    train_accuracy = accuracy(y_train, train_predictions)
    test_accuracy = accuracy(y_test, test_predictions)
    baseline_accuracy = majority_baseline(y_test)

    print(f"CRP 总数：{len(crps)}")
    print(f"训练样本：{len(y_train)}；测试样本：{len(y_test)}")
    print(f"训练准确率：{train_accuracy:.2%}")
    print(f"测试准确率：{test_accuracy:.2%}")
    print(f"测试集多数类基线：{baseline_accuracy:.2%}")


if __name__ == "__main__":
    main()
