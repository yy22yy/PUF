"""Train and evaluate a logistic-regression model on single-APUF CRPs."""

import argparse

import numpy as np

from attacks.common import accuracy, load_challenge_responses, load_crps, split_by_response
from puf.features import ChallengeFeatureTransformer
from project_config import load_project_config, resolve_project_path


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
        default=None,
        help="输入的 CRP .npy 文件，每行最后一列为 Response",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="测试集比例，默认 0.25",
    )
    parser.add_argument("--challenge-path", help="单独保存的 Challenge NPY 文件，需同时提供 --response-path")
    parser.add_argument("--response-path", help="单独保存的 Response NPY 文件，需同时提供 --challenge-path")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="训练/测试划分随机种子",
    )
    parser.add_argument("--config", help="项目 TOML 配置；也可通过 PUF_CONFIG 指定")
    args = parser.parse_args()

    separate_input = args.challenge_path is not None or args.response_path is not None
    if separate_input and (args.challenge_path is None or args.response_path is None):
        parser.error("--challenge-path 和 --response-path 必须同时提供")
    if separate_input and args.crp_path is not None:
        parser.error("分开的 Challenge/Response 输入不能与 --crp-path 同时使用")

    project_root, paths = load_project_config(args.config)

    try:
        if separate_input:
            challenges, responses = load_challenge_responses(
                resolve_project_path(args.challenge_path, project_root),
                resolve_project_path(args.response_path, project_root),
            )
        else:
            # CRP 每行格式为 [Challenge 的各位..., Response]。
            crp_path = resolve_project_path(args.crp_path, project_root) if args.crp_path else paths["default_crp"]
            crps = load_crps(crp_path)
            challenges = crps[:, :-1]
            responses = crps[:, -1].astype(np.uint8)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not 0.0 < args.test_size < 1.0:
        parser.error("--test-size 必须大于 0 且小于 1")

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

    print(f"CRP 总数：{len(challenges)}")
    print(f"训练样本：{len(y_train)}；测试样本：{len(y_test)}")
    print(f"训练准确率：{train_accuracy:.2%}")
    print(f"测试准确率：{test_accuracy:.2%}")


if __name__ == "__main__":
    main()
