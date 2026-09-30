"""Shared NumPy training code for neural-network APUF modeling attacks."""

import argparse

import numpy as np

from attacks.common import accuracy, load_crps, majority_baseline, split_by_response


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -500.0, 500.0)))


class BinaryMLP:
    """A fully connected ReLU network with a one-unit sigmoid output."""

    def __init__(
        self,
        input_size: int,
        hidden_layers: tuple[int, ...],
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        layer_sizes = (input_size, *hidden_layers, 1)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He initialization works well with ReLU hidden layers.
            self.weights.append(
                rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
            )
            self.biases.append(np.zeros((1, fan_out), dtype=np.float64))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        activations = features
        for index, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            values = activations @ weights + bias
            activations = (
                sigmoid(values) if index == len(self.weights) - 1 else relu(values)
            )
        return activations[:, 0]

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        weight_moments = [np.zeros_like(weight) for weight in self.weights]
        weight_variances = [np.zeros_like(weight) for weight in self.weights]
        bias_moments = [np.zeros_like(bias) for bias in self.biases]
        bias_variances = [np.zeros_like(bias) for bias in self.biases]
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        step = 0

        for _ in range(epochs):
            order = rng.permutation(features.shape[0])
            for start in range(0, features.shape[0], batch_size):
                indices = order[start : start + batch_size]
                batch_x = features[indices]
                batch_y = labels[indices, None]

                activations = [batch_x]
                pre_activations: list[np.ndarray] = []
                for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
                    values = activations[-1] @ weight + bias
                    pre_activations.append(values)
                    next_activation = (
                        sigmoid(values)
                        if index == len(self.weights) - 1
                        else relu(values)
                    )
                    activations.append(next_activation)

                # Sigmoid plus binary cross-entropy gives this output gradient.
                delta = activations[-1] - batch_y
                weight_gradients: list[np.ndarray] = [np.empty(0)] * len(self.weights)
                bias_gradients: list[np.ndarray] = [np.empty(0)] * len(self.biases)

                for layer in reversed(range(len(self.weights))):
                    weight_gradients[layer] = activations[layer].T @ delta / len(indices)
                    bias_gradients[layer] = np.mean(delta, axis=0, keepdims=True)
                    if layer > 0:
                        delta = delta @ self.weights[layer].T
                        delta *= pre_activations[layer - 1] > 0.0

                step += 1
                for layer in range(len(self.weights)):
                    weight_moments[layer] = (
                        beta1 * weight_moments[layer]
                        + (1.0 - beta1) * weight_gradients[layer]
                    )
                    weight_variances[layer] = (
                        beta2 * weight_variances[layer]
                        + (1.0 - beta2) * weight_gradients[layer] ** 2
                    )
                    bias_moments[layer] = (
                        beta1 * bias_moments[layer]
                        + (1.0 - beta1) * bias_gradients[layer]
                    )
                    bias_variances[layer] = (
                        beta2 * bias_variances[layer]
                        + (1.0 - beta2) * bias_gradients[layer] ** 2
                    )

                    corrected_weight_moment = weight_moments[layer] / (1.0 - beta1**step)
                    corrected_weight_variance = weight_variances[layer] / (1.0 - beta2**step)
                    corrected_bias_moment = bias_moments[layer] / (1.0 - beta1**step)
                    corrected_bias_variance = bias_variances[layer] / (1.0 - beta2**step)
                    self.weights[layer] -= learning_rate * corrected_weight_moment / (
                        np.sqrt(corrected_weight_variance) + epsilon
                    )
                    self.biases[layer] -= learning_rate * corrected_bias_moment / (
                        np.sqrt(corrected_bias_variance) + epsilon
                    )


def run_attack(
    name: str,
    hidden_layers: tuple[int, ...],
    argv: list[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser(
        description=f"使用 {name} 对单个 APUF 进行建模攻击"
    )
    parser.add_argument("--crp-path", default="data/crps.npy", help="输入 CRP .npy 文件")
    parser.add_argument("--test-size", type=float, default=0.25, help="测试集比例")
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="每次更新使用的样本数")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="划分及训练随机种子")
    args = parser.parse_args(argv)

    try:
        crps = load_crps(args.crp_path)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not 0.0 < args.test_size < 1.0:
        parser.error("--test-size 必须大于 0 且小于 1")
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs、batch-size 和 learning-rate 必须为正数")

    challenges = crps[:, :-1].astype(np.float64)
    responses = crps[:, -1].astype(np.uint8)
    train_x, test_x, train_y, test_y = split_by_response(
        challenges,
        responses,
        args.test_size,
        args.seed,
    )

    # Center raw Challenge bits in [-1, 1]; the network learns the mapping itself.
    train_x = 2.0 * train_x - 1.0
    test_x = 2.0 * test_x - 1.0
    model = BinaryMLP(challenges.shape[1], hidden_layers, args.seed)
    model.fit(
        train_x,
        train_y,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.seed + 1,
    )

    train_predictions = (model.predict_proba(train_x) >= 0.5).astype(np.uint8)
    test_predictions = (model.predict_proba(test_x) >= 0.5).astype(np.uint8)
    train_accuracy = accuracy(train_y, train_predictions)
    test_accuracy = accuracy(test_y, test_predictions)
    baseline_accuracy = majority_baseline(test_y)

    print(f"模型：{name}，隐藏层：{hidden_layers}")
    print(f"CRP 总数：{len(crps)}；训练样本：{len(train_y)}；测试样本：{len(test_y)}")
    print(f"训练准确率：{train_accuracy:.2%}")
    print(f"测试准确率：{test_accuracy:.2%}")
    print(f"测试集多数类基线：{baseline_accuracy:.2%}")
