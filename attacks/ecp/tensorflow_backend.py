"""CPU/float64 TensorFlow implementation of the ECP attack layer.

The contraction follows ECPRegressionLayer in the read-only TENAX reference.
It is fully vectorized and does not inspect tensor values in Python during a call.
"""

import numpy as np
import tensorflow as tf


PARAMETER_NAMES = ("factors", "coefficients", "bias")


class ECPModel(tf.Module):
    """Factorized scalar logits, with explicit dtype instead of Keras autocasting."""

    def __init__(self, parameters: dict[str, np.ndarray]) -> None:
        super().__init__(name="ecp")
        with tf.device("/CPU:0"):
            for name in PARAMETER_NAMES:
                setattr(
                    self,
                    name,
                    tf.Variable(
                        parameters[name], dtype=tf.float64, trainable=True, name=name
                    ),
                )

    def __call__(self, features: tf.Tensor) -> tf.Tensor:
        projections = tf.einsum("bd,rkd->brk", features, self.factors)
        rank_values = tf.reduce_prod(projections, axis=2)
        return tf.reduce_sum(rank_values * self.coefficients, axis=1) + self.bias[0]


class Backend:
    """NumPy-facing adapter with compiled, variable-batch TensorFlow functions."""

    version = tf.__version__

    def __init__(
        self, parameters: dict[str, np.ndarray], learning_rate: float = 0.001
    ) -> None:
        with tf.device("/CPU:0"):
            self.model = ECPModel(parameters)
            self._parameters = tuple(
                getattr(self.model, name) for name in PARAMETER_NAMES
            )
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
                amsgrad=False,
            )
            # Create slots before tracing; subsequent calls cannot add variables.
            self.optimizer.build(self._parameters)

        feature_signature = tf.TensorSpec(
            (None, parameters["factors"].shape[2]), dtype=tf.float64
        )
        batch_signature = [feature_signature, tf.TensorSpec((None,), tf.float64)]
        self._predict = tf.function(
            self._predict_tensors, input_signature=[feature_signature]
        )
        self._gradient = tf.function(
            self._loss_and_gradients_tensors, input_signature=batch_signature
        )
        self._train = tf.function(self._train_tensors, input_signature=batch_signature)

    @staticmethod
    def _tensor(values: np.ndarray) -> tf.Tensor:
        with tf.device("/CPU:0"):
            return tf.convert_to_tensor(values, dtype=tf.float64)

    def _predict_tensors(self, features: tf.Tensor) -> tf.Tensor:
        with tf.device("/CPU:0"):
            return self.model(features)

    def _loss_and_gradients_tensors(
        self, features: tf.Tensor, labels: tf.Tensor
    ) -> tuple[tf.Tensor, tuple[tf.Tensor, ...]]:
        with tf.device("/CPU:0"):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=labels, logits=self.model(features)
                    )
                )
            return loss, tuple(tape.gradient(loss, self._parameters))

    def _train_tensors(self, features: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        with tf.device("/CPU:0"):
            loss, gradients = self._loss_and_gradients_tensors(features, labels)
            self.optimizer.apply_gradients(zip(gradients, self._parameters))
            return loss

    def logits(self, features: np.ndarray) -> np.ndarray:
        return self._predict(self._tensor(features)).numpy().copy()

    def train_batch(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Apply one update and return the mean loss before that update."""
        return float(self._train(self._tensor(features), self._tensor(labels)).numpy())

    def export_parameters(self) -> dict[str, np.ndarray]:
        return {
            name: parameter.numpy().copy()
            for name, parameter in zip(PARAMETER_NAMES, self._parameters)
        }

    def loss_and_gradients(
        self, features: np.ndarray, labels: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Evaluate gradients without changing weights or optimizer state."""
        loss, gradients = self._gradient(self._tensor(features), self._tensor(labels))
        return float(loss.numpy()), {
            name: gradient.numpy().copy()
            for name, gradient in zip(PARAMETER_NAMES, gradients)
        }
