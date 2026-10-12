"""CPU/float64 PyTorch implementation of the ECP attack layer.

The contraction follows ECPRegressionLayer in the read-only TENAX reference.
All rank components are summed at once, avoiding the reference's chunk overwrite.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


PARAMETER_NAMES = ("factors", "coefficients", "bias")


class ECPModel(nn.Module):
    """Factorized scalar logits; with one branch this is linear in features."""

    def __init__(self, parameters: dict[str, np.ndarray]) -> None:
        super().__init__()
        for name in PARAMETER_NAMES:
            self.register_parameter(
                name,
                nn.Parameter(
                    torch.tensor(parameters[name], dtype=torch.float64, device="cpu")
                ),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projections = torch.einsum("bd,rkd->brk", features, self.factors)
        rank_values = torch.prod(projections, dim=2)
        return torch.sum(rank_values * self.coefficients, dim=1) + self.bias[0]


class Backend:
    """Small NumPy-facing adapter around PyTorch autograd and Adam."""

    version = torch.__version__

    def __init__(
        self, parameters: dict[str, np.ndarray], learning_rate: float = 0.001
    ) -> None:
        self.model = ECPModel(parameters)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )

    @staticmethod
    def _tensor(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float64, device="cpu")

    def logits(self, features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.model(self._tensor(features)).numpy().copy()

    def train_batch(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Apply one update and return the mean loss before that update."""
        self.optimizer.zero_grad(set_to_none=True)
        loss = functional.binary_cross_entropy_with_logits(
            self.model(self._tensor(features)), self._tensor(labels), reduction="mean"
        )
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().item())

    def export_parameters(self) -> dict[str, np.ndarray]:
        return {
            name: getattr(self.model, name).detach().numpy().copy()
            for name in PARAMETER_NAMES
        }

    def loss_and_gradients(
        self, features: np.ndarray, labels: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Evaluate gradients without changing weights or optimizer state."""
        loss = functional.binary_cross_entropy_with_logits(
            self.model(self._tensor(features)), self._tensor(labels), reduction="mean"
        )
        parameters = tuple(getattr(self.model, name) for name in PARAMETER_NAMES)
        gradients = torch.autograd.grad(loss, parameters)
        return float(loss.detach().item()), {
            name: gradient.detach().numpy().copy()
            for name, gradient in zip(PARAMETER_NAMES, gradients)
        }
