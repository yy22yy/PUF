"""Small deterministic checks; no CRP experiment or training run is performed.

Run with ``python -m unittest discover -s tests -v``. Optional framework checks
are skipped when that framework is not installed.
"""

import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from attacks.common import split_by_response
from attacks.ecp.common import (
    binary_metrics,
    file_sha256,
    initialize_parameters,
    load_parameters,
    numpy_logits,
    save_parameters,
    save_npz_with_csv,
    split_indices,
    validate_parameters,
)


FEATURES = np.array(
    [[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, -1.0, 1.0]],
    dtype=np.float64,
)
LABELS = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)


class NPZCSVTests(unittest.TestCase):
    def test_npz_writer_creates_archive_index_and_array_csvs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arrays.npz"
            save_npz_with_csv(path, matrix=np.array([[1, 2], [3, 4]]), vector=np.array([5, 6]))
            self.assertTrue((Path(directory) / "arrays.csv").is_file())
            self.assertTrue((Path(directory) / "arrays__matrix.csv").is_file())
            self.assertTrue((Path(directory) / "arrays__vector.csv").is_file())
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["matrix"], [[1, 2], [3, 4]])


def example_parameters(rank: int = 3) -> dict[str, np.ndarray]:
    return {
        "factors": np.array(
            [[[0.2, -0.3, 0.1]], [[-0.4, 0.1, 0.5]], [[0.3, 0.6, -0.2]]],
            dtype=np.float64,
        )[:rank].copy(),
        "coefficients": np.array([0.7, -0.5, 0.9], dtype=np.float64)[:rank].copy(),
        "bias": np.array([0.13], dtype=np.float64),
    }


def linear_reference(features: np.ndarray, parameters: dict) -> np.ndarray:
    """Independent single-APUF reduction: collapse CP rank into one weight."""
    weight = np.zeros(features.shape[1], dtype=np.float64)
    for coefficient, factor in zip(parameters["coefficients"], parameters["factors"]):
        weight += coefficient * factor[0]
    return features @ weight + parameters["bias"][0]


def loss_reference(features: np.ndarray, labels: np.ndarray, parameters: dict) -> float:
    logits = linear_reference(features, parameters)
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


class CommonTests(unittest.TestCase):
    def test_rank_one_and_multiple_ranks_reduce_to_linear_model(self):
        for rank in (1, 3):
            with self.subTest(rank=rank):
                parameters = example_parameters(rank)
                actual = numpy_logits(FEATURES, parameters)
                self.assertEqual(actual.shape, (len(FEATURES),))
                np.testing.assert_allclose(
                    actual, linear_reference(FEATURES, parameters), rtol=1e-13, atol=1e-13
                )

    def test_initialization_is_reproducible_and_uses_float64(self):
        first = initialize_parameters(feature_count=3, rank=4, seed=42)
        second = initialize_parameters(feature_count=3, rank=4, seed=42)
        self.assertEqual(first["factors"].shape, (4, 1, 3))
        self.assertEqual(first["coefficients"].shape, (4,))
        self.assertEqual(first["bias"].shape, (1,))
        for key in first:
            self.assertEqual(first[key].dtype, np.dtype(np.float64))
            np.testing.assert_array_equal(first[key], second[key])

    def test_parameter_round_trip_is_lossless(self):
        parameters = example_parameters()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_parameters(path, parameters)
            restored = load_parameters(path)
        self.assertEqual(set(restored), set(parameters))
        for key in parameters:
            np.testing.assert_array_equal(restored[key], parameters[key])
            self.assertEqual(restored[key].dtype, parameters[key].dtype)

    def test_parameter_validation_rejects_invalid_shapes_and_values(self):
        invalid = {
            "missing_branch_axis": ("factors", np.ones((3, 3))),
            "multiple_branches": ("factors", np.ones((3, 2, 3))),
            "inconsistent_rank": ("coefficients", np.ones(2)),
            "non_vector_coefficients": ("coefficients", np.ones((3, 1))),
            "non_vector_bias": ("bias", np.array(0.0)),
            "nonfinite": ("bias", np.array([np.nan])),
        }
        for name, (key, value) in invalid.items():
            with self.subTest(case=name):
                parameters = example_parameters()
                parameters[key] = value
                with self.assertRaises(ValueError):
                    validate_parameters(parameters)
        with self.assertRaises(ValueError):
            validate_parameters(example_parameters(), feature_count=4)

    def test_loader_rejects_malformed_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.npz"
            np.savez(path, factors=example_parameters()["factors"])
            with self.assertRaises(ValueError):
                load_parameters(path)

    def test_split_matches_existing_baselines_exactly(self):
        responses = np.array([0] * 19 + [1] * 10)
        row_ids = np.arange(len(responses))[:, None]
        for test_size in (0.25, 0.4):
            with self.subTest(test_size=test_size):
                train, test = split_indices(responses, test_size=test_size, seed=42)
                old_train, old_test, old_train_y, old_test_y = split_by_response(
                    row_ids, responses, test_size=test_size, seed=42
                )
                np.testing.assert_array_equal(train, old_train[:, 0])
                np.testing.assert_array_equal(test, old_test[:, 0])
                np.testing.assert_array_equal(responses[train], old_train_y)
                np.testing.assert_array_equal(responses[test], old_test_y)
                self.assertEqual(np.intersect1d(train, test).size, 0)
                np.testing.assert_array_equal(np.sort(np.concatenate([train, test])), row_ids[:, 0])

    def test_split_rejects_unstratifiable_data(self):
        for responses in (np.zeros(4), np.array([0, 0, 0, 1])):
            with self.subTest(responses=responses.tolist()):
                with self.assertRaises(ValueError):
                    split_indices(responses, test_size=0.25, seed=42)

    def test_metrics_are_stable_for_extreme_logits(self):
        labels = np.array([1.0, 0.0, 1.0, 0.0])
        logits = np.array([1000.0, -1000.0, -1000.0, 1000.0])
        with np.errstate(over="raise", invalid="raise"):
            metrics = binary_metrics(labels, logits)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["bce"], 500.0)
        self.assertTrue(np.isfinite(metrics["bce"]))

    def test_zero_logit_predicts_one(self):
        metrics = binary_metrics(np.ones(2), np.zeros(2))
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["bce"], np.log(2.0))

    def test_file_hash_uses_file_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "content"
            path.write_bytes(b"abc")
            self.assertEqual(
                file_sha256(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )


class BackendChecks:
    """Mixin uses forward/autodiff operations only, with no optimizer steps."""

    backend_module: str

    def make_backend(self, parameters):
        module = importlib.import_module(self.backend_module)
        return module.Backend(parameters, learning_rate=0.001)

    def test_initial_outputs_and_export_match_reference(self):
        for rank in (1, 3):
            with self.subTest(rank=rank):
                parameters = example_parameters(rank)
                backend = self.make_backend(parameters)
                np.testing.assert_allclose(
                    backend.logits(FEATURES), linear_reference(FEATURES, parameters),
                    rtol=1e-12, atol=1e-12,
                )
                exported = backend.export_parameters()
                for key in parameters:
                    np.testing.assert_array_equal(exported[key], parameters[key])
                restored = self.make_backend(exported)
                np.testing.assert_array_equal(restored.logits(FEATURES), backend.logits(FEATURES))

    def test_loss_and_gradients_match_finite_differences(self):
        parameters = example_parameters()
        backend = self.make_backend(parameters)
        loss, gradients = backend.loss_and_gradients(FEATURES, LABELS)
        self.assertAlmostEqual(float(loss), loss_reference(FEATURES, LABELS, parameters), places=12)
        self.assertEqual(set(gradients), set(parameters))
        epsilon = 1e-6
        for key, value in parameters.items():
            with self.subTest(parameter=key):
                expected = np.zeros_like(value)
                for index in np.ndindex(value.shape):
                    plus = {name: array.copy() for name, array in parameters.items()}
                    minus = {name: array.copy() for name, array in parameters.items()}
                    plus[key][index] += epsilon
                    minus[key][index] -= epsilon
                    expected[index] = (
                        loss_reference(FEATURES, LABELS, plus)
                        - loss_reference(FEATURES, LABELS, minus)
                    ) / (2.0 * epsilon)
                np.testing.assert_allclose(gradients[key], expected, rtol=1e-5, atol=2e-7)
        exported = backend.export_parameters()
        for key in parameters:
            np.testing.assert_array_equal(exported[key], parameters[key])


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
class TorchBackendTests(BackendChecks, unittest.TestCase):
    backend_module = "attacks.ecp.torch_backend"


@unittest.skipUnless(importlib.util.find_spec("tensorflow") is not None, "TensorFlow is not installed")
class TensorFlowBackendTests(BackendChecks, unittest.TestCase):
    backend_module = "attacks.ecp.tensorflow_backend"


if __name__ == "__main__":
    unittest.main()
