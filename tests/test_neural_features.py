"""Check controlled raw/APUF feature experiments for both neural entries."""

import contextlib
import io
import itertools
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from attacks.neural_net import BinaryMLP, prepare_features, run_attack
from puf import APUF


ROOT = Path(__file__).resolve().parents[1]


class NeuralFeatureTests(unittest.TestCase):
    def test_raw_preserves_the_previous_mapping_and_input_array(self):
        challenges = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
        original = challenges.copy()
        np.testing.assert_array_equal(prepare_features(challenges), [[-1, 1, -1], [1, -1, 1]])
        np.testing.assert_array_equal(challenges, original)

    def test_apuf_features_match_suffix_parity_for_all_three_bit_inputs(self):
        challenges = np.array(list(itertools.product((0, 1), repeat=3)), dtype=np.uint8)
        # A suffix has negative sign exactly when it contains an odd count of ones.
        expected = [[-1 if sum(row[i:]) % 2 else 1 for i in range(3)] for row in challenges]
        features = prepare_features(challenges, "apuf")
        np.testing.assert_array_equal(features, expected)
        self.assertEqual(features.shape, (8, 3))
        self.assertEqual(features.dtype, np.float64)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare_features(np.zeros((2, 3)), "unknown")

    def test_modes_preserve_labels_training_settings_and_initial_parameters(self):
        with tempfile.TemporaryDirectory(prefix="puf-features-") as directory:
            challenges = np.random.default_rng(7).integers(0, 2, (80, 8), dtype=np.uint8)
            labels = APUF(8, seed=42).response(challenges)
            path = Path(directory) / "crps.npy"
            np.save(path, np.column_stack((challenges, labels)))
            for layers in ((64,), (128, 64, 32)):
                captured = []

                def capture_fit(model, features, responses, epochs, batch_size, learning_rate, seed):
                    captured.append({
                        "features": features.copy(), "labels": responses.copy(),
                        "settings": (epochs, batch_size, learning_rate, seed),
                        "parameters": [value.copy() for value in model.weights + model.biases],
                    })

                for mode in ("raw", "apuf"):
                    with patch.object(BinaryMLP, "fit", autospec=True, side_effect=capture_fit):
                        with contextlib.redirect_stdout(io.StringIO()):
                            run_attack("test", layers, ["--crp-path", str(path), "--features", mode])
                raw, apuf = captured
                np.testing.assert_array_equal(raw["labels"], apuf["labels"])
                self.assertEqual(raw["settings"], apuf["settings"])
                self.assertEqual(raw["features"].shape, apuf["features"].shape)
                for first, second in zip(raw["parameters"], apuf["parameters"]):
                    np.testing.assert_array_equal(first, second)
                # The exact same training rows must be transformed, without label-driven inputs.
                original_rows = (raw["features"] + 1) / 2
                np.testing.assert_array_equal(apuf["features"], prepare_features(original_rows, "apuf"))

    def test_both_cli_entries_accept_both_modes_and_reject_invalid_choice(self):
        with tempfile.TemporaryDirectory(prefix="puf-features-cli-") as directory:
            challenges = np.random.default_rng(7).integers(0, 2, (80, 8), dtype=np.uint8)
            path = Path(directory) / "crps.npy"
            np.save(path, np.column_stack((challenges, APUF(8, seed=42).response(challenges))))
            for module in ("attacks.apuf_mlp", "attacks.apuf_dnn"):
                for mode in ("raw", "apuf"):
                    result = subprocess.run(
                        [sys.executable, "-m", module, "--crp-path", str(path), "--features", mode, "--epochs", "1"],
                        cwd=ROOT, capture_output=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(mode.encode(), result.stdout)
                    self.assertEqual(len(result.stdout.splitlines()), 4)
                invalid = subprocess.run(
                    [sys.executable, "-m", module, "--features", "unknown"],
                    cwd=ROOT, capture_output=True,
                )
                self.assertEqual(invalid.returncode, 2)


if __name__ == "__main__":
    unittest.main()
