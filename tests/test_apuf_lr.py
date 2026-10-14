"""Check that LR accepts equivalent paired and merged input data."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from attacks.common import load_challenge_responses
from puf import APUF


ROOT = Path(__file__).resolve().parents[1]


class LRInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="puf-lr-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.challenges = np.random.default_rng(7).integers(0, 2, (80, 8), dtype=np.uint8)
        self.responses = APUF(n=8, seed=42).response(self.challenges)
        self.challenge_path = self.directory / "challenges.npy"
        self.response_path = self.directory / "responses.npy"
        self.crp_path = self.directory / "crps.npy"
        np.save(self.challenge_path, self.challenges)
        np.save(self.response_path, self.responses)
        np.save(self.crp_path, np.column_stack((self.challenges, self.responses)))

    def run_lr(self, *options):
        return subprocess.run(
            [sys.executable, "-m", "attacks.apuf_lr", *options],
            cwd=ROOT, capture_output=True,
        )

    def test_separate_and_merged_inputs_produce_identical_metrics(self):
        merged = self.run_lr("--crp-path", str(self.crp_path))
        separate = self.run_lr(
            "--challenge-path", str(self.challenge_path),
            "--response-path", str(self.response_path),
        )
        self.assertEqual(merged.returncode, 0, merged.stderr)
        self.assertEqual(separate.returncode, 0, separate.stderr)
        self.assertEqual(merged.stdout, separate.stdout)
        self.assertEqual(len(merged.stdout.splitlines()), 4)

    def test_mlp_and_dnn_report_only_training_and_test_metrics(self):
        for module in ("attacks.apuf_mlp", "attacks.apuf_dnn"):
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", module, "--crp-path", str(self.crp_path), "--epochs", "1"],
                    cwd=ROOT, capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(result.stdout.splitlines()), 4)

    def test_column_responses_are_accepted_without_changing_metrics(self):
        merged = self.run_lr("--crp-path", str(self.crp_path))
        np.save(self.response_path, self.responses[:, None])
        separate = self.run_lr(
            "--challenge-path", str(self.challenge_path),
            "--response-path", str(self.response_path),
        )
        self.assertEqual(merged.returncode, 0, merged.stderr)
        self.assertEqual(separate.returncode, 0, separate.stderr)
        self.assertEqual(merged.stdout, separate.stdout)

    def test_incomplete_or_conflicting_input_arguments_are_rejected(self):
        for options in (
            ("--challenge-path", str(self.challenge_path)),
            ("--response-path", str(self.response_path)),
            ("--crp-path", str(self.crp_path), "--challenge-path", str(self.challenge_path), "--response-path", str(self.response_path)),
        ):
            with self.subTest(options=options):
                self.assertEqual(self.run_lr(*options).returncode, 2)

    def test_invalid_response_values_are_rejected_before_casting(self):
        invalid = self.responses.astype(float)
        invalid[0] = 0.5
        np.save(self.response_path, invalid)
        with self.assertRaises(ValueError):
            load_challenge_responses(self.challenge_path, self.response_path)

    def test_sample_count_mismatch_is_rejected(self):
        np.save(self.response_path, self.responses[:-1])
        with self.assertRaises(ValueError):
            load_challenge_responses(self.challenge_path, self.response_path)

    def test_invalid_challenge_shape_and_values_are_rejected(self):
        for invalid in (self.challenges[0], np.empty((80, 0)), self.challenges.astype(float) + 0.5):
            with self.subTest(shape=invalid.shape):
                np.save(self.challenge_path, invalid)
                with self.assertRaises(ValueError):
                    load_challenge_responses(self.challenge_path, self.response_path)


if __name__ == "__main__":
    unittest.main()
