"""Exercise model selection, saved CRP identity, and input pairing checks."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from merge_crp import merge_crp


ROOT = Path(__file__).resolve().parents[1]


class CRPGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="puf-crp-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.challenges = np.random.default_rng(42).integers(0, 2, (100, 64), dtype=np.uint8)
        self.challenge_path = self.directory / "challenges.npy"
        np.save(self.challenge_path, self.challenges)

    def generate(self, puf_type="apuf", xor_count=None, response_name="responses.npy"):
        output = self.directory / puf_type / response_name
        command = [
            sys.executable, str(ROOT / "generate_crp.py"),
            "--challenge-path", str(self.challenge_path),
            "--response-path", str(output),
            "--puf-type", puf_type, "--seed", "42", "--challenge-seed", "42",
        ]
        if xor_count is not None:
            command.extend(["--xor-count", str(xor_count)])
        completed = subprocess.run(command, cwd=ROOT, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr.decode(errors="replace"))
        return output

    def test_apuf_and_xor_saved_outputs_have_correct_identity_and_response(self):
        features = np.array([
            [np.prod(1 - 2 * row[i:].astype(np.int64)) for i in range(64)] + [1]
            for row in self.challenges
        ])
        for model_type, count in (("apuf", None), ("xor-apuf", 3)):
            with self.subTest(model_type=model_type):
                response_path = self.generate(model_type, count)
                responses = np.load(response_path)
                weights = np.load(response_path.with_name("puf_weights.npy"))
                metadata = json.loads(response_path.with_name("responses_metadata.json").read_text(encoding="utf-8"))
                expected_type = "APUF" if count is None else "XORAPUF"
                self.assertEqual(metadata["puf"]["type"], expected_type)
                self.assertEqual(metadata["puf"]["xor_count"], count or 1)
                self.assertEqual(metadata["puf"]["weight_shape"], list(weights.shape))
                self.assertEqual(responses.shape, (100,))
                if count is None:
                    self.assertEqual(weights.shape, (65,))
                    expected_response = (features @ weights >= 0).astype(np.uint8)
                else:
                    self.assertEqual(weights.shape, (3, 65))
                    self.assertEqual(np.unique(weights, axis=0).shape[0], 3)
                    # XOR is the parity of the component response bits.
                    expected_response = ((features @ weights.T >= 0).sum(axis=1) % 2).astype(np.uint8)
                    for seed, row in zip(metadata["puf"]["component_seeds"], weights):
                        np.testing.assert_array_equal(row, np.random.default_rng(seed).normal(size=65))
                np.testing.assert_array_equal(responses, expected_response)
                self.assertEqual(metadata["puf"]["weights_file"], "puf_weights.npy")
                for name, array in (("responses", responses), ("puf_weights", weights)):
                    csv = np.loadtxt(response_path.with_name(name + ".csv"), delimiter=",")
                    np.testing.assert_array_equal(csv, array)

                crp_path = response_path.with_name("crps.npy")
                completed = subprocess.run([
                    sys.executable, str(ROOT / "merge_crp.py"),
                    "--challenge-path", str(self.challenge_path),
                    "--response-path", str(response_path), "--output-path", str(crp_path),
                ], cwd=ROOT, capture_output=True)
                self.assertEqual(completed.returncode, 0, completed.stderr.decode(errors="replace"))
                crps = np.load(crp_path)
                self.assertEqual(crps.shape, (100, 65))
                np.testing.assert_array_equal(crps[:, :-1], self.challenges)
                np.testing.assert_array_equal(crps[:, -1], responses)
                np.testing.assert_array_equal(np.loadtxt(crp_path.with_suffix(".csv"), delimiter=","), crps)
                record = json.loads(crp_path.with_name("crps_metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(record["puf"]["type"], expected_type)
                self.assertEqual(record["sample_count"], 100)
                self.assertEqual(record["challenge"]["generation_seed"], 42)
                self.assertEqual(record["crp"]["values_sha256"], hashlib.sha256(crps.tobytes()).hexdigest())

    def test_weight_filename_is_independent_of_response_filename(self):
        response_path = self.generate(response_name="prediction.npy")
        self.assertTrue(response_path.with_name("puf_weights.npy").is_file())
        self.assertTrue(response_path.with_name("puf_weights.csv").is_file())
        record = json.loads(response_path.with_name("prediction_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(record["puf"]["weights_file"], "puf_weights.npy")

    def test_mismatched_challenges_are_rejected_before_writing(self):
        response_path = self.generate("xor-apuf", 3)
        changed = self.challenges.copy()
        changed[0, 0] ^= 1
        np.save(self.challenge_path, changed)
        output = self.directory / "rejected.npy"
        with self.assertRaisesRegex(ValueError, "Challenge"):
            merge_crp(self.challenge_path, response_path, output)
        self.assertFalse(output.exists())

    def test_modified_responses_are_rejected_before_writing(self):
        response_path = self.generate()
        responses = np.load(response_path)
        responses[0] ^= 1
        np.save(response_path, responses)
        output = self.directory / "rejected.npy"
        with self.assertRaisesRegex(ValueError, "Response"):
            merge_crp(self.challenge_path, response_path, output)
        self.assertFalse(output.exists())

    def test_legacy_response_metadata_is_accepted(self):
        response_path = self.generate()
        record_path = response_path.with_name("responses_metadata.json")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["record_version"] = 1
        for field in ("xor_count", "weight_shape", "component_weight_count"):
            record["puf"].pop(field)
        record_path.write_text(json.dumps(record), encoding="utf-8")
        output = self.directory / "legacy.npy"
        merge_crp(self.challenge_path, response_path, output)
        merged = json.loads(output.with_name("legacy_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(merged["puf"]["type"], "APUF")

    def test_external_data_without_metadata_has_unknown_type(self):
        response_path = self.directory / "external.npy"
        np.save(response_path, np.zeros((100, 1), dtype=np.uint8))
        output = self.directory / "external_crps.npy"
        merge_crp(self.challenge_path, response_path, output)
        record = json.loads(output.with_name("external_crps_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(record["puf"]["type"], "UNKNOWN")
        self.assertIsNone(record["response_metadata"])

    def test_invalid_model_options_are_rejected(self):
        for options in (("--puf-type", "xor-apuf"), ("--puf-type", "xor-apuf", "--xor-count", "0"), ("--xor-count", "3")):
            with self.subTest(options=options):
                result = subprocess.run([sys.executable, str(ROOT / "generate_crp.py"), *options], cwd=ROOT, capture_output=True)
                self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
