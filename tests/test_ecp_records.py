"""Portable experiment-record checks using a tiny deterministic fixture."""

import json
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from attacks.common import load_crps_snapshot
from attacks.ecp.common import (
    binary_metrics,
    file_sha256,
    numpy_logits,
    save_npz_with_csv,
    save_parameters,
    write_npz_csv_sidecars,
    write_json,
)
from attacks.ecp.compare import compare
from attacks.ecp.evaluate import evaluate
from puf.features import ChallengeFeatureTransformer


def create_record(root: Path, backend: str, crp_path: Path, crps: np.ndarray,
                  parameters: dict[str, np.ndarray]) -> Path:
    directory = root / backend
    directory.mkdir()
    features = ChallengeFeatureTransformer().transform(crps[:, :-1])
    labels = crps[:, -1]
    rows = np.arange(len(labels))
    train, test = rows[:12], rows[12:]
    logits = numpy_logits(features, parameters)
    metrics = {
        "train_count": len(train), "test_count": len(test),
        "ecp": {"train": binary_metrics(labels[train], logits[train]),
                "test": binary_metrics(labels[test], logits[test])},
        "lr": {"train": binary_metrics(labels[train], logits[train]),
               "test": binary_metrics(labels[test], logits[test])},
        "test_majority_baseline": 0.5,
    }
    config = {
        "record_version": 1, "model": "single_apuf_ecp", "backend": backend,
        "rank": 1, "branches": 1, "feature_count": features.shape[1],
        "epochs": 1, "batch_size": 4, "learning_rate": 0.001,
        "test_size": 1 / 3, "seeds": {"split": 42},
    }
    write_json(directory / "config.json", config)
    write_json(directory / "metrics.json", metrics)
    save_npz_with_csv(directory / "split.npz", train=train, test=test)
    save_parameters(directory / "initial.npz", parameters)
    save_parameters(directory / "model.npz", parameters)
    save_npz_with_csv(directory / "lr_model.npz", weights=np.zeros(features.shape[1]))
    with (directory / "history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epoch", "train_bce", "train_accuracy"))
        writer.writerow((1, 0.5, 1.0))
    save_npz_with_csv(directory / "predictions.npz", rows=test, labels=labels[test],
             logits=logits[test], probabilities=1.0 / (1.0 + np.exp(-logits[test])),
             predictions=(logits[test] >= 0).astype(np.uint8))
    save_npz_with_csv(directory / "initial_check.npz", rows=train[:4], logits=logits[train[:4]],
             loss=np.array(0.5), grad_factors=np.zeros_like(parameters["factors"]),
             grad_coefficients=np.zeros_like(parameters["coefficients"]),
             grad_bias=np.zeros_like(parameters["bias"]))
    artifacts = {path.name: file_sha256(path) for path in directory.iterdir()}
    write_json(directory / "manifest.json", {
        "record_version": 1, "status": "completed",
        "data": {"path": str(crp_path), "sha256": file_sha256(crp_path),
                 "shape": list(crps.shape), "dtype": str(crps.dtype)},
        "artifact_sha256": artifacts,
    })
    return directory


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(5)
        challenges = rng.integers(0, 2, size=(30, 2), dtype=np.uint8)
        features = ChallengeFeatureTransformer().transform(challenges)
        responses = (features[:, 0] + 0.4 * features[:, 1] >= 0).astype(np.uint8)
        self.crps = np.column_stack((challenges, responses))
        self.crp_path = self.root / "crps.npy"
        np.save(self.crp_path, self.crps)
        self.parameters = {
            "factors": np.ones((1, 1, 3), dtype=np.float64),
            "coefficients": np.ones(1, dtype=np.float64),
            "bias": np.zeros(1, dtype=np.float64),
        }
        self.torch = create_record(self.root, "torch", self.crp_path, self.crps, self.parameters)
        self.tf = create_record(self.root, "tensorflow", self.crp_path, self.crps, self.parameters)

    def tearDown(self):
        self.temp.cleanup()

    def test_crp_hash_and_parsed_data_share_one_snapshot(self):
        loaded, digest = load_crps_snapshot(self.crp_path)
        self.assertEqual(digest, file_sha256(self.crp_path))
        np.testing.assert_array_equal(loaded, self.crps)

    def test_numpy_model_reload_reproduces_recorded_metrics(self):
        result = evaluate(self.torch)
        self.assertTrue(result["matches_saved_metrics"])
        self.assertEqual(result["backend"], "numpy")

    def test_cross_framework_record_protocol_and_initial_probe_match(self):
        result = compare(self.torch, self.tf)
        self.assertTrue(result["same_data_protocol_and_initialization"])

    def test_changed_crp_is_rejected(self):
        changed = self.root / "changed.npy"
        corrupted = self.crps.copy()
        corrupted[0, 0] ^= 1
        np.save(changed, corrupted)
        with self.assertRaisesRegex(ValueError, "CRP 文件摘要"):
            evaluate(self.torch, changed)

    def test_tampered_artifact_is_rejected(self):
        with (self.torch / "model.npz").open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "产物已改变或损坏"):
            evaluate(self.torch)

    def test_overlapping_split_is_rejected(self):
        with np.load(self.torch / "split.npz") as saved:
            train, test = saved["train"], saved["test"]
        train[0] = test[0]
        np.savez(self.torch / "split.npz", train=train, test=test)
        manifest_path = self.torch / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifact_sha256"]["split.npz"] = file_sha256(self.torch / "split.npz")
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "索引必须互斥"):
            evaluate(self.torch)

    def test_compare_rejects_different_initialization(self):
        changed = dict(self.parameters)
        changed["bias"] = np.array([0.25])
        save_parameters(self.tf / "initial.npz", changed)
        write_npz_csv_sidecars(self.tf / "initial.npz")
        manifest_path = self.tf / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for path in (self.tf / "initial.npz", *self.tf.glob("initial*.csv")):
            manifest["artifact_sha256"][path.name] = file_sha256(path)
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "初始参数不同"):
            compare(self.torch, self.tf)


if __name__ == "__main__":
    unittest.main()
