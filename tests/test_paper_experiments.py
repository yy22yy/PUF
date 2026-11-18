"""实验协议回归测试：符号、k链结构、Run分母与实际CLI产物。"""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH, 'requires torch')
class PaperExperimentTests(unittest.TestCase):
    def test_models_represent_core_xor_for_even_and_odd_k(self):
        # 混淆k/rank或反转标签会破坏此测试。
        import torch
        from experiments.paper_baselines import Attack, dataset
        for k in (2, 3):
            x, y, _, _, weights = dataset(8, k, 100, 20, 11)
            lr = Attack(9, k, 'lr', 1)
            ecp = Attack(9, k, 'ecp', 1)
            with torch.no_grad():
                lr.factors.copy_(torch.tensor(weights[None], dtype=torch.float32))
                ecp.factors.copy_(torch.tensor(weights[None], dtype=torch.float32))
                ecp.alpha.fill_((-1) ** (k + 1))
            np.testing.assert_array_equal(lr.predict(x).numpy(), y.numpy())
            np.testing.assert_array_equal(ecp.predict(x).numpy(), y.numpy())
            self.assertEqual(ecp.parts(x).shape, (100, 1))

    def test_lr_has_both_tanh_layers(self):
        import torch
        from experiments.paper_baselines import Attack
        model = Attack(2, 2, 'lr')
        with torch.no_grad():
            model.factors.copy_(torch.tensor([[[1., 0.], [0., 2.]]]))
        # 手算 -tanh(tanh(1)*tanh(2))，不仅检查符号。
        self.assertAlmostEqual(float(model(torch.tensor([[1., 1.]])).detach()), -.6256268, places=6)

    def test_success_uses_run_denominator_and_successful_attempts_only(self):
        from experiments.paper_baselines import summarize
        rows = [dict(k=3, n_train=100, rank=1, run=r, accuracy=a, seconds=1.)
                for r, a in [(0, .95), (0, .5), (1, .9), (1, .6)]]
        result = summarize(rows)[0]
        self.assertEqual(result['successful_runs'], 1)
        self.assertEqual(result['total_runs'], 2)
        self.assertEqual(result['success_rate'], .5)
        self.assertEqual(result['mean_success_accuracy'], .95)
        failed = summarize([dict(rows[1], run=0)])[0]
        self.assertIsNone(failed['mean_success_accuracy'])

    def test_cli_outputs_fixed_data_attempts_and_submodels(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'run'
            command = [sys.executable, '-m', 'experiments.paper_baselines', '--experiment', 'both',
                       '--bits', '8', '--k', '2', '--train-sizes', '100', '--test-size', '40',
                       '--runs', '2', '--epochs', '2', '--rank', '2', '--output', str(out)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = list(csv.DictReader((out / 'attempts.csv').read_text().splitlines()))
            self.assertEqual(len(rows), 8)  # 2 LR + 2*(2 rank1 + 1 rank2)
            ecp = [r for r in rows if r['experiment'] == 'ecp']
            for run in ('0', '1'):
                self.assertEqual(len({r['data_sha256'] for r in ecp if r['run'] == run}), 1)
                self.assertEqual(len({r['model_seed'] for r in ecp if r['run'] == run}), 3)
                self.assertEqual(len({r['shuffle_seed'] for r in ecp if r['run'] == run}), 1)
            self.assertEqual(len(list(csv.DictReader((out / 'submodels.csv').read_text().splitlines()))), 8)
            self.assertEqual(json.loads((out / 'config.json').read_text())['status'], 'completed')
            repeated = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)


if __name__ == '__main__':
    unittest.main()
