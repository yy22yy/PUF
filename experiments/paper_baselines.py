"""两个可运行的小规模实验；复用本项目PUF，结构参考旧puf_modeling工程。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from puf import XORAPUF
from puf.features import ChallengeFeatureTransformer


class Attack(nn.Module):
    """LR: 每链tanh→乘积→tanh；ECP: k链乘积→R项加权和。"""
    def __init__(self, dim, k, method, rank=1):
        super().__init__()
        self.method, self.k = method, k
        self.factors = nn.Parameter(torch.randn(rank, k, dim) / np.sqrt(dim))
        if method == 'ecp':
            self.alpha = nn.Parameter(torch.randn(rank) / np.sqrt(rank))

    def parts(self, x):
        delay = torch.einsum('bd,rkd->brk', x, self.factors)
        if self.method == 'lr':
            delay = delay.tanh()
        return delay.prod(dim=2)

    def forward(self, x):
        terms = self.parts(x)
        if self.method == 'lr':
            # 核心PUF用delay>=0输出1；乘积符号须按k的奇偶映射。
            return ((-1) ** (self.k + 1) * terms[:, 0]).tanh()
        return terms @ self.alpha

    def predict(self, x):
        return self(x) >= 0


def dataset(bits, k, n_train, n_test, seed):
    """同一设备生成无重复且互斥的训练/测试CRP；真值仅用于事后核对。"""
    count = n_train + n_test
    if count > 2 ** bits:
        raise ValueError('训练+测试数量不能超过2**bits（需要无重复Challenge）')
    rng = np.random.default_rng(seed)
    if bits <= 20:
        numbers = rng.choice(2 ** bits, count, replace=False)
        c = ((numbers[:, None] >> np.arange(bits)) & 1).astype(np.uint8)
    else:
        c = np.empty((0, bits), dtype=np.uint8)
        while len(c) < count:
            fresh = rng.integers(0, 2, (count - len(c) + 64, bits), dtype=np.uint8)
            c = np.unique(np.concatenate((c, fresh)), axis=0)
        c = c[rng.permutation(len(c))[:count]]
    puf = XORAPUF(bits, xor_count=k, seed=seed)
    x = torch.tensor(ChallengeFeatureTransformer().transform(c), dtype=torch.float32)
    y = torch.tensor(puf.response(c), dtype=torch.float32)
    weights = np.stack([p.weights for p in puf.components])
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:], weights


def train(model, x, y, epochs, batch_size, lr, seed):
    """只用训练数据更新；测试集不用于选择模型或停止训练。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    started = time.perf_counter()
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        for index in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            output = model(x[index])
            loss = (F.binary_cross_entropy((output + 1) / 2, y[index])
                    if model.method == 'lr' else F.binary_cross_entropy_with_logits(output, y[index]))
            if not torch.isfinite(loss):
                raise FloatingPointError('非有限损失；请减小学习率')
            loss.backward()
            optimizer.step()
    return time.perf_counter() - started


def summarize(rows):
    """Run至少一个attempt准确率>90%即成功；准确率均值只包含成功attempt。"""
    result = []
    keys = ('experiment', 'k', 'n_train', 'rank')
    groups = dict.fromkeys(tuple(row.get(key, '') for key in keys) for row in rows)
    for group in groups:
        selected = [r for r in rows if tuple(r.get(key, '') for key in keys) == group]
        runs = {r['run'] for r in selected}
        success = [r for r in selected if r['accuracy'] > .9]
        result.append(dict(zip(keys, group), total_runs=len(runs),
                           successful_runs=len({r['run'] for r in success}),
                           success_rate=len({r['run'] for r in success}) / len(runs),
                           attempts=len(selected), mean_accuracy=float(np.mean([r['accuracy'] for r in selected])),
                           mean_success_accuracy=float(np.mean([r['accuracy'] for r in success])) if success else None,
                           mean_attempt_seconds=float(np.mean([r['seconds'] for r in selected]))))
    return result


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['lr', 'ecp', 'both'], default='both')
    parser.add_argument('--bits', type=int, default=16)
    parser.add_argument('--k', type=int, nargs='+', default=[2, 3])
    parser.add_argument('--train-sizes', type=int, nargs='+', default=[2000, 6000])
    parser.add_argument('--test-size', type=int, default=2000, help='独立测试CRP数量，不是比例')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--rank', type=int, default=5, help='高秩R；与每Run中R次Rank1独立训练比较')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True, help='必须是尚不存在的新目录')
    args = parser.parse_args()
    if min(args.bits, *args.k, *args.train_sizes, args.test_size, args.runs,
           args.epochs, args.batch_size) < 1 or args.rank < 2 or not 0 < args.lr < float('inf') or args.seed < 0:
        parser.error('数量必须为正数，rank>=2，学习率必须为有限正数，seed>=0')
    if max(args.train_sizes) + args.test_size > 2 ** args.bits:
        parser.error('训练+测试数量超过Challenge空间；增大bits或减少样本')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), 'output': str(args.output), 'torch': torch.__version__,
              'numpy': np.__version__, 'status': 'running', 'success_threshold': '>0.90',
              'note': 'PyTorch小规模结构实验，不是论文数值复现；每Run独立PUF，Run内固定数据',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    config_path = args.output / 'config.json'
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    rows, parts = [], []
    methods = ['lr', 'ecp'] if args.experiment == 'both' else [args.experiment]
    try:
        for k in args.k:
            for n in args.train_sizes:
                for run in range(args.runs):
                    data_seed = args.seed + k * 1000 + run
                    x, y, test_x, test_y, weights = dataset(args.bits, k, n, args.test_size, data_seed)
                    digest = hashlib.sha256(b''.join(v.numpy().tobytes() for v in (x, y, test_x, test_y))).hexdigest()
                    prefix = f'k{k}_n{n}_run{run}'
                    np.savez_compressed(args.output / f'{prefix}_data.npz', train_x=x.numpy(), train_y=y.numpy(),
                                        test_x=test_x.numpy(), test_y=test_y.numpy(), true_weights=weights)
                    for method in methods:
                        # 同一份数据上比较：R次独立Rank1 vs 1次联合RankR。
                        schedules = [(1, 1)] if method == 'lr' else [(1, args.rank), (args.rank, 1)]
                        for rank, attempts in schedules:
                            for attempt in range(attempts):
                                model_seed = args.seed + 100000 + run * 10000 + rank * 100 + attempt
                                torch.manual_seed(model_seed)
                                model = Attack(args.bits + 1, k, method, rank)
                                seconds = train(model, x, y, args.epochs, args.batch_size, args.lr, data_seed + 500000)
                                name = f'{prefix}_{method}_r{rank}_a{attempt}'
                                with torch.no_grad():
                                    predictions = model.predict(test_x)
                                    accuracy = float((predictions == test_y).double().mean())
                                    if method == 'ecp':
                                        terms = model.parts(test_x)
                                        for r in range(rank):
                                            # 含alpha的单项预测，允许符号翻转的值仅作事后诊断。
                                            acc = float(((terms[:, r] * model.alpha[r] >= 0) == test_y).double().mean())
                                            parts.append(dict(k=k, n_train=n, run=run, rank=rank, attempt=attempt,
                                                              component=r, accuracy=acc, sign_invariant_accuracy=max(acc, 1-acc)))
                                torch.save(model.state_dict(), args.output / f'{name}.pt')
                                np.save(args.output / f'{name}_predictions.npy', predictions.numpy())
                                rows.append(dict(experiment=method, k=k, n_train=n, rank=rank, run=run, attempt=attempt,
                                                 data_seed=data_seed, model_seed=model_seed, shuffle_seed=data_seed + 500000, data_sha256=digest,
                                                 accuracy=accuracy, success=accuracy > .9, seconds=seconds))
                                save_csv(args.output / 'attempts.csv', rows)
                                print(f'{name}: accuracy={accuracy:.2%}, time={seconds:.2f}s', flush=True)
        summary = summarize(rows)
        save_csv(args.output / 'summary.csv', summary)
        if parts:
            save_csv(args.output / 'submodels.csv', parts)
        config['status'] = 'completed'
        print('\n汇总（成功率 / 成功attempt平均准确率）：', flush=True)
        for row in summary:
            mean = row['mean_success_accuracy']
            print(f"{row['experiment']} k={row['k']} N={row['n_train']} rank={row['rank']}: "
                  f"{row['successful_runs']}/{row['total_runs']}, " + (f'{mean:.2%}' if mean is not None else '无成功attempt'))
        print(f'结果目录：{args.output.resolve()}')
    except BaseException as error:
        config.update(status='failed', error=str(error))
        raise
    finally:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
