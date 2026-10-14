"""ECP-TRN 教学示例；外部 CRP 模式还需要 NumPy 和 attacks/common.py。

教学仿真：python ecp_trn_minimal.py
现有数据：python ecp_trn_minimal.py --crp-path data/crps.npy --order 1
核心：z(phi) = bias + sum_r weight[r] * product_j dot(phi, factor[j, r])。
默认生成无噪声 16-bit 3-XOR 数据；外部模式直接使用文件中的响应。
不代表论文规模复现，攻击网络不读取 PUF 真值权重。
"""

import argparse
import json
from pathlib import Path
import time

import torch
from torch import nn


class ECPTRN(nn.Module):
    def __init__(self, input_dim, order=3, rank=4):
        super().__init__()
        if min(input_dim, order, rank) < 1:
            raise ValueError("input_dim、order、rank 必须为正整数")
        # K 个阶、R 个分量，每个因子都是 D 维可训练向量。
        self.factors = nn.Parameter(torch.randn(order, rank, input_dim) / input_dim**0.5)
        self.output = nn.Linear(rank, 1)  # 分量权重 lambda 和最终 bias。

    def forward(self, phi):
        projections = torch.einsum("bd,krd->bkr", phi, self.factors)
        products = projections.prod(dim=1)  # (B,K,R) -> (B,R)。
        return self.output(products).squeeze(-1)  # (B,)，单条样本也保留批维。


def load_external_crps(path):
    """加载合并 CRP，先去重，避免重复挑战进入训练和测试两侧。"""
    import numpy as np

    if path.suffix.lower() == ".npy":
        crps = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".csv":
        crps = np.loadtxt(path, delimiter=",", ndmin=2)
    else:
        raise ValueError("CRP 文件必须是 .npy 或 .csv")
    if crps.ndim != 2 or crps.shape[1] < 2:
        raise ValueError("CRP 必须为二维数组，最后一列是 Response")
    if not np.all((crps == 0) | (crps == 1)):
        raise ValueError("CRP 必须只包含 0/1")
    _, indices = np.unique(crps, axis=0, return_index=True)
    unique = crps[np.sort(indices)]
    if len(np.unique(unique[:, :-1], axis=0)) != len(unique):
        raise ValueError("同一 Challenge 有不同 Response；请先确定重复测量的标签汇总规则")
    print(f"CRP rows: {len(crps)}, unique rows: {len(unique)}")
    return unique[:, :-1], unique[:, -1]


def main():
    parser = argparse.ArgumentParser(description="ECP-TRN：教学仿真或外部 CRP 训练")
    parser.add_argument("--crp-path", type=Path, help="合并 CRP 的 .npy/.csv；省略时生成教学数据")
    parser.add_argument("--test-crp-path", type=Path, help="固定测试 CRP；此时 crp-path 的全部数据用于训练")
    parser.add_argument("--metrics-path", type=Path, help="另存实验指标 JSON；拒绝覆盖已有文件")
    parser.add_argument("--order", type=int, help="单 APUF 为 1，K-XOR 为 K；外部默认 1，教学默认 3")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--test-size", type=float, default=0.25, help="外部 CRP 的测试比例")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    order = args.order if args.order is not None else (1 if args.crp_path is not None else 3)
    if min(order, args.rank, args.epochs) < 1 or not 0 < args.learning_rate < float("inf"):
        parser.error("order、rank、epochs 和 learning-rate 必须为正数")
    if not 0 < args.test_size < 1:
        parser.error("test-size 必须在 0 和 1 之间")
    if args.test_crp_path is not None and args.crp_path is None:
        parser.error("test-crp-path 必须与 crp-path 一起使用")
    if args.metrics_path is not None and args.metrics_path.exists():
        parser.error("metrics-path 已存在，请选择新的输出文件")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    bits, rank = 16, args.rank
    n_train, n_test = 5000, 1000
    epochs, learning_rate = args.epochs, args.learning_rate

    # 1. 外部模式沿用本项目的分层划分；教学模式无放回抽样。
    if args.crp_path is not None:
        import numpy as np
        from attacks.common import split_by_response

        try:
            raw_c, raw_y = load_external_crps(args.crp_path)
            if args.test_crp_path is None:
                train_c, test_c, train_y, test_y = split_by_response(
                    raw_c, raw_y, args.test_size, args.seed
                )
            else:
                train_c, train_y = raw_c, raw_y
                test_c, test_y = load_external_crps(args.test_crp_path)
                if train_c.shape[1] != test_c.shape[1]:
                    raise ValueError("训练集和测试集的 Challenge 位数不一致")
                all_c = np.concatenate([train_c, test_c])
                if len(np.unique(all_c, axis=0)) != len(all_c):
                    raise ValueError("训练集和固定测试集包含相同 Challenge")
                if min(len(train_y), len(test_y)) < 1:
                    raise ValueError("训练集和测试集不能为空")
        except (OSError, ValueError) as error:
            parser.error(str(error))
        bits = raw_c.shape[1]
        n_train, n_test = len(train_y), len(test_y)
        challenges = torch.tensor(np.concatenate([train_c, test_c]), dtype=torch.float32)
        labels = torch.tensor(np.concatenate([train_y, test_y]), dtype=torch.float32)
        print(f"source: {args.crp_path.resolve()}")
    else:
        # randperm(2**bits) 仅适合教学用的小位数。
        integers = torch.randperm(2**bits)[:n_train + n_test]
        challenges = ((integers[:, None] >> torch.arange(bits)) & 1).float()
        print(f"source: simulated {bits}-bit {order}-XOR APUF")

    # 2. challenge -> ±1 -> 从右向左累乘，最后追加常数 1。
    signs = 1 - 2 * challenges
    phi = signs.flip(1).cumprod(1).flip(1)
    phi = torch.cat([phi, torch.ones(len(phi), 1)], dim=1)

    # 3. 仅教学模式生成标签；外部模式使用文件中的 Response。
    if args.crp_path is None:
        puf_weights = torch.randn(order, bits + 1)
        component_responses = (phi @ puf_weights.T < 0).long()
        labels = (component_responses.sum(dim=1) % 2).float()
    train_x, test_x = phi[:n_train], phi[n_train:]
    train_y, test_y = labels[:n_train], labels[n_train:]
    print(f"bits: {bits}, order: {order}, rank: {rank}")
    print(f"train samples: {n_train}, test samples: {n_test}")

    # 4. 全批次训练：前向 -> BCE -> 自动求导 -> Adam 更新。
    model = ECPTRN(bits + 1, order, rank)
    criterion = nn.BCEWithLogitsLoss()  # 直接接收 logit，不提前做 sigmoid。
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    started = time.perf_counter()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"epoch {epoch + 1:3d}, train loss: {loss.item():.4f}")
    fit_seconds = time.perf_counter() - started

    # 5. 测试集只在训练结束后评分；logit >= 0 等价于 sigmoid >= 0.5。
    model.eval()
    with torch.no_grad():
        train_accuracy = ((model(train_x) >= 0) == train_y.bool()).float().mean().item()
        test_correct = int(((model(test_x) >= 0) == test_y.bool()).sum().item())
        test_accuracy = test_correct / n_test
        majority = train_y.mean() >= 0.5
        baseline = (test_y.bool() == majority).float().mean().item()
        train_one_rate = train_y.mean().item()
        test_one_rate = test_y.mean().item()
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"train accuracy: {train_accuracy:.2%}")
    print(f"test accuracy:  {test_accuracy:.2%}")
    print(f"test correct: {test_correct}/{n_test}, errors: {n_test - test_correct}")
    print(f"majority baseline (chosen from training labels): {baseline:.2%}")
    print(f"response distribution (test): 0={1 - test_one_rate:.2%}, 1={test_one_rate:.2%}")
    print(f"baseline always predicts: {int(majority.item())}")
    print(f"fit seconds: {fit_seconds:.2f}")
    if args.metrics_path is not None:
        import numpy as np

        metrics = {
            "crp_path": str(args.crp_path.resolve()) if args.crp_path else None,
            "test_crp_path": str(args.test_crp_path.resolve()) if args.test_crp_path else None,
            "bits": bits, "order": order, "rank": rank, "seed": args.seed,
            "epochs": epochs, "learning_rate": learning_rate,
            "n_train": n_train, "n_test": n_test,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "train_accuracy": train_accuracy, "test_accuracy": test_accuracy,
            "test_errors": n_test - test_correct, "majority_baseline": baseline,
            "majority_response": int(majority.item()),
            "train_label_one_rate": train_one_rate, "test_label_one_rate": test_one_rate,
            "fit_seconds": fit_seconds,
            "versions": {"torch": torch.__version__, "numpy": np.__version__},
        }
        args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_path.open("x", encoding="utf-8") as stream:
            json.dump(metrics, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
