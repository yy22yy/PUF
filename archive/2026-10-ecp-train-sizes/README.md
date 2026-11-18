# 归档：ECP 训练规模扫描（2026-10）

单 APUF 的 ECP 攻击在不同 CRP 训练规模下的表现（6 个规模 × 3 个初始化种子 = 18 次训练），
外加一次未完成的重复运行。

## 结论

测试集 25001 条 CRP，固定划分、嵌套训练前缀：

| n_train | 测试准确率（3 次均值） |
| ---: | ---: |
| 1 000 | 98.67% |
| 5 000 | 99.28% |
| 10 000 | 99.50% |
| 20 000 | 99.68% |
| 50 000 | 99.80% |
| 74 998 | 99.85% |

单 APUF 的 ECP 在 rank=4、order=1 时等价于线性模型，因此这是"实现正确性 + 跨划分稳定性"
的验证，不是 ECP 对多分支 PUF 攻击能力的证据（见 `attacks/ecp/README.md` 的"模型与限制"）。

## 目录

```text
archive/2026-10-ecp-train-sizes/
├── README.md            本文件：结论与用法
├── PROVENANCE.md        来源、缺失项、可复现性说明（重要）
├── datasets/            归档的固定划分（npy，唯一副本）
│   ├── fixed_test.npy   n_test = 25001
│   └── train_<n>.npy    6 个嵌套训练前缀
├── primary/             已完成的主运行记录
│   ├── manifest.json    status=completed, completed_runs=18
│   ├── metrics.csv      18 行逐次指标
│   ├── summary.json     按 n_train 汇总
│   └── summary.csv
├── repeat/              未完成的重复运行（仅保留指标）
│   ├── manifest.json    status=running（中断，未补完）
│   └── metrics.csv      17 行（缺 1 次）
└── repro/               独立重实现的可复现结果
    ├── config.json
    ├── metrics.csv
    └── summary.json
```

## 如何重跑

```bash
# 需要 PyTorch 环境；约 13 秒
.venv-torch/bin/python -m experiments.ecp_train_sizes \
    --datasets archive/2026-10-ecp-train-sizes/datasets \
    --output runs/ecp-train-sizes-repro2
```

`repro/` 是 2026-11 用上述命令重跑得到的记录，与 `primary/` 的差异见 `PROVENANCE.md`。

## 数据说明

`datasets/*.npy` 每行 33 列：32 位 Challenge + 1 位 Response。
归档只保留 `.npy`；CSV 镜像可按需重建：

```bash
for f in archive/2026-10-ecp-train-sizes/datasets/*.npy; do
  .venv/bin/python -c "import numpy as np,sys; np.savetxt(sys.argv[1].replace('.npy','.csv'), np.load(sys.argv[1]), delimiter=',', fmt='%d')" "$f"
done
```