# PUF 项目

Arbiter PUF（APUF）与 XOR-APUF 的软件仿真，以及针对它们的建模攻击实验：
Logistic Regression、MLP、DNN、ECP（PyTorch / TensorFlow 双后端）。

所有数据由固定种子生成，所有实验都要求可重跑：**一个实验 = 一个目录 + 一份 manifest**。

## 实验一 / 实验二（新）

Improved LR 与多分支 ECP-TRN 的精简可运行实验、终端命令和实际结果见 [实验运行指南](<experiments/README.md>)。只需 NumPy + PyTorch，可先跑小规模，再运行64-bit 4-XOR；不宣称完整复现论文数值。

## 快速开始

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .          # Windows: .\.venv\Scripts\activate 后再 pip install -e .

RUN=data/apuf/n64_w42_c42_m500_run01          # 参考数据已随仓库提供，可直接跳到攻击步骤

# 1) Challenge → 2) Response → 3) CRP
python -m pufkit.challenge --num 500 --c_bit 64 --seed 42 --path "$RUN/challenges.npy" --overwrite
python -m pufkit.crp generate --challenge-path "$RUN/challenges.npy" \
    --response-path "$RUN/responses.npy" --challenge-seed 42 --seed 42
python -m pufkit.crp merge --challenge-path "$RUN/challenges.npy" \
    --response-path "$RUN/responses.npy" --output-path "$RUN/crps.npy"

# 4) 攻击
python -m attacks.apuf_lr --crp-path "$RUN/crps.npy" --test-size 0.25 --seed 42
```

完整命令（含 XOR-PUF、PowerShell 变量写法、MLP/DNN 参数）见 [直接运行](直接运行.md)。

## 目录结构

```text
PUF/
├── puf/              PUF 模型与特征（APUF、XORAPUF、ChallengeFeatureTransformer）
├── attacks/          LR / MLP / DNN 基线；attacks/ecp/ 为 ECP 双后端实现
├── pufkit/           数据流程基础设施：配置、落盘约定、Challenge、CRP
├── experiments/      可复现实验脚本（一个脚本 = 一个实验，产物写 runs/）
├── examples/         独立教学示例（不参与正式实验）
├── tests/            unittest 用例（无需 CRP 数据即可运行）
├── docs/             中文说明文档（模型、数据流程、各攻击方法）
├── data/             已生成的 CRP 数据，按 PUF 类型与运行编号分目录
├── runs/             实验产物目录（本地生成、不入库，契约见 runs/README.md）
├── archive/          历史产物归档（附 PROVENANCE，说明能否重跑）
├── papers/           论文研读资料（PDF、翻译、复现笔记）
├── puf.toml          数据与运行目录配置
└── pyproject.toml    依赖与工具配置
```

模块清单与推荐运行顺序见 [模块总览](docs/模块总览.md)。

## 环境

主项目只需要 NumPy；ECP 的两个后端各自独立，避免把深度学习框架带进基础环境：

| 环境 | 用途 | 关键依赖 |
| --- | --- | --- |
| `.venv` | 主环境：模型、数据流程、LR/MLP/DNN、测试 | `numpy`（`pip install -e .`） |
| `.venv-torch` | ECP PyTorch 后端（`attacks.ecp.train --backend torch`） | `requirements/ecp-torch.txt` |
| `.venv-tensorflow` | ECP TensorFlow 后端（`--backend tensorflow`） | `requirements/ecp-tensorflow.txt` |

```bash
# 可选：开发依赖（pytest / ruff）
.venv/bin/python -m pip install -e ".[dev]"

# 测试：主环境即可，41 个用例，框架相关用例在缺少框架时自动跳过
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'

# ECP 后端环境（版本锁在 requirements/*.txt，目标 macOS arm64 + Python 3.11）
uv venv --python 3.11 .venv-torch
uv pip sync --python .venv-torch/bin/python requirements/ecp-torch.txt
```

三个虚拟环境合计约 2 GB，已在 `.gitignore` 中排除，可随时按上表重建。

## 数据与产物约定

### 数据：`data/<puf_type>/<run>/`

目录名记录生成参数：`n64`（Challenge 位数）、`k3`（子 PUF 数，仅 XOR）、`w42`（权重种子）、
`c42`（Challenge 种子）、`m500`（样本数）、`run01`（运行编号）。新实验用新编号，不覆盖旧数据。

每个目录固定包含 `challenges`、`responses`、`puf_weights`、`crps` 四组数组，以及
`responses_metadata.json`、`crps_metadata.json` 两份来源记录。

### 数组落盘：`.npy` 必须配同名 `.csv`

`name.npy` 写入时必须同时写 `name.csv`（便于人工查看与交换）；精确类型与形状以 NPY 为准。
ECP 的运行记录使用 `.npz` + 数组清单 CSV + 每个成员的 CSV（见 [ECP 实现说明](attacks/ecp/README.md)）。

### 实验产物：`runs/<experiment-id>/`

每个实验一个新目录，至少包含 `manifest.json`（状态、时间、数据摘要、环境、源文件摘要、产物摘要）、
`config.json`、逐轮 `history.csv`、`metrics.json` 与预测文件。契约见 [runs/README.md](runs/README.md)。
`runs/` 默认不入库；需要长期保留的结论应归档到 `archive/` 并写清来源。

### 真值权重不是攻击输入

`puf_weights.npy` 是实验真值，用于生成 Response 和事后核对；攻击入口只读取 CRP，
不得把权重或模型种子交给攻击模型。

## 攻击方法一览

| 方法 | 入口 | 输入特征 | 额外依赖 |
| --- | --- | --- | --- |
| Logistic Regression | `python -m attacks.apuf_lr` | APUF 阶段特征 | 无（NumPy 实现） |
| MLP（1×64） | `python -m attacks.apuf_mlp` | `--features raw/apuf` | 无 |
| DNN（128-64-32） | `python -m attacks.apuf_dnn` | `--features raw/apuf` | 无 |
| ECP-TRN | `python -m attacks.ecp.train` | APUF 阶段特征 | `.venv-torch` 或 `.venv-tensorflow` |

三种 NumPy 入口都会留出测试集、报告训练/测试准确率；ECP 还会写出完整运行记录并支持重载复算。
单分支 APUF 的 ECP 在数学上可化简为线性模型，因此该实验用于验证实现与跨框架一致性，
不代表 ECP 对多分支 PUF 的攻击能力。

## 实验规范

后续 PUF 仿真、CRP 生成和攻击实验统一按以下流程，任一步缺失都视为实验不完整：

1. 读取实验配置（数据路径、PUF 类型、攻击方法、划分比例、种子、超参数）；
2. 生成或加载 Challenge（记录生成参数与摘要）；
3. 生成或加载 Response（记录 PUF 参数、噪声、权重种子）；
4. 构建 CRP（按行配对，记录摘要）；
5. 划分数据集（记录原始行号、比例、种子）；
6. 训练攻击模型（通过统一入口）；
7. 保存训练过程（逐轮指标、检查点、运行状态）；
8. 评估并保存结果（测试预测、指标、基线）；
9. 生成实验清单（配置、代码版本、环境、输入摘要、时间、状态、产物摘要）；
10. 重载验证（从记录复算指标，与已保存结果一致）。

一次实验应能回答：用了什么 PUF 和数据、如何划分、跑了哪个攻击及参数、训练是否完成、
结果是什么、能否重载复算。

## 文档导航

- [模块总览](docs/模块总览.md)：模块职责、文件速查、推荐运行顺序
- [PUF 仿真说明](docs/PUF仿真说明.md)：APUF / XOR-APUF 模型与特征转换
- [Challenge 与 CRP 数据流程](docs/Challenge与CRP数据流程.md)：数据准备的可执行步骤
- [直接运行](直接运行.md)：从零生成数据并跑通 LR/MLP/DNN 的完整命令
- 攻击方法：[LR](docs/LR攻击说明.md)、[MLP](docs/MLP攻击说明.md)、[DNN](docs/DNN攻击说明.md)、[ECP](docs/ECP攻击说明.md)
- [MLP 与 DNN 优化说明](docs/MLP与DNN优化说明.md)
- [ECP 实现与实验记录](attacks/ecp/README.md)
- [实验产物契约](runs/README.md)
- 历史归档：[2026-10 ECP 训练规模扫描](archive/2026-10-ecp-train-sizes/PROVENANCE.md)
- 论文研读：[论文索引](papers/index.md)