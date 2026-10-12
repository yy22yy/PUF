# 单 APUF ECP 原型

本目录参考 `TENAX_source-tf2/TRL.py` 的 ECP 攻击层，使用项目现有的 CRP 和 `ChallengeFeatureTransformer`。参考目录中的 PUF 和数据生成代码不进入本流程。

## 路径配置

项目路径集中配置在仓库根目录的 `puf.toml`。训练入口从当前工作目录向上查找该配置，并以配置文件所在目录作为项目根目录；也可用 `--config /path/to/puf.toml` 或环境变量 `PUF_CONFIG` 指定配置文件位置。`[paths]` 中的 `default_crp`、`runs_dir` 等相对路径均相对配置文件解析。命令行传入的相对 `--crp` 和 `--output` 路径也同样相对配置文件，不依赖 shell 当前目录。

## NPZ 与 CSV 配套规则

程序每次生成 `name.npz`，必须同时生成 `name.csv` 作为数组清单，以及每个 NPZ 成员对应的 `name__成员名.csv`。清单记录成员名、shape、dtype 和成员 CSV 文件名；成员 CSV 保存该数组，首列为数组第一维索引，其余列为展开后的值。标量数组写为一行。修改 NPZ 写入代码时必须通过 `save_npz_with_csv`（参数模型通过 `save_parameters`），不可只写 NPZ。NPZ 和所有 CSV 都纳入运行 manifest 的 SHA-256 校验。CSV 是便于检查/交换的副本，精确类型及高维形状以 NPZ 为准。

## 模型与限制

ECP 的可训练参数按 CP 秩分解组织。对单个 APUF（一个分支），公式可化简为阶段特征上的线性函数；因此 rank=4 表示四个秩分量，并不代表 4-XOR PUF。此原型用于验证公式、两框架实现和实验记录，并以同一划分上的 LR 作为基线。

PyTorch 和 TensorFlow 都在 CPU 上以 float64 运算，使用同一个 NumPy 初始化、样本划分和每轮批次顺序。每次运行单独保存参数、配置、数据摘要、训练/测试行索引、逐轮训练指标、预测和最终指标。两套框架的 Adam 在 epsilon 细节上可能产生轻微差异；对照命令严格比较共同初始参数下的 logits、损失和梯度，不要求训练后的权重逐位相同。

保存的 `model.npz` 是框架无关参数格式，可以由 NumPy、PyTorch 或 TensorFlow 重新计算。原型保存最终参数，不含优化器状态，因此不能从中断的 epoch 恢复训练。当前仓库 CRP 的 PUF 权重和生成种子未知，记录只保存 CRP 文件摘要，不推断其来源。

## 环境

主项目环境只需 NumPy。两个攻击后端使用独立 Python 3.11 虚拟环境和独立依赖锁，避免将框架依赖带入现有工作流：

```bash
uv venv --python 3.11 .venv-torch
uv pip sync --python .venv-torch/bin/python requirements/ecp-torch.txt

uv venv --python 3.11 .venv-tensorflow
uv pip sync --python .venv-tensorflow/bin/python requirements/ecp-tensorflow.txt
```

依赖约束在 `requirements/ecp-torch.in` 和 `requirements/ecp-tensorflow.in`，固定版本在对应的 `.txt` 锁文件中。本次锁定目标是本机 macOS arm64、Python 3.11；在其他平台可按目标 Python 重新解析：

```bash
uv pip compile --python .venv-torch/bin/python --output-file requirements/ecp-torch.txt requirements/ecp-torch.in
uv pip compile --python .venv-tensorflow/bin/python --output-file requirements/ecp-tensorflow.txt requirements/ecp-tensorflow.in
```

锁文件目标为本机 macOS arm64、Python 3.11；PyTorch 2.14.1 和 TensorFlow 2.21.0 已安装在各自的忽略环境中。

## 使用

从项目根目录运行。先分别训练两套实现：

```bash
.venv-torch/bin/python -m attacks.ecp.train --backend torch --crp data/crps.npy
.venv-tensorflow/bin/python -m attacks.ecp.train --backend tensorflow --crp data/crps.npy
```

默认 rank=4、200 epochs、batch size=256、learning rate=0.001、测试集比例=0.25、种子=42。每次自动在配置中的 `runs_dir` 下建立全新 `ecp-...` 目录；可用 `--output` 指定一个尚不存在的目录，绝不覆盖已有运行或输入数据。也可通过 `--config` 选择另一份路径配置。

对照两套结果时，将下列路径替换为训练命令打印的实际目录：

```bash
.venv/bin/python -m attacks.ecp.compare --runs runs/ecp-...-torch-... runs/ecp-...-tensorflow-...
```

在 NumPy 主环境中重新加载模型并验证原记录：

```bash
.venv/bin/python -m attacks.ecp.evaluate --run runs/ecp-...-torch-...
```

也可将 `--backend` 设为 `torch` 或 `tensorflow`，用相应环境重新计算。重载会验证 CRP 和运行产物的摘要、划分、预测及指标；文件内容发生变化时会拒绝复算。

只运行数学检查，不启动训练：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_ecp*.py' -v
```

框架缺失时对应后端的梯度检查会跳过；安装框架后需要在两个环境分别执行，以验证各自的 autograd 实现。

## 当前验证记录

2026-10-02，使用仓库中的 10,000 条、32 位 Challenge CRP，按默认参数分别训练 200 epochs。两后端测试准确率均为 99.52%，同一划分的 LR 基线为 99.04%。两个独立实验目录为 `runs/ecp-backend-validation-torch/` 和 `runs/ecp-backend-validation-tensorflow/`。

两后端在共同初始参数下的最大绝对差值为：logits `8.9e-16`、loss `2.3e-16`、梯度不超过 `1.2e-16`。两个模型都通过原框架与 NumPy 的记录重载验证。完整数学/记录测试各运行 21 项，均通过；每个环境因另一框架未安装而跳过对方的 2 项测试。上述 `runs/` 和 `.venv-*` 内容由 Git 忽略，只保存在当前工作区。
