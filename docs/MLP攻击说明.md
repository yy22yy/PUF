# MLP 攻击：可执行说明

## 方法与代码

`attacks/apuf_mlp.py` 使用一个含 64 个神经元的隐藏层，对单个 APUF 进行建模攻击。网络直接读取 Challenge 位，将 0/1 输入映射到 -1/+1；不会使用 APUF 阶段特征，也不会读取 PUF 仿真权重。

模型结构和 NumPy 训练循环位于 `attacks/neural_net.py`，CRP 校验、分层划分和指标函数位于 `attacks/common.py`。该实现不需要 PyTorch 或 TensorFlow。

## 环境与输入

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

输入为二维 CRP NPY 文件，最后一列为 Response。准备数据的步骤见[Challenge/CRP 数据流程](Challenge与CRP数据流程.md)。

## 运行

```powershell
python -m attacks.apuf_mlp --crp-path data/crps.npy --epochs 300 --batch-size 64 --learning-rate 0.001 --test-size 0.25 --seed 42
```

`--crp-path` 可省略，此时读取 `puf.toml` 中的默认 CRP 路径。划分与初始化由 `--seed` 固定；批次打乱使用 `seed + 1`。

## 参数与输出

- `--epochs`：训练轮数，默认 `300`。
- `--batch-size`：每次参数更新使用的样本数，默认 `64`。
- `--learning-rate`：Adam 学习率，默认 `0.001`。
- `--test-size`：测试集比例，默认 `0.25`。
- `--seed`：划分和网络初始化种子，默认 `42`。
- `--crp-path`、`--config`：输入数据及项目配置路径。

终端会显示模型名、隐藏层、样本数、训练准确率、测试准确率和多数类基线。该 CLI 不保存训练权重。完整参数说明：

```powershell
python -m attacks.apuf_mlp --help
```
