# ECP 攻击与教学示例：可执行说明

## 模块职责与文件

此板块有 9 个主要实现文件：

- `attacks/ecp/train.py`：统一训练入口。
- `attacks/ecp/torch_backend.py`、`attacks/ecp/tensorflow_backend.py`：两个训练后端。
- `attacks/ecp/common.py`：模型数学、数据和指标共用逻辑。
- `attacks/ecp/records.py`：可重载实验记录与 NPZ/CSV 产物。
- `attacks/ecp/evaluate.py`：重载模型并复算实验结果。
- `attacks/ecp/compare.py`：比较两种后端的实验记录。
- `attacks/ecp/__init__.py`：包入口。
- `ecp_trn_minimal.py`：独立 PyTorch 教学原型，不属于统一记录流程。

该 ECP 实验以单 APUF CRP 为输入。`rank` 是 CP 积分解的分量数；单 APUF 的 `order=1`，不表示多路 XOR PUF。正式入口会在新运行目录保存配置、数据摘要、划分、训练过程、模型、预测和指标。

## 环境准备

核心项目环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

ECP 训练后端按需安装。PyTorch：

```powershell
python -m pip install -r requirements/ecp-torch.txt
```

TensorFlow：

```powershell
python -m pip install -r requirements/ecp-tensorflow.txt
```

两个后端依赖较大，建议各自使用隔离虚拟环境。`requirements/ecp-torch.txt` 和 `requirements/ecp-tensorflow.txt` 的锁定目标平台见 `attacks/ecp/README.md`；Windows 等其他平台可从对应 `.in` 文件安装依赖：

```powershell
python -m pip install -r requirements/ecp-torch.in
```

或：

```powershell
python -m pip install -r requirements/ecp-tensorflow.in
```

## 训练

在仓库根目录、已激活相应环境后运行：

```powershell
python -m attacks.ecp.train --backend torch --crp data/crps.npy --epochs 200
```

或：

```powershell
python -m attacks.ecp.train --backend tensorflow --crp data/crps.npy --epochs 200
```

训练默认使用 `rank=4`、`batch-size=256`、`learning-rate=0.001`、`test-size=0.25`、`seed=42`。省略 `--output` 时，在配置的 `runs_dir` 下面创建新目录；如需固定目录，传入尚不存在的路径：

```powershell
python -m attacks.ecp.train --backend torch --crp data/crps.npy --output runs/ecp-torch-manual
```

程序拒绝覆盖已有运行目录。每个运行文件夹应保留完整，作为评估和比较的输入。

## 重载评估与后端对比

重载并使用 NumPy 复算：

```powershell
python -m attacks.ecp.evaluate --run runs/实际运行目录
```

需要用已安装的框架重新计算时，添加 `--backend torch` 或 `--backend tensorflow`。如 CRP 被移动，可以用 `--crp` 指定新路径；文件内容必须与原记录一致。

两种后端都训练完成后进行对比：

```powershell
python -m attacks.ecp.compare --runs runs/torch运行目录 runs/tensorflow运行目录
```

比较要求两次实验使用相同 CRP、训练协议、数据划分及初始化参数。运行命令的完整说明也见 [`attacks/ecp/README.md`](../attacks/ecp/README.md)。

## 独立教学原型

`ecp_trn_minimal.py` 可以使用小规模合成数据演示，也可读取外部 CRP：

```powershell
python -m pip install torch
python ecp_trn_minimal.py --epochs 50
python ecp_trn_minimal.py --crp-path data/crps.npy --order 1 --epochs 50
```

此脚本是简化教学示例，不生成正式 ECP 运行记录。其训练耗时和结果受硬件及依赖版本影响。
