# PUF 模型与特征：可执行说明

## 模块职责

`puf/` 提供延迟型 PUF 软件模型和 APUF 特征转换。它本身不读写 CRP 实验文件。

主要文件共 6 个：

- `puf/base.py`：基础 PUF 接口。
- `puf/apuf.py`：单级 APUF 仿真。
- `puf/xor_apuf.py`：XOR-APUF 仿真。
- `puf/features.py`：Challenge 到 APUF 阶段特征的转换。
- `puf/__init__.py`：公开包接口。
- `puf/demo.py`：演示入口。

## 环境准备

在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

主项目需要 Python 3.11 或更高版本和 NumPy；`pip install -e .` 会按 `pyproject.toml` 安装项目及依赖。

## 运行演示

```powershell
python -m puf.demo --n 8 --num 4 --seed 42 --challenge-seed 7
```

演示会在终端输出模型计算结果。若只想查看可用参数：

```powershell
python -m puf.demo --help
```

## 在代码中调用

```python
import numpy as np
from puf import APUF, XORAPUF
from puf.features import ChallengeFeatureTransformer

challenges = np.random.default_rng(42).integers(0, 2, size=(10, 32), dtype=np.uint8)
apuf = APUF(n=32, seed=42)
responses = apuf.response(challenges)
features = ChallengeFeatureTransformer().transform(challenges)

xor_puf = XORAPUF(n=32, xor_count=3, seed=42)
xor_responses = xor_puf.response(challenges)
print(responses.shape, features.shape, xor_responses.shape)
```

固定 `seed` 可复现同一组软件仿真权重。运行此示例只在内存中生成数据，不会写入项目文件。
