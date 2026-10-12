# LR 攻击：可执行说明

## 方法与代码

`attacks/apuf_lr.py` 用 Logistic Regression 对单个 APUF 建模。它先通过 `puf/features.py` 将每位 Challenge 转换为 APUF 阶段特征，再用批量梯度下降拟合二分类权重。损失包含 L2 正则项，偏置特征不受正则化。

相关文件：`attacks/apuf_lr.py`、`attacks/common.py`、`puf/features.py`、`project_config.py`。

## 环境与输入

在项目根目录安装主项目依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

输入为二维 CRP NPY 文件；每行的最后一列是 Response，其余列是二进制 Challenge。若没有 CRP，先按[数据流程说明](Challenge与CRP数据流程.md)生成 `data/crps.npy`。

## 运行

```powershell
python -m attacks.apuf_lr --crp-path data/crps.npy --test-size 0.25 --seed 42
```

省略 `--crp-path` 时读取 `puf.toml` 的 `default_crp`。用 `--config 路径` 或环境变量 `PUF_CONFIG` 可选择其他项目配置；配置中的相对路径以 TOML 文件所在目录为基准。

## 参数与输出

- `--crp-path`：输入 CRP NPY 文件。
- `--test-size`：测试集比例，默认 `0.25`，必须在 0 和 1 之间。
- `--seed`：训练/测试分层划分种子，默认 `42`。
- `--config`：项目 TOML 配置文件。

程序输出 CRP 总数、训练和测试样本数、两侧准确率及测试集多数类基线。LR 的优化器内部默认最多迭代 5000 次、学习率 `0.1`、正则化系数 `0.001`；这些值当前不作为命令行参数。查看命令帮助：

```powershell
python -m attacks.apuf_lr --help
```
