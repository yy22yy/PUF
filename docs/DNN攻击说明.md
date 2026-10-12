# DNN 攻击：可执行说明

## 方法与代码

`attacks/apuf_dnn.py` 使用三个隐藏层（128、64、32 个神经元）对单个 APUF 进行建模攻击。它与 MLP 一样直接读取 Challenge 位，并将输入从 0/1 映射到 -1/+1；不会预先生成 APUF 阶段特征，也不会读取仿真权重。

入口文件把层数传给共用实现 `attacks/neural_net.py`。该模块用 NumPy 完成 ReLU、反向传播和 Adam 参数更新。数据读取、分层划分及指标函数由 `attacks/common.py` 提供。

## 环境与输入

在仓库根目录准备 NumPy 主项目环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

输入是二维 CRP NPY 数组，每行最后一列为 Response。若需要先构造 CRP，请按[数据流程说明](Challenge与CRP数据流程.md)操作。

## 运行

```powershell
python -m attacks.apuf_dnn --crp-path data/crps.npy --epochs 300 --batch-size 64 --learning-rate 0.001 --test-size 0.25 --seed 42
```

省略 `--crp-path` 时程序读取配置中的 `default_crp`。需要其他路径配置时，可传入 `--config` 或设置 `PUF_CONFIG`。

## 参数与输出

- `--epochs`：训练轮数，默认 `300`。
- `--batch-size`：批次样本数，默认 `64`。
- `--learning-rate`：Adam 学习率，默认 `0.001`。
- `--test-size`：测试集比例，默认 `0.25`。
- `--seed`：数据划分和网络初始化种子，默认 `42`。
- `--crp-path`、`--config`：输入 CRP 和项目配置路径。

程序输出 DNN 结构、CRP 与训练/测试样本数量、训练准确率、测试准确率和多数类基线。权重只保留在本次进程内，不会由 CLI 写入文件。查看命令帮助：

```powershell
python -m attacks.apuf_dnn --help
```
