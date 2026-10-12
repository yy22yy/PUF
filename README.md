# PUF 项目

项目各模块的文件清单和可执行命令见[模块运行说明](docs/模块总览.md)。

默认数据与实验目录集中配置在根目录 `puf.toml`。Challenge/CRP 生成脚本、LR/MLP/DNN 和 ECP 训练入口以配置文件所在目录作为项目根目录；可通过 `PUF_CONFIG=/path/to/puf.toml` 指定配置文件，各 CLI 入口也支持 `--config`。配置中的相对路径按 `puf.toml` 所在目录解析，而非依赖启动命令时所在目录。显式传入的相对文件路径也按配置文件位置解析。

所有 NPZ 文件必须配套 CSV：`name.npz` 对应 `name.csv`（数组清单）以及每个成员的 `name__成员名.csv`（展开后的数组数据）。NPZ 保留精确类型和形状，CSV 便于查看和交换。新增 NPZ 写入点时也必须遵循此规则；ECP 实验使用 `save_npz_with_csv` / `save_parameters`。

## 统一实验流程（后续规范）

后续 PUF 仿真、CRP 生成和攻击实验统一按以下流程组织。各步骤应由同一实验配置串联，并写入同一个独立运行目录；不得只保存最终准确率而丢失数据来源、参数或中间状态。

1. **读取实验配置**：确定数据路径、PUF 类型、攻击方法、训练/测试比例、随机种子和超参数。
2. **生成或加载 Challenge**：保存生成参数、随机种子和 Challenge 文件摘要；真实设备数据应记录采集条件。
3. **生成或加载 Response**：仿真时记录 PUF 类型、仿真参数、噪声和权重种子；真实硬件实验记录设备及采集条件，不推断未知参数。
4. **构建 CRP 数据**：按行配对 Challenge 与 Response，保存 CRP 并记录文件摘要。
5. **划分数据集**：记录训练集和测试集的原始行号、划分比例及随机种子，确保可以复现同一划分。
6. **训练攻击模型**：通过统一接口调用 LR、MLP、DNN、ECP 或后续新增的攻击实现。
7. **保存训练过程**：记录逐轮训练指标、检查点和运行状态；若要求中断后续训，同时保存优化器状态及当前训练步数。
8. **评估并保存结果**：保存测试集预测、评估指标和基线结果，并确保测试数据不参与训练或模型选择。
9. **生成实验清单**：记录配置、代码版本、运行环境、输入摘要、开始/结束时间、完成状态及所有产物的摘要。
10. **重载验证**：从实验记录重载模型与数据，复算指标和预测，并验证结果与已保存记录一致。

一次实验应能据此回答：用了什么 PUF 和数据、如何划分、运行了哪个攻击及参数、训练是否完成、结果是什么，以及是否能够重载复算。当前 ECP 原型已经实现其中一部分记录与重载验证；将 LR、MLP、DNN 和 PUF/CRP 生成流程接入统一运行记录，是后续完善方向。

## Challenge 生成

`challenge.py` 主要用于生成 Challenge 数组。它只负责生成由 `0` 和 `1` 组成的二维数组，不负责计算 PUF 的 Response。

数组的行表示 Challenge 的组数，列表示每组 Challenge 的位数。例如，生成 100 组、每组 64 位的 Challenge，数组形状为 `(100, 64)`。

### 参数说明

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--num` | 生成的 Challenge 组数，也就是数组的行数 | `100` |
| `--c_bit` | 每组 Challenge 的位数，也就是数组的列数 | `64` |
| `--seed` | 随机种子；固定为同一个值时，可重复生成相同数组 | `42` |

### 运行示例

```bash
python challenge.py --num 100 --c_bit 64 --seed 42
```

该命令生成 100 组、每组 64 位的 Challenge，并在终端显示数组及其形状。

固定随机种子便于实验复现；这只是软件生成的输入数据，不代表真实硬件的随机测量结果。

## Challenge 保存与读取

`challenge_data.py` 在 Challenge 数组生成之后，提供保存和读取功能。项目约定：生成或保存 `.npy` 数组时，同时生成同目录、同名的 `.csv` 文件，方便人工查看；读取功能目前从 `.npy` 文件加载数组。Challenge、Response 和合并后的 CRP 均遵循这个约定。

### 终端使用方法

运行下面的命令，生成 100 组、每组 64 位的 Challenge，并保存到 `data/challenges.npy` 和 `data/challenges.csv`：

```bash
python challenge_data.py --num 100 --c_bit 64 --seed 42 --path data/challenges.npy
```

程序会在指定路径写入 `.npy` 文件，并在同一目录下生成同名 `.csv` 文件。完成后，它还会重新读取 `.npy` 文件并显示数组形状。

### 参数说明

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--num` | 生成的 Challenge 组数 | `100` |
| `--c_bit` | 每组 Challenge 的位数 | `64` |
| `--seed` | 随机种子；固定后可重复生成相同数组 | `42` |
| `--path` | `.npy` 文件的保存路径；程序会自动生成同名 `.csv` | `data/challenges.npy` |
| `--overwrite` | 允许覆盖已存在的 `.npy` 或 `.csv` 文件 | 默认不覆盖 |

如果目标 `.npy` 或 `.csv` 文件已经存在，程序会停止以避免覆盖。确认要替换已有文件时，加上 `--overwrite`：

```bash
python challenge_data.py --num 100 --c_bit 64 --seed 42 --path data/challenges.npy --overwrite
```

也可以单独使用代码中的函数：`save_challenges()` 保存 `.npy`、`save_challenges_csv()` 保存 `.csv`，`load_challenges()` 从 `.npy` 读取，`generate_and_save_challenges()` 生成并保存 `.npy`。

## 生成 PUF Response

`generate_crp.py` 是一次实验的入口：它读取已经保存的 Challenge，创建一个 APUF 仿真实例，计算对应的 Response，并把 Response 同时保存为 `.npy` 和同名 `.csv` 文件。Challenge 和 Response 按行一一对应，因此两者可以组成 CRP（Challenge–Response Pair）。CSV 中每行保存一个 0/1 Response。

```bash
python generate_crp.py --challenge-path data/challenges.npy --response-path data/responses.npy --challenge-seed 7 --seed 42
```

`--challenge-path` 指定输入 Challenge 文件，`--response-path` 指定 Response NPY 输出文件，程序会在同一目录下用相同文件名生成 CSV，`--seed` 决定仿真 APUF 的权重；可选的 `--challenge-seed` 用于记录生成 Challenge 时的种子。程序还会单独保存 PUF 权重为 `responses_puf_weights.npy/.csv`，并写出 `responses_metadata.json`，记录模型配置、输入/输出数组摘要、Python/NumPy/平台信息，以及生成脚本和 APUF 实现的源文件摘要。权重是实验真值，不会被合并进攻击使用的 CRP 文件；不要把权重文件提供给攻击模型。

### 合并为 CRP 数据

Challenge 和 Response 分开保存时，按相同的行号配对：`challenges[i]` 对应 `responses[i]`。需要合并成一个文件时，可以运行 `merge_crp.py`：

```bash
python merge_crp.py --challenge-path data/challenges.npy --response-path data/responses.npy --output-path data/crps.npy
```

输出是同名的二维 `.npy` 和 `.csv` 数组，每行依次包含 Challenge 的所有位和最后一位 Response。例如 32 位 Challenge 对应的 CRP 矩阵形状为 `(样本数, 33)`。因此读取后可用 `crps[:, :-1]` 取得 Challenge，用 `crps[:, -1]` 取得 Response。合并时程序会检查样本数量一致，且 Response 只包含 0/1。

## 单个 APUF 的 Logistic Regression 攻击

`attacks/apuf_lr.py` 使用合并后的 CRP 训练一个 Logistic Regression 分类器，并用保留的测试集评估它对未见 Challenge 的预测准确率。模型只读取 CRP，不读取 APUF 的内部权重或生成种子。攻击会使用 `ChallengeFeatureTransformer` 生成的 APUF 阶段特征；逻辑回归的梯度下降由 NumPy 实现，不需要安装额外机器学习依赖。

```bash
python -m attacks.apuf_lr --crp-path data/crps.npy --test-size 0.25 --seed 42
```

默认将 25% 的 CRP 留作测试集，并按 Response 类别分层抽样。程序会报告训练准确率、测试准确率和测试集多数类基线。测试集准确率用于衡量对未见样本的预测效果；多数类基线可帮助判断模型是否优于简单地总猜同一类。当前 `data/crps.npy` 包含 10,000 条 32 位 Challenge 的 CRP；评估攻击效果时仍应报告数据来源、划分种子及测试条件。

## Challenge 特征转换

`puf/features.py` 中的 `ChallengeFeatureTransformer` 会把 0/1 Challenge 矩阵转换成 APUF 使用的特征矩阵。以一行为例，它先把 Challenge 位从 `0/1` 映射为 `+1/-1`：

```python
signs = 1 - 2 * array.astype(np.int8)
```

`astype(np.int8)` 把数组元素转成有符号的 8 位整数。对于普通的有符号整数数组，数学结果通常不变，因此在这种情况下可以省略转换。但 Challenge 数组也可能是 `uint8`（无符号整数）；无符号整数不能表示 `-1`，表达式 `1 - 2 * 1` 可能发生下溢，得到错误结果。因此这里显式转换为 `np.int8`，确保映射结果确实是 `+1` 和 `-1`。

接着，代码从右向左计算累积乘积：

```python
stage_features = np.cumprod(signs[:, ::-1], axis=1)[:, ::-1]
```

`signs[:, ::-1]` 将每一行的列顺序反转；`np.cumprod(..., axis=1)` 沿每行计算累积乘积；最后的 `[:, ::-1]` 再把结果反转回来。这样得到的每个位置都是该位到最右侧位的乘积，也就是 APUF 各阶段的特征。之后再添加一个偏置项 `1`，所以长度为 `n` 的 Challenge 会转换为 `n+1` 维特征。

## MLP 和 DNN 攻击

MLP 是多层感知机；DNN 泛指更深的神经网络，通常也属于 MLP。因此这里用两个不同深度的全连接网络作入门对比：MLP 使用一个 64 神经元隐藏层，DNN 使用三个隐藏层（128、64、32）。两者直接读取 Challenge 位并从 CRP 学习 Challenge 到 Response 的映射；与 LR 攻击不同，它们不预先使用 APUF 阶段特征。网络训练代码用 NumPy 实现，不需要额外机器学习依赖。

```bash
python -m attacks.apuf_mlp --crp-path data/crps.npy --epochs 300 --seed 42
python -m attacks.apuf_dnn --crp-path data/crps.npy --epochs 300 --seed 42
```

`attacks/apuf_mlp.py` 和 `attacks/apuf_dnn.py` 分别运行浅层 MLP 和较深的 DNN；两者共用 `attacks/neural_net.py` 的训练实现。`attacks/common.py` 提供 CRP 检查、相同的分层数据划分和准确率计算。三个入口都会留出测试集，并报告训练准确率、测试准确率和多数类基线。也可以调整 `--test-size`、`--batch-size` 和 `--learning-rate`。比较攻击效果时，使用相同数据集和划分参数即可复用同一训练/测试集。

## 单 APUF ECP 原型

`attacks/ecp/` 包含 PyTorch 和 TensorFlow 两种 ECP 实现及统一的实验记录流程。此原型针对单个 APUF，使用现有 `data/crps.npy`，并与同一训练/测试划分下的 Logistic Regression 对照。由于单分支 ECP 可化简为线性函数，该实验主要验证实现和跨框架一致性，不代表 ECP 在多分支 PUF 上的攻击能力。环境、运行、重载和对照命令见 [`attacks/ecp/README.md`](attacks/ecp/README.md)。

两种实现已在默认参数下完成 200 轮训练，测试准确率均为 99.52%，LR 基线为 99.04%；跨框架初始输出、损失和梯度检查通过。实验记录保存在本地 `runs/ecp-backend-validation-*` 目录。
