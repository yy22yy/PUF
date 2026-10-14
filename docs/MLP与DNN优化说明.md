# MLP 与 DNN 优化说明：使用 APUF 阶段特征

## 一、原来的输入是什么

原来的 MLP 和 DNN 都直接使用 Challenge 的各个二进制位，只进行 `0 → -1`、`1 → +1` 的映射。这是改变数值的表示方式，没有计算 APUF 的阶段组合特征。

例如，一条 3 位 Challenge 原本这样送给网络：

```text
Challenge：[0, 1, 0]
原始模式输入：[-1, +1, -1]
```

模型需要通过训练样本自己发现这些输入位之间的关系。当前浅层 MLP 在 5000 条单 APUF 数据上，训练准确率为 100%，测试准确率为 51.08%，说明它虽然拟合了训练样本，却没有有效学到对保留样本的预测规律。

## 二、优化的是输入表示

这次优化把每条 Challenge 转成 APUF 阶段特征，提前计算对该模型有用的组合关系。可以理解为：给网络整理过的输入材料，再让它学习如何预测响应。

仍以 `[0,1,0]` 为例。特征变换先采用 `0 → +1`、`1 → -1`：

```text
[0, 1, 0] → [+1, -1, +1]
```

然后从每个位置开始，将该位与右侧所有位相乘：

```text
第 1 个阶段特征：(+1) × (-1) × (+1) = -1
第 2 个阶段特征：       (-1) × (+1) = -1
第 3 个阶段特征：              (+1) = +1

特征模式输入：[-1, -1, +1]
```

对应的 Response 不变。特征变换只读取 Challenge，没有读取这条数据的答案，也没有读取目标 PUF 的真实权重或权重生成种子。

64 位 Challenge 同样生成 64 个阶段特征。它们与单 APUF 的计算结构相关，因此模型不必完全从原始位中自己发现这些组合关系。MLP 和 DNN 使用同一个输入处理函数，优化思路一致。

### 为什么去掉最后的常数列

项目的 `ChallengeFeatureTransformer` 输出阶段特征后，还会在末尾加一个固定的 `1`，供 LR 表示偏置。对于上面的例子，完整输出为 `[-1,-1,+1,1]`。

MLP、DNN 每层已经有独立偏置，所以这里只保留前三个阶段特征。64 位 Challenge 仍对应 64 个网络输入，避免多加一个常数输入节点。

## 三、代码如何实现

修改位于 [attacks/neural_net.py](../attacks/neural_net.py)。新增共用的 `prepare_features()`，默认模式为 `raw`，原有命令仍使用原始位映射：

```python
# 原始模式：逐位映射，不计算累积乘积
return 2.0 * np.asarray(challenges, dtype=np.float64) - 1.0
```

选择 `apuf` 模式时，使用现有变换器计算阶段特征：

```python
# 计算阶段特征；[:, :-1] 保留所有行并去掉末尾常数列
return ChallengeFeatureTransformer().transform(challenges)[:, :-1]
```

数据先按原来的规则划分，再对训练和测试输入分别执行同一种处理：

```python
# 使用指定输入表示，标签和样本对应关系保持不变
train_x = prepare_features(train_x, args.features)
test_x = prepare_features(test_x, args.features)
```

使用训练好的网络预测新 Challenge 时，也必须先调用同一模式的 `prepare_features()`。不能用阶段特征训练，却在预测时输入原始位。

### 如何做到公平对照

两种模式只改变输入表示。使用相同数据、划分种子和测试比例，保证训练与测试样本一致；使用相同网络结构、初始化种子、训练轮数、批次大小、学习率和批次打乱种子。

由于去掉了常数列，两种模式的输入维度相同，各层参数形状也相同。相同初始化种子能得到相同的初始权重和偏置。MLP 仍是一个 64 神经元隐藏层，DNN 仍是三个隐藏层 `(128,64,32)`；对于 64 位输入，参数量分别仍为 4225 和 18689。

两种模式的损失仍为二元交叉熵，更新器仍为 Adam，输出仍用 `0.5` 判决。因此，对照主要考察阶段特征对预测效果的影响。

## 四、MLP 如何运行

以下命令在项目根目录执行，样本数以输入文件的实际行数为准，目录名中的 `m500` 是手动标签。下文记录的优化对照使用了 5000 条 CRP；重新生成 500 条数据后，使用这些命令运行得到的是新的 500 条实验结果。

```powershell
# 原始输入模式，作为对照
python -m attacks.apuf_mlp --crp-path data/apuf/n64_w42_c42_m500_run01/crps.npy --features raw --epochs 300 --batch-size 64 --learning-rate 0.001 --test-size 0.25 --seed 42
```

```powershell
# 优化模式：使用 APUF 阶段特征，其余训练参数与对照相同
python -m attacks.apuf_mlp --crp-path data/apuf/n64_w42_c42_m500_run01/crps.npy --features apuf --epochs 300 --batch-size 64 --learning-rate 0.001 --test-size 0.25 --seed 42
```

## 五、DNN 如何运行

```powershell
# 原始输入模式，网络为三个隐藏层
python -m attacks.apuf_dnn --crp-path data/apuf/n64_w42_c42_m500_run01/crps.npy --features raw --epochs 300 --batch-size 64 --learning-rate 0.001 --test-size 0.25 --seed 42
```

```powershell
# 优化模式：三个隐藏层使用 APUF 阶段特征作为输入
python -m attacks.apuf_dnn --crp-path data/apuf/n64_w42_c42_m500_run01/crps.npy --features apuf --epochs 300 --batch-size 64 --learning-rate 0.001 --test-size 0.25 --seed 42
```

终端第一行会显示 `输入特征：raw` 或 `输入特征：apuf`，用于辨认实验模式；随后输出样本数、训练准确率和测试准确率。

## 六、如何判断优化效果

主要比较同一测试集上的准确率，而不是只看训练准确率。训练 100% 说明模型拟合了已用样本；测试准确率提高，才说明对这批保留样本的预测有所改善。

本次 5000 条单 APUF 数据划分为 3749 条训练、1251 条测试。四次对照都使用上述 300 轮、批次 64、学习率 0.001、种子 42 的设置，训练准确率均为 100%。测试结果为：

- **MLP**：原始模式 51.08%，APUF 特征模式 96.00%。
- **DNN**：原始模式 53.40%，APUF 特征模式 94.72%。

这支持阶段特征在本次配置下帮助了预测，也说明更深的 DNN 在本次测试中并未超过较浅 MLP。但这些是单次固定种子实验，不意味着所有数据、种子或 PUF 类型都得到同样结果。

对于 XOR-PUF，阶段特征对应各子 APUF 的输入表示，最终响应还包含多模型的异或组合，其攻击效果需要另行实验。当前程序仍只在内存中保留预测模型，没有保存模型参数或损失曲线。
