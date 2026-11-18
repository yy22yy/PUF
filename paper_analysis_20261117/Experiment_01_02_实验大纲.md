# Experiment 01–02：Improved Logistic Regression 与 ECP-TRN 实验大纲

> 本文将用户提供的两个实验大纲整理为 Markdown，保留研究问题与实验意图。具体表格参数、指标定义及重复次数应以论文原文为准；本文件不表示实验已实施或已完成复现。

## 论文来源

- **论文标题**：Neural Network Modeling Attacks on Arbiter-PUF-Based Designs
- **发表信息**：IEEE Transactions on Information Forensics and Security（TIFS），2022
- **原文 PDF**：[论文原文](<Neural_Network_Modeling_Attacks_on_Arbiter-PUF-Based_Designs_TIFS2022.pdf>)
- **可检索全文**：[论文全文文本](<paper_fulltext.txt>)
- **中文材料**：[论文翻译与总结](<Neural_Network_Modeling_Attacks_论文翻译与总结.md>)
- **原文核对与项目盘点**：[现有内容盘点与复现差距](<Experiment_01_02_现有内容盘点与复现差距.md>)，其中列出两张表的逐行参数及对本大纲的修正说明。

## Experiment 01：Improved Logistic Regression

- **对应表格**：Table I
- **论文位置**：Section IV · PDF 第 4–5 页（按用户提供的位置）

### 实验目的

建立传统机器学习攻击的性能基线。

### 核心研究问题

不同 XOR 数量的 Arbiter PUF，需要多少 CRPs 才能被传统 LR 攻击成功建模？

### 实验内容

1. 使用 Additive Delay Model 构建 XOR Arbiter PUF。
2. 设置不同 XOR 数量 $k$。
3. 按照论文给定的数据规模生成 CRPs。
4. 实现 Improved LR 攻击。
5. 对不同 PUF 实例进行独立训练。
6. 使用独立测试集评估预测准确率。
7. 统计攻击成功率和平均成功准确率。

### 核心变量

- **XOR 数量**：$k$
- **CRP 数量**：$N$
- **随机性来源**：PUF 实例及模型初始化

### 评价指标

- Success Rate
- Mean Success Accuracy
- Training Time
- Data Complexity

### 预期产出

复现 Table I，建立后续 ECP-TRN、Aseeri MLP、Mursi MLP 的统一比较基准。

### 特别注意

论文的 Improved LR 不是原始 LR 的直接照搬。作者使用了 **Keras、Mini-batch、Adam 和额外的 tanh 激活**进行改进。复现时需要保留这些差异，不能用普通单层 Logistic Regression 的结果替代。

## Experiment 02：ECP-TRN

- **对应表格**：Table II
- **论文位置**：Section V · PDF 第 5–6 页（按用户提供的位置）

### 实验目的

验证高 Rank Tensor Regression Network 是否具有实际攻击优势。

### 核心研究问题

Rank $R$ 联合训练，是否比 $R$ 次独立 Rank 1 训练更有效？

### 实验内容

1. 实现 ECP-TRN。
2. 设置不同 Rank $R$。
3. 固定 64-bit XOR Arbiter PUF。
4. 固定同一个 PUF 实例和 CRP 数据集。
5. 分别执行 Rank $R$ 联合训练和 Rank 1 独立训练。
6. 改变模型初始化，重复训练。
7. 比较两种方案的攻击成功率。

### 核心变量

- **XOR 数量**：$k=4,5,6,7$
- **CRPs**：40K、320K、800K
- **Rank**：1、5、10、100
- **每次 Run 的训练尝试次数**

> 上述变量来自用户大纲，不意味着所有 $k$、$N$、$R$ 都做笛卡尔积；正式实验矩阵需按 Table II 逐行展开。

### 评价指标

- Run Success Rate
- Mean Attempt Success Accuracy

### 预期产出

复现 Table II，验证增加 Rank 是否真正提高攻击能力。

### 补充分析

还需要分析成功训练的 Rank $R$ 模型中，各个子模型是否都学习到了有效信息。

## 后续实施原则

1. 分离 PUF 权重种子、Challenge/CRP 种子、数据划分种子和攻击模型初始化种子。
2. 明确一次 Attempt、一次 Run 和独立 PUF 实例分别指什么，避免混用统计单位。
3. 训练集、验证集和测试集分别记录来源与用途，测试集不参与训练、早停或模型选择。
4. 原论文表格复现与扩展分析分别输出，额外记录 Training Time 和 Data Complexity 时注明测量口径。
5. 对照论文逐项确认超参数、成功阈值、重复次数和指标分母后，再启动正式批量实验。
