# Experiment 01–02：现有内容盘点与复现差距

- **检查日期**：2026-11-18
- **范围**：当前 PUF 项目的仿真、数据、攻击、实验脚本、测试与历史结果；对照论文全文中的 Section IV、Section V 及 Table I、Table II。
- **本轮工作**：文档整理与静态代码检查；未修改攻击代码，未启动训练，也未重新运行项目测试。
- **原始大纲**：[Experiment 01–02 实验大纲](<Experiment_01_02_实验大纲.md>)。
- **证据来源**：[论文可检索全文](<paper_fulltext.txt>)；全文文本存在双栏交错，以下按表头与对应行核对。

## 一、结论先行

**不是从零开始，但现有 LR 和正式 ECP 不能直接充当 Table I、Table II 的复现实现。**

1. **PUF 仿真与 CRP 流程可作为基础复用**：已有 Additive Delay Model、任意正整数 k 的 XOR-APUF、ATF 特征、固定种子、真值权重保存及数据摘要。
2. **Experiment 01 的核心攻击模型尚缺**：现有 LR 是单 APUF 的 NumPy 普通 LR，不是论文的 Improved LR。需要增加 k 条链、每链 tanh、乘积与外侧 tanh，以及 Keras、mini-batch、Adam 训练流程。
3. **Experiment 02 有两类可用起点**：教学示例已经表达一般 k 阶 ECP 乘积结构；正式双后端 ECP 有较完整的记录、校验和测试，但正式参数布局固定为单分支，需要重新对齐并扩展。
4. **已有结果不能直接填入论文表格**：现存 ECP 历史实验是 32-bit、单分支任务；现有 64-bit XOR 数据只有 k=3、100/500 条，适合流程冒烟检查，不符合目标表格的数据规模。
5. **还缺两张表的批量实验驱动及统计**：需要分别实现独立 PUF 重复实验、Attempt/Run 分层汇总、成功阈值、仅成功模型平均准确率和子模型诊断。

## 二、先修正实验协议中的关键细节

### 2.1 Experiment 01：Table I 原表矩阵

下表为论文目标值，**不是本项目已跑出的结果**。CRPs 指训练数据规模；测试数据需另行规定。

| k | 训练 CRPs | 成功数 / 独立攻击次数 | 论文训练时长 | Mean Success Accuracy |
|---:|---:|---:|---:|---:|
| 4 | 30K | 10/10 | 1 min | 95.5% |
| 5 | 260K | 10/10 | 4 min | 96.9% |
| 6 | 2M | 20/20 | 1 min | 97.6% |
| 7 | 20M | 10/10 | 3 min | 98.5% |
| 8 | 150M | 10/10 | 28 min | 96.4% |
| 9 | 500M | 7/10 | 14 min | 96.8% |
| 10 | 1B | 6/10 | 41 min | 95.7% |

来源：[Table I 表头、表注及数值](<paper_fulltext.txt#L315-L335>)。

注意：

- 不是每组统一 10 次：k=6 是 **20 次**。
- k≤8 使用 4 threads，k≥9 使用 40 threads；k=9/10 的论文内存用量分别为 **132/197 GiB**。不同 CPU、线程数与框架下的时间不能直接比较。
- duration 是 wall-clock training time；正文没有明确该列是所有运行均值、成功运行均值、中位数还是代表值，不应擅自称为“平均训练时间”。见[时间口径说明](<paper_fulltext.txt#L341-L348>)。
- Table I 的表头/表注未逐行明确 n；实施配置应显式记录 Challenge 位宽，并通过论文上下文或官方代码补核，而不是只记录 k。
- Table I 不足以单独确定精确 learning rate、batch size、epoch 上限、早停与 train/validation/test 划分。正文仅说明这些超参数重要，需继续核对官方实现或标注自定协议。见[超参数讨论](<paper_fulltext.txt#L360-L384>)。

论文 Improved LR 的结构是：

$$
f'_{\mathrm{LR}}(c)=\tanh\!\left(\prod_{\ell=1}^{k}\tanh\!\left(\langle W_\ell,x\rangle+b_\ell\right)\right).
$$

**新增的 tanh 在每条 arbiter chain 的线性延迟值上**，不是简单给普通 LR 随便增加一个激活函数。原文还要求 Keras 重实现、mini-batch 更新、Adam 替代 resilient backpropagation。见[方法定义](<paper_fulltext.txt#L261-L307>)。

### 2.2 Experiment 02：Table II 原表矩阵

全部为 **64-bit**。以下同样是论文目标值，不是本项目已有结果。

| k | 训练 CRPs | 模型 Rank | Attempts / Run | Total Runs | Run Success Rate | Mean Attempt Success Accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 40K | 1 | 5 | 10 | 100% | 95.2% |
| 4 | 40K | 5 | 1 | 10 | 100% | 94.6% |
| 5 | 320K | 1 | 10 | 10 | 90% | 94.9% |
| 5 | 320K | 10 | 1 | 10 | 90% | 94.9% |
| 6 | 800K | 1 | 10 | 10 | 80% | 95.1% |
| 6 | 800K | 10 | 1 | 10 | 80% | 95.0% |
| 7 | 800K | 1 | 100 | 5 | 20% | 95.4% |
| 7 | 800K | 1000 | 1 | 4 | 0% | — |

来源：[Table II](<paper_fulltext.txt#L397-L422>)。

需要修正原始大纲的地方：

1. **100 是 k=7 的 Attempts / Run，不是 Rank；高秩模型是 Rank 1000。** 正式矩阵中的 Rank 集合是 1、5、10、1000。
2. k=7 实际比较的是 Rank1×100 attempts 与 Rank1000×1 attempt，且 Total Runs 分别为 5、4；不是相等 R 预算。可另外增加等预算扩展实验，但不能替换原表后还称为逐行复现。
3. “固定同一个 PUF 和 CRP”明确用于多次重启的控制：同一目标的 attempts 之间仅改变初始化。正文未进一步清楚说明不同 total runs 之间是否都复用同一 token/数据，实施时需确认并记录这一点。见[重启实验设计](<paper_fulltext.txt#L373-L393>)。
4. Rank R 表示 **R 个 k-XOR 候选子模型联合训练**；不是 k，也不是 R 次独立训练。输出是 R 个乘积项的加权和再取符号，参数数为 $nkR+kR+R$。见[模型定义](<paper_fulltext.txt#L323-L347>)。
5. Table V 列出 TRN 使用 Adam、BCE、GlorotNormal，学习率为 multiple；没有给出一个全表统一的确切学习率。见[Table V](<paper_fulltext.txt#L684-L695>)。

### 2.3 指标口径

论文方法节的成功阈值为预测准确率 **>90%**；Fig.4 图注另写 ≥90%，复现时须显式选择并记录边界，不能在不同方法间混用。见[方法节](<paper_fulltext.txt#L256-L279>)与[Fig.4 图注](<paper_fulltext.txt#L750-L755>)。

建议在复现配置中明确如下统计定义，并注明这些是实施时形式化的约定：

- **Success Rate（Table I）**：成功独立攻击次数 / 全部独立攻击次数。例：7/10=70%，不是只统计成功模型后得到100%。
- **Mean Success Accuracy（Table I）**：仅成功攻击的测试准确率均值；没有成功时为缺失值，不写成0%。
- **Run Success Rate（Table II）**：成功 Run 数 / 全部 Run 数；独立 Rank1 多次尝试的一组是一个 Run，不能把 attempts 数当分母。
- **Run 成功的聚合规则**：按研究问题，可约定一个 Run 内至少一个 attempt 成功即 Run 成功；正文没有另给公式化定义，需在复现协议中明示该解释。
- **Mean Attempt Success Accuracy（Table II）**：成功 attempts 的平均预测准确率，不是全部 attempts 或全部 Rank 子模型的准确率均值；跨 Run 的具体加权方式与测试样本量仍需补核。
- **Training Time**：本地另存每个 attempt 的训练用时，并明确是否包括预处理、验证与记录开销；不要把整个脚本运行时间误当训练时间。
- **Data Complexity**：Table I/II 逐行 CRP 数是实验条件；Fig.4 的“成功率至少50%时的最小候选训练规模”是另一套搜索协议，不能直接混同。

## 三、项目中具体可以复用什么

### 3.1 PUF 仿真、特征与 CRP 生成：可以作为基础复用

- [APUF](<../puf/apuf.py#L9-L46>)：固定权重种子、正态权重、线性加性延迟 `phi @ weights`、0/1 响应。
- [XORAPUF](<../puf/xor_apuf.py#L9-L66>)：支持任意正整数 k，派生子 PUF 种子，对各链响应做逐位 XOR。对应大纲中的 PUF 实例构建与 k 扫描。
- [ChallengeFeatureTransformer](<../puf/features.py#L6-L48>)：0/1 Challenge 映射到 ±1，计算后缀乘积，追加偏置常数。64-bit 得到65维显式含偏置特征。
- [Challenge 生成工具](<../pufkit/challenge.py#L17-L39>)：可指定样本数、位宽与独立随机种子。
- [Response 与 CRP 流程](<../pufkit/crp.py#L27-L187>)：保存响应、真值权重、PUF/Challenge 种子、数组摘要、源文件摘要；合并时检查来源，防止不同批次错配。
- [数据读取与快照摘要](<../attacks/common.py#L10-L64>)：二进制取值检查、合并/分离输入、对同一份输入字节求摘要。

复用边界：

1. 本地模型使用0/1编码，论文使用±1；需单测两者的映射、响应符号与 XOR 奇偶关系，不能直接混用标签。
2. 目前是无噪声、等效权重独立标准正态的实现；应核对论文官方仿真的权重/偏置分布，而不是仅凭“都是 Additive Delay Model”就认定逐项一致。
3. 训练 CRP 数 N 不等于包含测试集的总行数。现有分层比例划分可作工具，但不能直接生成 N 条后留出25%，再声称使用了 N 条训练数据。
4. 真值权重只供生成与事后诊断，不能作为攻击模型输入。

### 3.2 Experiment 01：现有普通 LR 只适合基础校验

现有实现：[普通 LR](<../attacks/apuf_lr.py#L18-L39>)。

它使用：单个权重向量、sigmoid、全批量梯度下降、固定学习率、L2项、零初始化。

它缺少：k 条独立延迟链、每链 tanh、k路乘积、外侧 tanh、Keras 和 Adam。**即使输入 XOR-PUF 的 CRP 并运行成功，也不代表运行的是论文的 XOR Improved LR。**

可复用的是读取输入、特征处理和单 APUF 基础检查；正式 Improved LR 应作为独立攻击实现，避免改变原有普通 LR 的含义。

[现有神经网络训练代码](<../attacks/neural_net.py#L31-L137>)虽然有 mini-batch Adam，但网络为 ReLU 隐藏层与 sigmoid 输出，也不是 Improved LR 或论文 Aseeri/Mursi 模型的直接实现。

论文资料中的[官方代码线索](<README.md#L27-L38>)指向 pypuf 的 `2021-mlp` 分支及 `LRAttack2021`。本轮未下载或核验该外部版本；后续应先核对官方实现，再决定直接接入还是按相同协议移植。

### 3.3 Experiment 02：一般阶数教学模型 + 正式实验基础设施

#### A. 教学模型：可作为 k 阶 ECP 数学结构起点

[教学 ECPTRN](<../examples/ecp_trn_minimal.py#L19-L31>)支持 `order=k`、`rank=R`，先求每条链的投影、再对 k 条链相乘、最后对 R 项加权求和。比正式单分支模型更接近 Table II 的结构。

[外部固定训练/测试集入口](<../examples/ecp_trn_minimal.py#L56-L105>)可复用做冒烟检查；[Adam+BCE训练](<../examples/ecp_trn_minimal.py#L132-L151>)可作参考。

但不能直接当正式论文复现：这是 PyTorch 教学示例，默认小位宽、全批量、不同初始化，并包含额外输出 bias；完整 checkpoint、批量 Run/Attempt 统计和子模型诊断不齐。初始化、输出参数及损失对应关系都应按论文核对。

#### B. 正式 ECP：优先复用记录、校验与双后端测试设计

- [正式参数布局](<../attacks/ecp/common.py#L19-L49>)硬性要求 `factors=(rank,1,feature_count)`，拒绝多分支；Rank 不等于 XOR 数。
- [正式前向](<../attacks/ecp/torch_backend.py#L16-L32>)保留乘积框架，但因链数固定为1，最终等价于一个线性模型。这解释了为什么单 APUF 上较高准确率不能证明多 XOR 攻击有效。
- [初始化与 NumPy 前向](<../attacks/ecp/common.py#L38-L78>)、[PyTorch Adam/BCE](<../attacks/ecp/torch_backend.py#L40-L69>)以及[TensorFlow 后端](<../attacks/ecp/tensorflow_backend.py>)提供数值与后端校验的基础。
- [正式训练记录](<../attacks/ecp/train.py#L93-L154>)保存初始/最终参数、划分、预测、逐轮指标、配置、环境、代码与产物摘要、中断状态。
- [记录校验](<../attacks/ecp/records.py>)、[重载评估](<../attacks/ecp/evaluate.py>)、[跨后端比较](<../attacks/ecp/compare.py>)的思路可复用，但当前格式限定单 APUF，需同步扩展。

不能只把参数第二维由1改k就宣布完成：还需对齐论文每链偏置、输出是否含额外 bias、初始化、BCE输入、数值精度与训练停止规则，并新增多分支前向/梯度测试。

### 3.4 固定数据、仅改初始化：已有合适实验脚手架

[训练规模脚本](<../experiments/ecp_train_sizes.py#L96-L165>)已有：

- 独立 `split_seed`；
- 固定测试集；
- 固定训练顺序下的嵌套前缀；
- 模型初始化种子单独变化；
- 每次训练用时与预测保存。

这是 Table II 控制变量的合适基础，但目前只训练单分支 ECP，汇总所有初始化的准确率均值/标准差，并不计算目标成功指标；同时该脚本未保存训练后的模型参数，不能拿已有规模结果补做 Rank 子模型诊断。

另外，[正式训练入口](<../attacks/ecp/train.py#L59-L82>)把一个 seed 同时用于划分和初始化，shuffle用seed+1。**直接反复改变该入口的 `--seed` 会连数据划分一起改变，不满足仅改初始化的要求。**

### 3.5 现有数据与结果：适合验证流程，不是 Table I/II 的结果

- 现有数据有单 APUF 与 k=3 XOR-APUF，64-bit、100或500条。示例[500条 XOR 数据元数据](<../data/xor_apuf/k3_n64_w42_c42_m500_run01/crps_metadata.json#L3-L39>)记录的是 k=3、N=500，不是目标实验的 k=4–10。
- [训练规模复现配置](<../runs/ecp-train-sizes-repro/config.json>)和[汇总结果](<../runs/ecp-train-sizes-repro/summary.csv>)为32-bit、单分支、Rank4，不能改表头后用于Table II。
- [PyTorch 后端验证记录](<../runs/ecp-backend-validation-torch/config.json>)与[TensorFlow 后端验证记录](<../runs/ecp-backend-validation-tensorflow/config.json>)主要用于相同数据/初始化下检查两个后端，不是不同Rank的攻击比较。
- [历史结果来源说明](<../archive/2026-10-ecp-train-sizes/PROVENANCE.md#L10-L44>)已指出原始源数据/脚本缺失、一次重复运行中断、重实现不逐位一致。历史数据只作参考，不应当作新表格复现证据。

### 3.6 测试与环境

可复用测试设计：

- [CRP生成测试](<../tests/test_crp_generation.py>)：生成、种子、权重、元数据与数据配对。
- [LR输入测试](<../tests/test_apuf_lr.py>)：合并/分离CRP输入一致性与非法数据检查。
- [ECP数学测试](<../tests/test_ecp.py>)：NumPy参考、有限差分、参数往返、后端前向与梯度。
- [ECP记录测试](<../tests/test_ecp_records.py>)：记录与重载、输入/产物变更检测。

本轮未执行上述测试；现有覆盖不能替代新增 Improved LR、k>1 ECP、Run统计和子模型分析的测试。

依赖锁定文件已有[TensorFlow版本记录](<../requirements/ecp-tensorflow.txt#L27-L28>)、[TensorFlow框架版本](<../requirements/ecp-tensorflow.txt#L72-L73>)与[PyTorch版本记录](<../requirements/ecp-torch.txt#L21-L22>)。它们是当前项目环境声明，不是论文原始环境，也不代表本轮已核验实际安装成功。应单独记录复现环境，特别是旧Keras接口与当前版本的差异。

## 四、启动正式复现前，还缺哪些内容

### Experiment 01

1. 论文结构的 Improved LR 攻击实现及小规模单元测试。
2. Table I 逐行配置，正确区分 N_train 与额外测试样本；k=6重复20次。
3. 独立 PUF/token、Challenge、初始化、划分与shuffle种子的管理。
4. 全部攻击的逐次准确率、成功标记、训练时长与仅成功模型平均准确率。
5. 内存与批量数据处理方案。论文最高规模达1B训练CRPs，不宜直接把所有65维float64特征及CSV镜像一次性展开到内存/磁盘。

### Experiment 02

1. 64-bit、k=4/5/6/7的ECP训练实现，以及Rank1/5/10/1000的逐行配置。
2. 同一Run内固定目标与数据、独立Rank1重启、RankR联合训练两条分支。
3. Attempt与Run两级记录，显式记录成功阈值、失败/NaN/中断处理及聚合分母。
4. 全部Rank子模型的参数、单项预测、整体预测；分别报告原始单项与含alpha权重后的单项表现。
5. 成功网络中有效子模型个数、整体与单项准确率、移除其他项后的消融；若比较真实权重，应考虑链置换、符号和尺度不唯一性。
6. 匹配初始化或等训练预算可作为扩展分析，但要和Table II原始不等预算的k=7两行分开报告。

论文关于“恰好一个有效子模型，其余约50%”的直接观察针对 **k=4/5/6的成功网络**，不能外推成所有k和所有失败网络的预设结论。见[子模型分析原文](<paper_fulltext.txt#L351-L371>)。

## 五、建议的下一步顺序

1. **锁定协议**：将本文两张矩阵变成明确配置，补核官方实现中的损失、初始化、学习率/批量/停止条件、测试集规模和Run间数据关系。
2. **先做小规模正确性检查**：用现有生成器制作新数据；验证符号映射、k链乘积、梯度、保存重载、Attempt/Run统计，不把流程验证当成论文结果。
3. **先落地Experiment 01的低k基线**：从Table I的k=4、30K训练CRPs开始，确认Improved LR而非普通LR确实在运行。
4. **再做Experiment 02的k=4两行**：40K、Rank1×5与Rank5×1，补全子模型分析与记录后再扩展k=5/6。
5. **最后评估高成本行**：Table I的150M–1B与Table II的Rank1000，在内存、时间与磁盘预算明确后单独开展。

以上只是后续实施建议；本轮没有创建新攻击实现，也没有声称任何表格已复现。
