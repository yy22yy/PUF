# 实验一 / 实验二：直接运行指南

这是精简、可运行的结构实验，**不是论文 Table I/II 的完整数值复现**。一个约200行的[脚本](<paper_baselines.py>)复用现有[PUF仿真](<../puf/xor_apuf.py>)；不依赖旧论文目录，不需要下载数据，也不需要TensorFlow。

## 1. 在终端进入项目

以下命令适用于你当前这台 Mac，可直接复制：

```bash
cd /Users/across/Desktop/PUF
.venv-torch/bin/python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
```

本机已验证该环境能运行。若换电脑或该环境不存在，只需Python 3.11及NumPy、PyTorch：

```bash
python3 -m venv .venv-torch
.venv-torch/bin/python -m pip install numpy torch
```

不需要激活虚拟环境，后面的命令直接使用它的Python。

## 2. 最简单：一次得到两个实验结果

```bash
.venv-torch/bin/python -m experiments.paper_baselines --output runs/my-first-experiments
```

默认：16-bit，k=2、3，训练CRPs=2000、6000，独立测试2000条，每组3个Run，100 epochs，batch256，Adam学习率0.01，ECP高秩R=5。本机通常几十秒内完成，机器不同时间不同。

终端先输出每个attempt的准确率和训练时间，最后输出各组成功率及成功attempt平均准确率。**输出目录必须是新目录**，第二次运行请换名字，例如 `runs/my-second-experiments`；不需要删除旧结果。

## 3. 分别运行实验一和实验二

### 实验一：Improved LR

```bash
.venv-torch/bin/python -m experiments.paper_baselines \
  --experiment lr --bits 64 --k 4 --train-sizes 30000 \
  --runs 3 --epochs 100 --output runs/my-exp01
```

实现结构：每条延迟链先tanh，k路相乘后再tanh，使用mini-batch Adam。由于当前PUF的0/1响应约定，最终符号按k的奇偶做固定转换。训练用BCE((输出+1)/2, 标签)。

这保留了Improved LR的关键结构，但采用PyTorch，不是原论文Keras；超参数、初始化、重复次数为本实验设置。默认小规模只证明可以跑通和得到结果，不代表论文数据复杂度结论。

### 实验二：ECP-TRN

```bash
.venv-torch/bin/python -m experiments.paper_baselines \
  --experiment ecp --bits 64 --k 4 --train-sizes 40000 \
  --rank 5 --runs 3 --epochs 100 --output runs/my-exp02
```

每Run中比较：

- Rank1独立训练5次；
- Rank5联合训练1次。

同一Run内PUF、训练集、测试集和每轮批次顺序相同，**只改变初始化**；不同Run更换PUF和数据。`--rank 10`就变成10次Rank1对一次Rank10。模型为k路线性投影乘积、R项加权求和，BCEWithLogitsLoss + Adam，无额外输出bias。

注意：相同训练轮数不表示相同wall-clock时间；Rank1策略每Run耗时应把R次attempt时间相加，不能只比较单attempt平均时间。

### 同时运行64-bit两个实验

```bash
.venv-torch/bin/python -m experiments.paper_baselines \
  --bits 64 --k 4 --train-sizes 40000 \
  --rank 5 --runs 3 --epochs 100 --output runs/my-64bit-experiments
```

这里两个方法共享40K训练数据，便于比较；不是Table I的k=4、30K那一行。

## 4. 怎样看结果

新输出目录包含：

- `summary.csv`：每个方法/k/N/Rank的汇总，最先看它。
- `attempts.csv`：每次训练的准确率、成功标记、训练秒数、种子、数据摘要。
- `submodels.csv`：实验二每个Rank分量的单项测试准确率。
- `config.json`：所有参数、版本、脚本摘要和运行状态。只有 `status=completed` 才是完整运行；失败/中断不要当完整结果。
- `*_data.npz`：实际训练/测试特征、标签和真值权重。攻击训练不读取真值权重。
- `*.pt`：每个attempt的模型参数；`*_predictions.npy`：对应测试预测。

可以用表格软件打开CSV；也可以终端显示汇总：

```bash
.venv-torch/bin/python - <<'PY'
from pathlib import Path
print(Path('runs/my-first-experiments/summary.csv').read_text())
PY
```

### 汇总指标

- 成功阈值：独立测试准确率 **严格大于90%**。
- `total_runs` / `successful_runs`：全部/成功Run数。实验一每Run只有一次attempt。
- `success_rate`：成功Run数 / 全部Run数。实验二一个Run内只要一个attempt成功即Run成功。
- `mean_success_accuracy`：所有成功attempt的准确率均值，**不含失败attempt**；无成功时CSV留空。
- `mean_accuracy`：全部attempt的准确率均值，避免只看成功者。
- `mean_attempt_seconds`：单次训练的平均耗时，包括mini-batch准备与优化，不含PUF数据生成、测试和保存。
- `n_train`：真实参与训练的CRP数量；`--test-size`是额外的测试数量，不是比例。

### 子模型分析

`submodels.csv`中的 `accuracy` 是该乘积项乘以训练所得alpha后，单独取符号的预测准确率。`sign_invariant_accuracy=max(acc,1-acc)`仅是允许翻转符号的事后诊断，**不能当成攻击的最终准确率**。约50%表示该分量单独没有明显预测力；多个分量也可能共同产生作用，不能预设“只能有一个分量有效”。测试数据不参与参数更新、早停或重试选择。

## 5. 修改规模与检查代码

```bash
# 多个k和训练规模会做笛卡尔积；先小后大
.venv-torch/bin/python -m experiments.paper_baselines \
  --bits 32 --k 3 4 --train-sizes 5000 10000 \
  --runs 3 --rank 5 --epochs 100 --output runs/my-scan

# 查看全部参数
.venv-torch/bin/python -m experiments.paper_baselines --help

# 全套测试（包含数学结构、成功率统计及CLI产物检查）
.venv-torch/bin/python -m unittest discover -s tests -p 'test_*.py'
```

增大k后可能训练失败或停留50%左右，这是有效实验结果，不是程序必须报错；不要用测试集筛选更好的种子。需要更多训练时可以另建目录、增加epochs或数据量，但要在报告中注明协议变化。

生成器保证训练/测试Challenge互斥且无重复；全部数据在内存中生成，**不适合直接运行论文150M–1B规模**。16-bit最多65536条不同Challenge，训练与测试之和不能超过该数。当前版本不增加GPU、早停、复杂配置系统或论文逐表批量调度，以保持代码简短。

## 6. 本轮已运行的结果入口

- [默认小规模汇总](<../runs/paper-experiments-small/summary.csv>)：16-bit、k=2/3、N=2000/6000。
- [64-bit实验汇总](<../runs/paper-experiments-64bit/summary.csv>)：64-bit、k=4、N=40000。

上述目录由本次实际运行产生，不是填入的论文数值。时间依赖机器，种子/数据和当前脚本版本可在各自配置及逐次记录中查到。早期 `paper-baselines-*` 目录是开发验证产物，不作为最终结果引用。

### 本机实际结果（2026-11-18）

64-bit、k=4、40K训练、2K测试、3个Run、100 epochs：

- Improved LR：成功3/3，成功attempt平均准确率 **98.10%**。
- ECP Rank1×5 attempts：Run成功3/3，15次attempt平均准确率 **96.44%**，每Run总训练时间平均约 **9.42秒**。
- ECP Rank5×1 attempt：Run成功3/3，平均准确率 **95.57%**，每Run训练时间平均约 **2.00秒**。

三个Rank5网络中，各有一个含alpha单项准确率较高，分别为 **95.60%、97.25%、96.75%**；其余项约47.55%–52.65%。这是这三个成功实例的事后观察，不保证所有设置都如此，也不能仅凭预测准确率声称已恢复真实权重。

默认16-bit实验还保留失败实例：k=3、N=6000的Improved LR成功 **1/3**，全部attempt平均准确率 **82.12%**，仅成功attempt平均准确率 **98.80%**。这说明不能把“成功者均值”误当作所有运行都达到的水平，也不保证样本增加时每次随机优化都更好。

最终105次训练的模型、预测、数据摘要和汇总均已重载复核；全套测试45项，2项因未安装TensorFlow跳过。
