# PUF 仿真实现汇报

## 输入参数与当前产物

> **当前阶段：创建模型并生成权重**
>
> 创建 `APUF(n, seed)` 时：
>
> - **输入参数**：Challenge 位数 `n` 和随机种子 `seed`。
> - **生成产物**：保存在模型对象中的权重向量 `weights`，形状为 **`(n+1,)`**。
> - **当前状态**：**尚未输入 Challenge，也未计算特征、延迟或 Response。**

> **后续阶段：输入 Challenge 并计算响应**
>
> 调用 `puf.response(challenges)` 时：
>
> - **输入参数**：形状为 **`(m,n)`**、只含 `0/1` 的 Challenge 矩阵，其中 `m` 为样本数，`n` 与模型的 Challenge 位数一致。
> - **数据来源**：使用 [challenge.py](../challenge.py) 中的 `generate_challenges(num_challenges=m, challenge_length=n, seed=7)` 生成 Challenge。
> - **生成产物**：内部计算特征矩阵和延迟数组，最终返回形状为 **`(m,)`** 的 `0/1` Response 数组。
> - **执行条件**：**先生成 Challenge，再显式调用响应接口。** XOR-APUF 在创建模型时指定 `xor_count`，随后同样调用 `response(challenges)`，返回各子模型响应异或后的结果。

**两处 `seed` 的区别**：创建 `APUF(n, seed=42)` 时，种子控制模型权重；生成 Challenge 时，`seed=7` 控制输入数据。固定模型种子可复现同一组权重，固定 Challenge 种子可复现同一批输入（其余参数保持相同）。两者独立使用，数值可以相同，也可以不同；这里用 `42` 和 `7` 便于区分。XOR-APUF 的模型种子还用于派生各子 APUF 的种子。

| 阶段 | 输入 | 产生的数据 | 形状 |
| --- | --- | --- | --- |
| 创建 APUF 模型 | `n`、`seed` | 固定权重 `weights` | `(n+1,)` |
| 创建 XOR-APUF 模型 | `n`、`xor_count`、`seed` | `xor_count` 组两两不同的子模型权重 | 每组 `(n+1,)` |
| 调用 APUF 响应 | `challenges` | 特征 `features`、延迟 `delays`、响应 `responses` | `(m,n+1)`、`(m,)`、`(m,)` |
| 调用 XOR-APUF 响应 | `challenges`、`xor_count` | 组合响应 `xor_responses` | `(m,)` |

模型对象和上述计算结果都在内存中；这些模型类本身不会自动生成或保存 `.npy`、`.csv` 文件。要形成 CRP 数据文件，需要另外运行项目中的 Challenge/CRP 数据流程。

**“n-bit APUF”的含义**：本项目中的 `n` 指输入 Challenge 的位数。在标准 APUF 结构中，n 位 Challenge 分别控制 n 个级联开关级，每次输入产生 1 位 Response；线性加性延迟模型则使用 n+1 个等效权重参数。因此，“32-bit APUF”表示 32 位 Challenge、32 个开关级、每次 1 位响应，权重向量长度为 33。[APUF 结构与模型公式](https://link.springer.com/article/10.1186/s42400-024-00308-7)

同样，“32-bit 3-XOR APUF”表示 3 条各含 32 个开关级的 APUF 链使用同一个 32 位 Challenge，输出异或后仍为 1 位。这里 `n=32` 是 Challenge 长度，`xor_count=3` 是子模型数量，二者含义不同。[XOR-APUF 仿真参数说明](https://pypuf.readthedocs.io/en/latest/)

## 调用示例

下面分别创建一个 32 位 APUF，以及由 3 个 32 位 APUF 组成的 XOR-PUF。两段代码仅初始化模型并生成权重。

**创建一个 32 位 APUF**

```python
from puf import APUF

apuf = APUF(n=32, seed=42)

print(apuf.weights)        # 33 个等效延迟参数，末项是常数项系数
print(apuf.weights.shape)  # (33,)
```

**创建由 3 个 32 位 APUF 组成的 XOR-PUF**

```python
import numpy as np
from puf import XORAPUF

xor_puf = XORAPUF(n=32, xor_count=3, seed=42)
weights = np.stack([component.weights for component in xor_puf.components])

print(weights)        # 每行是一组子 APUF 的权重，共 3 组不同参数
print(weights.shape)  # (3, 33)
```

`n=32` 表示模型接受 32 位 Challenge；每个子 APUF 生成 33 个等效权重参数。`xor_count=3` 表示创建 3 个权重向量不同的 APUF 子模型，`seed=42` 用于复现模型参数。此时尚未输入 Challenge，也未计算 Response。

## 一、工作概述

本模块实现了基于延迟模型的仲裁器 PUF（APUF）及其 XOR 组合模型。模型初始化会生成固定权重；调用响应接口并传入由 `0/1` 组成的 Challenge 后，程序才会将其转换为阶段特征、计算模拟延迟，并按延迟符号输出 `0/1` Response。

**实现目标**

- 建立可重复的 APUF 软件仿真模型。
- 展示 Challenge 到 Response 的计算链路及其数学含义。
- 通过 XOR 组合提供多个 APUF 子模型的组合响应。

**核心结论**：单个 APUF 的输出由特征向量与固定权重的内积决定；XOR-APUF 对多个子模型的输出逐位异或。

## 二、整体计算流程

1. 输入 `(m,n)` 的 Challenge 矩阵，并检查维度、位数和元素取值。
2. 将每个 Challenge 从 `0/1` 映射为 `+1/-1`，计算阶段特征并补上偏置项，得到 `(m,n+1)` 特征矩阵。
3. 将特征矩阵与模型固定的权重相乘，得到 `(m,)` 延迟数组。
4. 按 `延迟 ≥ 0` 输出 `1`、`延迟 < 0` 输出 `0`，得到 `(m,)` Response。
5. 若使用 XOR-APUF，则对多个子 APUF 的 Response 逐位异或，得到最终 `(m,)` 输出数组。

对应的源文件职责如下：

| 文件 | 实现职责 |
| --- | --- |
| `puf/base.py` | 统一检查 Challenge，并定义模型接口 |
| `puf/features.py` | 将 Challenge 转换为 APUF 阶段特征 |
| `puf/apuf.py` | 生成固定权重并计算 APUF Response |
| `puf/xor_apuf.py` | 创建多个子 APUF 并组合其输出 |
| `puf/__init__.py` | 暴露包级接口 |
| `puf/__main__.py` | 命令行初始化模型并显示权重，支持 `python -m puf` |

## 三、关键数学模型

### 3.1 Challenge 到阶段特征

给定 `n` 位 Challenge：

```text
c = (c₁, c₂, ..., cₙ),  cᵢ ∈ {0, 1}
```

先把每一位映射到符号域：

```text
xᵢ = 1 - 2cᵢ
```

因此 `0 → +1`，`1 → -1`。第 `i` 阶段特征由该位及其右侧各位的符号乘积构成：

```text
φᵢ(c) = ∏ⱼ₌ᵢⁿ xⱼ,  i = 1, 2, ..., n
φ(c) = [φ₁(c), φ₂(c), ..., φₙ(c), 1]
```

向量末尾的 `1` 是偏置特征，因此 `n` 位 Challenge 对应 `n+1` 维特征向量。

**示例**：若 `c = (1, 0, 1)`，则 `x = (-1, +1, -1)`，阶段特征为：

```text
φ₁ = (-1)(+1)(-1) = +1
φ₂ = (+1)(-1) = -1
φ₃ = -1
φ(c) = [+1, -1, -1, 1]
```

### 3.2 APUF 延迟与响应

模型使用 `n+1` 个等效延迟参数组成权重向量，将特征与权重相乘求和，得到模拟延迟差：

```text
Δ(c) = φ(c) · w = Σᵢ₌₁ⁿ⁺¹ φᵢ(c)wᵢ
```

由于 `φₙ₊₁(c)=1`，最后一个系数 `wₙ₊₁` 在公式中作为常数项（也称偏置）。它属于 n 个开关级延迟参数组合得到的等效模型参数，不代表额外的第 n+1 个物理级。其余等效权重也不能简单理解成对应开关级的原始延迟。当前软件直接随机生成这些等效参数，没有逐级模拟实际电路路径。[等效权重推导见公式 (4)–(5)](https://link.springer.com/article/10.1186/s42400-024-00308-7)

响应通过零阈值量化为一位：

```text
r(c) = { 1,  Δ(c) ≥ 0
       { 0,  Δ(c) < 0
```

权重在创建 APUF 实例时由随机种子生成并固定。给定相同的 `n` 和 `seed`，可复现同一组软件模型权重。

### 3.3 XOR-APUF 组合

由 `k` 个 APUF 子模型共同处理同一 Challenge。每个子模型持有不同的权重向量 `w⁽q⁾`，分别计算响应，再对各自响应进行异或：

```text
r_XOR(c) = r₁(c) ⊕ r₂(c) ⊕ ... ⊕ rₖ(c)
```

`xor_count` 决定子模型数量；总种子用于派生子模型种子。初始化时会检查子模型权重向量，发现重复参数就重新生成该子模型，确保各 APUF 的权重向量两两不同。固定 `n`、`xor_count` 和 `seed` 可复现同一组子模型参数。

## 四、公式与代码的对应关系

### 4.1 特征变换：`puf/features.py`

代码使用有符号整数完成 `0/1 → +1/-1`，以保证 `-1` 可正确表示：

```python
signs = 1 - 2 * array.astype(np.int8)
```

通过先反转列、执行累积乘积、再反转回来，得到每一位到最右侧位的乘积：

```python
stage_features = np.cumprod(
    signs[:, ::-1],
    axis=1,
)[:, ::-1]

bias = np.ones((array.shape[0], 1), dtype=np.int8)
features = np.concatenate([stage_features, bias], axis=1).astype(np.float64)
```

其中 `axis=1` 表示按每个样本的 Challenge 位计算；添加偏置列后，若输入形状为 `(m,n)`，输出形状为 `(m,n+1)`。

### 4.2 延迟计算和阈值判决：`puf/apuf.py`

实例初始化时抽取 `n+1` 个权重，后续响应计算复用它们：

```python
rng = np.random.default_rng(seed)
self.weights = rng.normal(0.0, 1.0, size=self.n + 1)
```

批量计算特征内积，并将延迟符号转为整数响应：

```python
features = self.feature_transformer.transform(binary_challenges)
delays = features @ self.weights
responses = (delays >= 0.0).astype(np.uint8)
```

若输入有 `m` 行，则特征矩阵为 `(m,n+1)`，权重为 `(n+1,)`，延迟与响应均为 `(m,)`。

### 4.3 多模型异或：`puf/xor_apuf.py`

各组件先分别生成长度为 `m` 的响应，再堆叠并沿组件维度异或归约：

```python
component_responses = np.stack(
    [component.response(binary_challenges) for component in self.components],
    axis=0,
)
responses = np.bitwise_xor.reduce(component_responses, axis=0)
```

中间数组形状为 `(xor_count,m)`，归约后得到 `(m,)`。

## 五、汇报总结

本实现将 APUF 的阶段累积效应编码为特征向量，再通过线性加权和及阈值判决得到二进制响应。XOR-APUF 在多个独立 APUF 输出上执行逐位异或。代码结构与计算步骤一一对应，适合用于解释 PUF 仿真流程、构造 CRP 数据，以及作为后续建模攻击实验的响应生成模块。
