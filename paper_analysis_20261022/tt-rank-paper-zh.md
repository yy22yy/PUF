# TT 秩预算的结构依据与容量权衡：TENAX 中文论文补充稿

版本：2026-11-04，中文定稿（补充符号说明）。对应 [完整推导与思考版](tt-rank-derivation-zh.md) 和 [英文论文稿](tt-rank-paper-en.md)。插入位置及逐项理论依据见完整推导稿附录；符号说明后的三节为论文正文。

## 符号说明

| 符号 | 含义与约定 |
|---|---|
| \(x\) | 权重张量的阶数，也是 TT 核的数量；在本文模型中对应 XOR 链数。 |
| \(n\) | 输入 challenge 的位数。 |
| \(p_j\)、\(p\) | 第 \(j\) 模的长度；等模长时各 \(p_j=p\)，TENAX 扩维后 \(p=2(n+1)\)。 |
| \(k\)、\(j\)、\(i_j\) | \(k\) 为内部连接位置，取值为 1 至 \(x-1\)；\(j\) 为模位置；\(i_j\) 为第 \(j\) 模的元素索引。 |
| \(\mathcal W\)、\(W\) | TT 表示的权重张量；式 (P4) 中的 \(W\) 表示网络结构允许的任一权重张量。 |
| \(X\) | 经过固定特征映射得到的输入张量，与权重张量具有相同形状。 |
| \(\mathcal W^{\langle k\rangle}\) | 第 \(k\) 个连续展开矩阵：前 \(k\) 个模组成行索引，其余模组成列索引。 |
| \(L_k\)、\(B_k\) | 分别收缩前 \(k\) 个核和其余核得到的左右矩阵因子，其乘积为上述展开矩阵。 |
| \(r_k\) | 第 \(k\) 个连续展开矩阵的实际秩，即该位置的实际 TT 秩，由权重取值决定。 |
| \(R_k\)、\(R\) | \(R_k\) 为人为设置的第 \(k\) 条连接尺寸，称为秩预算；\(R\) 表示所有内部连接采用的统一预算。边界固定为 \(R_0=R_x=1\)。 |
| \(U_k\) | 第 \(k\) 个展开由行列尺寸决定的秩上界；等模长时为 \(p^{\min(k,x-k)}\)。实际秩同时受 \(R_k\) 与 \(U_k\) 约束。 |
| \(\alpha\)、\(A_\alpha\)、\(B_\alpha\) | \(\alpha\) 为可分离项编号；\(A_\alpha\) 与 \(B_\alpha\) 分别依赖切分左侧和右侧的索引组。 |
| \(G\)、\(V\)、\(\lvert V\rvert\) | 固定张量网络结构、其核节点集合及节点数；TT 链中节点数等于 \(x\)。这里的 \(G\) 不是单个核张量。 |
| \(N_G\)、\(\mathcal T(G)\) | 结构 \(G\) 的核参数总数，以及该结构可表示的权重张量集合。 |
| \(h\)、\(\mathcal H_G^{\mathrm{classif}}\) | 单个分类器，以及由结构 \(G\) 参数化的符号内积分类器族。 |
| \(S\)、\(m\) | 独立同分布的训练样本集合及其样本数。 |
| \(\mathcal R(h)\)、\(\widehat{\mathcal R}_S(h)\) | 分类器的总体风险和在样本 \(S\) 上的经验风险；二者之差为泛化间隙。 |
| \(\operatorname{VCdim}\)、\(\operatorname{sign}\)、\(\langle W,X\rangle\) | 分别表示 VC 维、符号函数和张量内积。 |
| \(O(\cdot)\)、\(\log_2\) | 分别表示渐近上界量级和以 2 为底的对数；式 (P5) 的 \(\log\) 在量级表达中不区分固定底数。 |

## 1. 对称秩预算的结构依据

在前述 TT 表示中，连接尺寸不仅决定核形状，也限制了相应连续分割处的表达容量。将前 \(k\) 个核及其余核分别收缩，可以把权重张量的连续展开表示为内维数为 \(R_k\) 的矩阵乘积。记其实际展开秩为 \(r_k\)，则

$$
\mathcal W^{\langle k\rangle}=L_kB_k,\qquad
r_k:=\operatorname{rank}(\mathcal W^{\langle k\rangle})
\le\min\left\{R_k,\prod_{j=1}^{k}p_j,\prod_{j=k+1}^{x}p_j\right\}.
\tag{P1}
$$

其中 \(x\) 为张量阶数，\(p_j\) 为第 \(j\) 模长度，\(R_k\) 为设置的连接预算，边界满足 \(R_0=R_x=1\)。该展开将前 \(k\) 个模组成行索引，其余模组成列索引，直接对应 TT 核链的分割 [1，第 8 章]。在本文共同特征输入的模型中，各核的特征轴具有相同长度；采用前述扩维特征后，该长度为 \(p=2(n+1)\)，其中 \(n\) 为 challenge 位数。

具体而言，第 \(k\) 条连接对应的展开矩阵尺寸为 \(p^k\times p^{x-k}\)，其对称位置第 \(x-k\) 条连接对应的尺寸为 \(p^{x-k}\times p^k\)。二者的行列尺寸互换，而矩阵秩均受较小一侧维数限制。因此，由矩阵尺寸确定的最大可能秩满足 [1]

$$
U_k=p^{\min(k,x-k)}=U_{x-k}.
\tag{P2}
$$

例如，七阶张量的六条内部连接按 \((1,6)\)、\((2,5)\)、\((3,4)\) 配对，各对位置的最大可能秩分别为 \(p\)、\(p^2\)、\(p^3\)。这说明左右对应位置在尺寸层面具有相同的容量上限。在没有额外证据表明某一侧需要更多容量时，可以考虑采用 \(R_k=R_{x-k}\) 的对称预算，以相同的预算约束对应位置的表达能力。

这里的对称性指最大秩容量的对称，为采用相同预算提供结构动机；实际展开秩仍由权重张量的具体取值决定。

## 2. 边缘较小、中央较大的容量分配

式 (P2) 同时说明，边缘切分受较小一侧的维数限制，而中央切分的两侧均包含更多组合状态，因而允许更高的矩阵秩。展开秩具有明确的表达含义：若其秩为 \(r_k\)，权重张量可写为最少 \(r_k\) 个跨该分割的可分离项之和，

$$
\mathcal W(i_1,\ldots,i_x)=\sum_{\alpha=1}^{r_k}
A_\alpha(i_1,\ldots,i_k)B_\alpha(i_{k+1},\ldots,i_x).
\tag{P3}
$$

式 (P3) 可由展开矩阵的紧奇异值分解直接得到；任何少于 \(r_k\) 项的此类表示都会与其矩阵秩矛盾。因此，秩预算限制的是跨该切分的独立线性耦合项数。边缘采用较小预算、中央预留较大预算，是与连续分割容量相适应的一种配置思路。这种分配对应的是容量预算的轮廓，而非要求实际秩随位置单调变化。

## 3. 增大秩不保证泛化收益

在相同特征映射下，小尺寸 TT 核可以通过补零嵌入更大的核，因此逐分量增大连接预算会扩大可表示函数族，而不会提高相同未正则化经验损失的理论下确界。这一表示能力结论并不保证实际优化得到更好的解，也不保证测试风险降低。

TT 核参数量为 \(N_G=\sum_{k=1}^{x}p_kR_{k-1}R_k\)。在等模长、统一内部预算 \(R\) 和边界 \(R_0=R_x=1\) 的情形，\(x\ge2\) 时有 \(N_G=p[2R+(x-2)R^2]\)；对于包含相邻可增长连接的高阶链，参数规模可随 \(R\) 平方增长。Khavari 与 Rabusseau [2] 对张量网络符号内积分类族证明了

$$
\operatorname{VCdim}(\mathcal H_G^{\mathrm{classif}})
\le 2N_G\log_2(12|V|),
\qquad
\mathcal H_G^{\mathrm{classif}}
=\{X\mapsto\operatorname{sign}\langle W,X\rangle:W\in\mathcal T(G)\},
\tag{P4}
$$

其中 \(G\) 为固定张量网络结构，\(\mathcal T(G)\) 为该结构可表示的权重张量集合，\(|V|\) 为核节点数，对于 TT 链有 \(|V|=x\)。对于独立同分布的 \(m\) 个样本及取值于 \([0,1]\) 的分类损失，该文定理 4 和推论 6 进一步给出统一泛化界。在固定置信水平、固定阶数及适用的样本量区间内，对统一内部预算 \(R\) 的 TT 分类器，其泛化间隙满足

$$
\mathcal R(h)-\widehat{\mathcal R}_S(h)
\le O\!\left(\sqrt{\frac{xpR^2\log m}{m}}\right)
\tag{P5}
$$

并以高概率对该模型族同时成立。其中 \(\mathcal R\) 和 \(\widehat{\mathcal R}_S\) 分别表示总体风险和经验风险。对固定特征映射，这一结果为有限样本条件下限制不必要的秩预算提供了统计学习依据：额外容量可能增加对样本偶然模式的拟合空间和估计负担，因而更强的表示能力不必转化为更好的泛化表现。复杂度上界的增长并不意味着实际测试误差必然增大，但说明一味增加秩预算并不带来泛化收益保证。

## 参考文献

[1] Grey Ballard, Tamara G. Kolda. *Tensor Decompositions for Data Science*. Cambridge University Press, final draft, March 25, 2025. 第 8 章，尤其注记 8.1、8.4。 [本地来源](/Users/across/Desktop/pre-summer-paper/papers/PUF/tensor_textbook.pdf)。

[2] Behnoush Khavari, Guillaume Rabusseau. *Lower and Upper Bounds on the Pseudo-Dimension of Tensor Network Models*. Advances in Neural Information Processing Systems 34, 2021. 定理 2、定理 4、推论 6。 [官方主文](https://proceedings.neurips.cc/paper/2021/file/5a9d8bf5b7a4b35f3110dde8673bdda2-Paper.pdf)；[官方补充证明](https://proceedings.neurips.cc/paper/2021/file/5a9d8bf5b7a4b35f3110dde8673bdda2-Supplemental.pdf)。
