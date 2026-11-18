# Structural Rationale and Capacity Trade-offs for TT Rank Budgets: Supplementary Text for TENAX

Version: October 31, 2026. English translation of the [final Chinese manuscript text](tt-rank-paper-zh.md). Source verification and insertion guidance appear in the [full Chinese derivation](tt-rank-derivation-zh.md). The following three sections constitute the manuscript text.

## 1. Structural Rationale for Symmetric Rank Budgets

In the tensor-train (TT) representation introduced above, bond dimensions determine core shapes and constrain representational capacity across the corresponding prefix-suffix partitions. Contracting the first \(k\) cores and the remaining cores separately yields a matrix factorization with inner dimension \(R_k\). Denoting the rank of this unfolding by \(r_k\), we obtain

$$
\mathcal W^{\langle k\rangle}=L_kB_k,\qquad
r_k:=\operatorname{rank}(\mathcal W^{\langle k\rangle})
\le\min\left\{R_k,\prod_{j=1}^{k}p_j,\prod_{j=k+1}^{x}p_j\right\}.
\tag{P1}
$$

Here, \(x\) is the tensor order, \(p_j\) is the size of mode \(j\), and \(R_k\) is the prescribed rank budget, with \(R_0=R_x=1\). The first \(k\) modes index rows, and the remaining modes index columns, matching a partition of the TT chain [1, Chapter 8]. Our model uses a common feature input, so all core feature axes have the same size. After the feature expansion described above, this size is \(p=2(n+1)\), where \(n\) is the challenge bit length.

The unfolding at bond \(k\) therefore has size \(p^k\times p^{x-k}\). At its mirrored position, bond \(x-k\), the unfolding has size \(p^{x-k}\times p^k\). These dimensions are interchanged, and both matrix ranks are bounded by the smaller dimension. Thus, the maximum rank permitted by the unfolding dimensions satisfies [1]

$$
U_k=p^{\min(k,x-k)}=U_{x-k}.
\tag{P2}
$$

For example, the six internal bonds of a seventh-order tensor form the pairs \((1,6)\), \((2,5)\), and \((3,4)\). Their respective maximum ranks are \(p\), \(p^2\), and \(p^3\). Mirrored positions therefore have identical dimension-based capacity limits. In the absence of evidence favoring greater capacity on either side, symmetric budgets \(R_k=R_{x-k}\) provide matching capacity constraints at mirrored positions.

This symmetry concerns maximum rank capacity and motivates equal budgets. The actual unfolding ranks still depend on the entries of the weight tensor.

## 2. Allocating Smaller Budgets Near the Boundaries and Larger Budgets Near the Center

Equation (P2) also shows that boundary partitions are constrained by the dimension of their smaller side. Central partitions contain more joint index configurations on both sides and consequently permit higher matrix ranks. The unfolding rank has a precise representational interpretation. If its rank is \(r_k\), the weight tensor admits a decomposition into a minimum of \(r_k\) terms separable across that partition,

$$
\mathcal W(i_1,\ldots,i_x)=\sum_{\alpha=1}^{r_k}
A_\alpha(i_1,\ldots,i_k)B_\alpha(i_{k+1},\ldots,i_x).
\tag{P3}
$$

Equation (P3) follows directly from the compact singular value decomposition of the unfolding matrix. Any such representation with fewer than \(r_k\) terms would contradict its matrix rank. The rank budget therefore limits the number of independent linear coupling terms across the partition. Allocating smaller budgets near the boundaries and reserving larger budgets near the center is consistent with these position-dependent capacity limits. This allocation describes a budget profile, rather than requiring actual ranks to vary monotonically with position.

## 3. Increasing Rank Does Not Guarantee Better Generalization

For a fixed feature map, smaller TT cores can be embedded into larger cores by zero-padding. Increasing bond budgets componentwise therefore enlarges the representable function class. Consequently, the infimum of the same unregularized empirical loss cannot increase. This representational property guarantees neither better solutions from practical optimization nor lower test risk.

The number of TT core parameters is \(N_G=\sum_{k=1}^{x}p_kR_{k-1}R_k\). For equal mode sizes, a uniform internal budget \(R\), and \(R_0=R_x=1\), this becomes \(N_G=p[2R+(x-2)R^2]\) for \(x\ge2\). In higher-order chains with adjacent growing bond dimensions, the parameter count can therefore grow quadratically with \(R\). Khavari and Rabusseau [2] established the following bound for sign-of-inner-product classifiers parameterized by tensor networks,

$$
\operatorname{VCdim}(\mathcal H_G^{\mathrm{classif}})
\le 2N_G\log_2(12|V|),
\qquad
\mathcal H_G^{\mathrm{classif}}
=\{X\mapsto\operatorname{sign}\langle W,X\rangle:W\in\mathcal T(G)\},
\tag{P4}
$$

Here, \(G\) is a fixed tensor network structure, \(\mathcal T(G)\) is its representable weight-tensor family, and \(|V|\) is the number of core nodes. For a TT chain, \(|V|=x\). Their Theorem 4 and Corollary 6 further provide uniform generalization bounds for \(m\) independent and identically distributed samples and classification losses taking values in \([0,1]\). At a fixed confidence level and tensor order, within the applicable sample-size regime, TT classifiers with uniform internal budget \(R\) satisfy

$$
\mathcal R(h)-\widehat{\mathcal R}_S(h)
\le O\!\left(\sqrt{\frac{xpR^2\log m}{m}}\right)
\tag{P5}
$$

uniformly over the model class with high probability. Here, \(\mathcal R\) and \(\widehat{\mathcal R}_S\) denote population and empirical risks, respectively. For a fixed feature map, this result provides a statistical rationale for limiting unnecessary rank budgets under finite samples. Additional capacity may increase the scope for fitting incidental sample patterns and the associated estimation burden. Greater representational power therefore need not translate into better generalization. A growing complexity bound does not imply that actual test error must increase, but increasing rank budgets alone offers no guarantee of generalization gains.

## References

[1] Grey Ballard and Tamara G. Kolda. *Tensor Decompositions for Data Science*. Cambridge University Press, final draft, March 25, 2025. Chapter 8, especially Remarks 8.1 and 8.4. [Local source](/Users/across/Desktop/pre-summer-paper/papers/PUF/tensor_textbook.pdf).

[2] Behnoush Khavari and Guillaume Rabusseau. *Lower and Upper Bounds on the Pseudo-Dimension of Tensor Network Models*. Advances in Neural Information Processing Systems 34, 2021. Theorems 2 and 4; Corollary 6. [Official paper](https://proceedings.neurips.cc/paper/2021/file/5a9d8bf5b7a4b35f3110dde8673bdda2-Paper.pdf); [official supplementary proofs](https://proceedings.neurips.cc/paper/2021/file/5a9d8bf5b7a4b35f3110dde8673bdda2-Supplemental.pdf).
