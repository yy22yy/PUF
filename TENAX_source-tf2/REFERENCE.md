# TENAX 参考代码目录

此目录作为本项目后续自主实现 ECP-TRN 的参考资料保留。现阶段只解析代码，不将这里的脚本并入根目录实验流程，也不把它视为已验证的 ECP-TRN 实现。

## 与 ECP-TRN 相关的代码

- `TRL.py` 定义 `ECPRegressionLayer`、`ETTRegressionLayer` 和 `ETTRegressionLayer_LSPUF`。ECP 层使用 CP 秩分量对多个 APUF 阶段特征的线性结果求积，再对秩分量求和；ETT 层使用张量列车核逐步收缩。
- `Datasets/data.py` 负责合成 PUF 数据、读取和划分数据，并提供 PCA 处理函数。
- `PUF/` 提供 APUF、XOR-APUF、IPUF 和 LSPUF 仿真模型。
- `main.py`、`AttackWithPca_main.py`、`AttackIPUF_main.py`、`AttackLSPUF_main.py` 等是 ETT/LR 实验入口；当前没有发现实例化 `ECPRegressionLayer` 的训练入口。

参考前需留意：`ECPRegressionLayer.call()` 的分块循环会覆盖前一个分块的输出，且依赖 `batch_size.numpy()`；这些行为需要在自主实现时重新核对。此目录的依赖及运行环境与根目录 `pyproject.toml` 不同，不能直接视作当前 NumPy 流程的一部分。
