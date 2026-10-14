# 数据目录

当前保留两组可复现的仿真数据，完整生成命令见 [直接运行](../直接运行.md)。

```text
data/
  apuf/n64_w42_c42_m100_run01/          单 APUF，64 位 Challenge，100 条 CRP
  xor_apuf/k3_n64_w42_c42_m100_run01/   3-XOR APUF，64 位 Challenge，100 条 CRP
```

目录名记录 Challenge 位数、子模型数量、模型种子、Challenge 种子、样本数和运行编号。每组目录保存 Challenge、Response、权重、CRP 以及配套 CSV 和元数据；以元数据中的模型类型和参数为准。

根目录 `puf.toml` 的默认数据路径指向上述单 APUF 数据。使用 XOR-PUF 时，显式指定 `data/xor_apuf/k3_n64_w42_c42_m100_run01/crps.npy`。

这两组数组已验证可由现有代码和记录的种子重建。若删除某组数据，需要先运行生成命令再进行攻击实验。每次新实验使用新的运行编号，避免覆盖已有结果。
