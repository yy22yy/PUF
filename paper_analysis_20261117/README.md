# 论文分析：Neural Network Modeling Attacks on Arbiter-PUF-Based Designs

## 论文信息

- **标题**：Neural Network Modeling Attacks on Arbiter-PUF-Based Designs
- **作者**：Nils Wisiol, Bipana Thapaliya, Khalid T. Mursi, Jean-Pierre Seifert, Yu Zhuang
- **发表**：IEEE Transactions on Information Forensics and Security (TIFS), vol. 17, pp. 2719–2731, 2022
- **DOI**：10.1109/TIFS.2022.3189533（IEEE 正式版需订阅）
- **本目录 PDF 来源**：IACR ePrint 2021/555 作者公开版（内容与正式版一致），经 Internet Archive 存档副本下载（ePrint 官网有 Cloudflare 拦截）

## 本目录文件

| 文件 | 说明 |
|---|---|
| `Neural_Network_Modeling_Attacks_on_Arbiter-PUF-Based_Designs_TIFS2022.pdf` | 论文原文（363 KB） |
| `paper_fulltext.txt` | 用 pdfplumber 提取的全文纯文本（13 页，便于检索/引用） |
| `Neural_Network_Modeling_Attacks_论文翻译与总结.md` | 完整中文翻译（摘要→致谢）+ 核心方法/贡献/实验总结 |
| [Experiment 01–02 实验大纲](<Experiment_01_02_实验大纲.md>) | 用户提供的 Improved LR、ECP-TRN 实验大纲整理版 |
| [现有内容盘点与复现差距](<Experiment_01_02_现有内容盘点与复现差距.md>) | Table I/II 原文参数核对、可复用代码、历史结果边界及实施缺口 |

## 已完成的工作（2026-11 会话）

1. **论文获取**：下载作者公开版 PDF，核验标题/作者与引用一致（IEEE 正式版为闭源订阅）。
2. **全文提取**：pdfplumber 提取 13 页全文到 `paper_fulltext.txt`。
3. **翻译与总结**：完成全文中文翻译；总结核心方法（ATF 特征变换 + 镜像 PUF 结构的三层 tanh-MLP + Keras 统一实现 + 攻击成功率指标）、五点贡献（含证伪 Mursi 极低数据复杂度与 Nguyen “LR 最优”）、实验设计与结论。
4. **配套仓库确认**：官方代码 = pypuf 框架 `2021-mlp` 分支（论文贡献点 5 明示）。
5. **复现可行性分析**（详见下节）。

## 配套实验仓库

- 仓库：<https://github.com/nils-wisiol/pypuf>（一作 Wisiol 即 pypuf 维护者）
- 本论文钉住分支：<https://github.com/nils-wisiol/pypuf/tree/2021-mlp>
- 关键文件对照：
  - `pypuf/attack/mlp2021.py` — `MLPAttack2021`，一个类覆盖 Mursi 变体（`(2^{k-1}, 2^k, 2^{k-1})` + tanh）与 Aseeri 变体（`(2^k, 2^k, 2^k)` + ReLU）
  - `pypuf/attack/lr2021.py` — `LRAttack2021`（改进 LR 基线：mini-batch + Adam + 每链 tanh）
  - `pypuf/simulation/delay.py` — `XORArbiterPUF` / `FeedForwardArbiterPUF` / `XORFeedForwardArbiterPUF` / `InterposePUF` 等仿真器
  - `pypuf/io.py` — CRP 生成（论文用来替换 Mursi 有 bug 的生成器，bug：约 20% 时延被误置零）
  - `pypuf/batch.py` — `StudyBase` 参数扫描框架（SLURM 可用）
  - `docs/attacks/mlp.rst` — 超参数 + 表 IV 同款性能基准 + 可运行 doctest
- 真实数据（论文 VII-E）：主分支 `pypuf.io.MTZAA20.xor_arbiter_puf_{4..9}_xor`（Zenodo 自动下载）、`pypuf.io.AM21.interpose_puf`

## 复现缺口（需自行补齐）

1. **ECP-TRN（Santikellur）攻击不在仓库中** —— attack 目录只有 `lr2021.py`/`mlp2021.py`；复现表 II 需自实现（本仓库根目录 `ecp_trn_minimal.py` 已在做这件事）。
2. **逐实验 study 脚本未公开** —— 表 I–IV/图 2–4 的参数矩阵驱动代码需按 `docs/batch.rst` 模板自写。
3. **分割攻击（Interpose）需拼装** —— 分治逻辑在旧版 v1 分支（2020 年 LR 版），需移植到 `2021-mlp`。
4. **FF-APUF 的 FPGA 部分（VHDL 工程 + Artix-7 实测数据）未公开** —— 只能复现其仿真部分（`FeedForwardArbiterPUF` + `MLPAttack2021(net=[n, n/2, n/2, n], tanh)`）。
5. **环境代差** —— 代码为 TF 2.4/Keras2 时代；严格对齐论文数字建议装旧版依赖。

## 建议复现顺序

1. clone `2021-mlp` + 旧版 TF → 跑 `docs/attacks/mlp.rst` 的 5-XOR doctest（验证相似度 >96%）；
2. `StudyBase` 写小型 study 复现表 IV 的 n=64, k=4..6 三行；
3. Aseeri 变体 + `LRAttack2021` → 复现图 4 对比；
4. 主分支拉 `MTZAA20`/`AM21` 数据复现图 3（真实数据）；
5. 自补 ECP-TRN 与分割攻击拼装。
