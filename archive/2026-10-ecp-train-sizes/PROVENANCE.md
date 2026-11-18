# 来源与可复现性说明

本目录是**历史产物的归档**，不是可流水线重跑的实验目录。归档目的只有一个：让结论有据可查，
同时诚实地记录它的缺口。

## 原始来源

| 项目 | 记录值 |
| --- | --- |
| 生成机器 | Windows（`primary/manifest.json` 中 `source` 为 `C:\Users\across\Desktop\PUF\data\crps.npy`） |
| 源数据 | `data/crps.npy`，32 位 Challenge，去重后 99 999 条 CRP |
| 源数据 sha256 | `c055c28c…852cc74d0b`（仅记录，文件已不在本仓库） |
| 划分 | `test_size=0.25`，`split_seed=42`，`n_test=25001` |
| 模型 | 单 APUF ECP，`order=1`，`rank=4`，参数 137 个，Adam，`epochs=500`，`learning_rate=0.01`，全批量 |
| 协议 | 同一次打乱后的训练集取嵌套前缀；种子只改变初始化；测试集固定 |

## 缺口（必须知道）

1. **生成脚本从未入库。** 仓库中曾不存在任何能产出 `primary/` 的代码（`grep ecp_train_sizes` 零命中）。
   归档里的 `manifest.json` 也没有 `git_commit`、`environment`、`artifact_sha256` 字段。
2. **源数据已丢失。** `data/crps.npy`（32 位、99 999 条）已经不在仓库里；现在 `data/` 下只有
   64 位 Challenge 的 4 组数据（每组 100 或 500 条）。因此无法从 CRP 源头重放这套划分——
   唯一能重放的入口是归档下来的 `datasets/*.npy` 本身。
3. **`repeat/` 是中断的运行。** `manifest.json` 停在 `status: "running"`，`metrics.csv` 只有 17 行
   （少 1 次训练），没有 `summary.*`。保留它是为了不隐藏失败记录，不要当作已完成结果引用。
4. **重复数据已删除。** `repeat/datasets/` 曾与 `primary/datasets/` 逐字节相同（sha256 全等），
   且两份 `.csv` 只是 `.npy` 的可读镜像。归档时删除了 `repeat/datasets/` 与全部 CSV，
   仓库体积从 35 MB 降到 6 MB（重建命令见 `README.md`）。

## 与独立重实现的一致性

2026-11 用 `experiments/ecp_train_sizes.py` 直接读取 `datasets/` 重跑 18 次训练（`repro/`），
与 `primary/metrics.csv` 对比：

| 指标 | 结果 |
| --- | --- |
| 最大 \|Δ测试准确率\| | **0.00088**（≈22 / 25001 条） |
| 平均 \|Δ测试准确率\| | 0.00023 |
| 趋势与结论 | 完全一致：随 n_train 单调上升并在 74 998 处饱和于 99.85% |
| 逐位一致性 | **不成立** |

不完全一致的原因是可解释的：原始脚本未入库，初始化方式、精度（原始 `train_accuracy`
保留 float32 舍入特征，如 `0.996999979019165`）与框架版本都无法还原。
差异量级（≤0.09 个百分点）小于同一规模 3 个种子之间的波动范围，因此**不影响任何结论**。

## 以哪个为准

- **结论引用**：用 `primary/summary.json`（原始记录，唯一一次完整运行）。
- **复现与再分析**：用 `repro/`，并配合 `experiments/ecp_train_sizes.py`（有配置、有脚本、可重跑）。
- **新实验**：不要往本目录追加。新实验一律写到 `runs/<experiment-id>/`（见 `runs/README.md`）。