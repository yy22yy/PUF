# Challenge/CRP 数据流程：可执行说明

## 模块职责与文件

该板块负责从 Challenge 到 CRP 的数据准备，共 7 个主要文件：

- `challenge.py`：生成内存中的二进制 Challenge 矩阵。
- `challenge_data.py`：Challenge 的校验、保存/读取及 CLI。
- `generate_crp.py`：加载 Challenge，仿真 APUF 或 XOR-APUF Response，并保存仿真元数据和权重。
- `merge_crp.py`：把每条 Challenge 和对应 Response 合并为一行 CRP。
- `project_config.py`、`puf.toml`：默认文件位置及路径解析。
- `run.py`：固定参数的轻量 Challenge 示例。

CRP 每行格式为 `[Challenge 各位..., Response]`。本流程默认使用软件 APUF 仿真；输出不是物理芯片测量结果。NPY 是机器读取的主格式，脚本同时产生同名 CSV 供检查。

## 环境

在仓库根目录准备主项目环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## 从零生成一份 CRP

以下命令生成 10,000 条、每条 32 位的 Challenge，仿真 APUF Response，再合并 CRP：

```powershell
python -m challenge_data --num 10000 --c_bit 32 --seed 42 --path data/challenges.npy --overwrite
python generate_crp.py --challenge-path data/challenges.npy --response-path data/responses.npy --seed 42 --challenge-seed 42
python merge_crp.py --challenge-path data/challenges.npy --response-path data/responses.npy --output-path data/crps.npy
```

预期生成：

- `data/challenges.npy` 和 `data/challenges.csv`
- `data/responses.npy` 和 `data/responses.csv`
- `data/puf_weights.npy`、配套 CSV，以及 `data/responses_metadata.json`
- `data/crps.npy`、`data/crps.csv` 和 `data/crps_metadata.json`

这些命令会写入 `data/`。若只需先查看参数，可执行 `python -m challenge_data --help`、`python generate_crp.py --help` 或 `python merge_crp.py --help`。Challenge 输出文件已存在时，增加 `--overwrite` 才允许覆盖。

## APUF 与 XOR-PUF 的数据标识

响应生成入口默认使用单 APUF；增加 `--puf-type xor-apuf --xor-count 3` 后，会创建 3 个不同参数的子 APUF，并将它们的最终异或响应保存为 Response。XOR 权重文件的形状为 `(3,n+1)`。

两类数据使用独立实验目录，完整命令和命名规则见 [直接运行](../直接运行.md)。CRP 本身均为 `(样本数,n+1)`，通过配套 `crps_metadata.json` 中的 `puf.type`、`xor_count`、`challenge_bits` 和种子识别模型。响应元数据缺失时，合并记录的类型为 `UNKNOWN`。

## 读取现有 CRP

其他攻击入口可通过 `--crp-path` 指定 CRP：

```powershell
python -m attacks.apuf_lr --crp-path data/crps.npy
```

如果省略显式路径，入口按 `puf.toml` 中的 `default_crp` 寻找数据，当前指向 `data/apuf/n64_w42_c42_m100_run01/crps.npy`。可用 `--config 路径` 或环境变量 `PUF_CONFIG` 选择其他 TOML 配置。配置中的相对路径以 TOML 文件所在目录为准。

## 数据对应关系

`challenges[i]` 对应 `responses[i]`，合并后的 `crps[i, :-1]` 是 Challenge，`crps[i, -1]` 是 Response。合并脚本会检查两者的样本数和 0/1 标签；不要独立打乱其中一个数组后再合并。
