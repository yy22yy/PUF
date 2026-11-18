# Challenge/CRP 数据流程：可执行说明

## 模块职责

数据准备由 `pufkit/` 一套代码完成，不再分散在根目录的多个脚本里：

| 模块 | 职责 | 命令行 |
| --- | --- | --- |
| `pufkit/challenge.py` | Challenge 生成、校验、保存/读取 | `python -m pufkit.challenge` |
| `pufkit/crp.py` | 仿真 Response（`generate`）／合并 CRP（`merge`） | `python -m pufkit.crp` |
| `pufkit/config.py` | 读取 `puf.toml`、解析项目路径 | 被上面两者调用 |
| `pufkit/io.py` | `.npy` + 同名 `.csv` 写入约定、内容摘要 | 被上面两者调用 |

模型在 `puf/`（`APUF`、`XORAPUF`、`ChallengeFeatureTransformer`），攻击在 `attacks/`。

CRP 每行格式为 `[Challenge 各位..., Response]`。本流程使用软件 PUF 仿真；输出**不是**物理芯片测量结果。
NPY 是机器读取的主格式，脚本同时产生同名 CSV 供人工检查。

## 环境

```bash
python -m venv .venv                 # 主环境只需 NumPy
.venv/bin/python -m pip install -e . # Windows: .\.venv\Scripts\Activate.ps1 后 python -m pip install -e .
```

## 从零生成一份 CRP

下面按"一个实验一个目录"的约定，生成 10,000 条 32 位 Challenge 的 APUF 数据。
目录名规则（`n`=Challenge 位数，`k`=子 PUF 数，`w`=权重种子，`c`=Challenge 种子，`m`=样本数）见 [直接运行](../直接运行.md)。

```bash
RUN=data/apuf/n32_w42_c42_m10000_run01

# 1) Challenge
python -m pufkit.challenge --num 10000 --c_bit 32 --seed 42 --path "$RUN/challenges.npy"

# 2) Response（同时保存真值权重与 responses_metadata.json）
python -m pufkit.crp generate --challenge-path "$RUN/challenges.npy" \
    --response-path "$RUN/responses.npy" --challenge-seed 42 --seed 42

# 3) 合并 CRP（同时保存 crps_metadata.json）
python -m pufkit.crp merge --challenge-path "$RUN/challenges.npy" \
    --response-path "$RUN/responses.npy" --output-path "$RUN/crps.npy"
```

预期产物（全部在同一目录）：

- `challenges.npy` / `challenges.csv`
- `responses.npy` / `responses.csv`
- `puf_weights.npy` / `puf_weights.csv`（真值权重，**不要**给攻击模型）
- `responses_metadata.json`（模型类型、种子、数组摘要、源文件摘要）
- `crps.npy` / `crps.csv`（形状 `(10000, 33)`）
- `crps_metadata.json`（模型标识、样本数、输入摘要）

查看参数用 `python -m pufkit.challenge --help` 与 `python -m pufkit.crp generate --help`。
Challenge 输出文件已存在时需要 `--overwrite` 才允许覆盖；`pufkit.crp` 不覆盖任何输入。

## APUF 与 XOR-PUF 的数据标识

`generate` 默认单 APUF；加 `--puf-type xor-apuf --xor-count 3` 会创建 3 个不同参数的子 APUF，
保存它们的异或响应，权重形状变为 `(3, n+1)`。

两类数据必须分目录存放。CRP 本身都是 `(样本数, n+1)`，靠配套 `crps_metadata.json` 里的
`puf.type`、`xor_count`、`challenge_bits` 和种子区分；响应元数据缺失时，合并记录标为 `UNKNOWN`。

## 读取现有 CRP

攻击入口既可以显式指定路径，也可以省略而走 `puf.toml`：

```bash
python -m attacks.apuf_lr --crp-path data/apuf/n64_w42_c42_m100_run01/crps.npy

# 省略 --crp-path 时读取 puf.toml 的 default_crp
python -m attacks.apuf_lr
```

`--config 路径` 或环境变量 `PUF_CONFIG` 可切换另一份 TOML；配置里的相对路径以 TOML 所在目录为准。
当前默认值指向 `data/apuf/n64_w42_c42_m100_run01/crps.npy`。

## 数据对应关系

`challenges[i]` 对应 `responses[i]`；合并后 `crps[i, :-1]` 是 Challenge，`crps[i, -1]` 是 Response。
`merge` 会核对样本数、0/1 取值范围，以及（当元数据存在时）两个输入数组的摘要，
因此不要把其中一个数组单独打乱后再合并。