"""项目的基础设施：配置解析、数组落盘约定、Challenge 与 CRP 数据流程。

模块划分：

* ``pufkit.config``    读取 ``puf.toml``，解析项目根目录与路径
* ``pufkit.io``        ``.npy`` + 同名 ``.csv`` 写入约定与内容摘要
* ``pufkit.challenge`` Challenge 生成、保存、读取（``python -m pufkit.challenge``）
* ``pufkit.crp``       Response 仿真与 CRP 合并（``python -m pufkit.crp``）

PUF 模型本身在 ``puf/``，攻击实现在 ``attacks/``，可复现实验脚本在 ``experiments/``。
"""

__all__: list[str] = []