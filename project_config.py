"""兼容 shim：路径配置已迁到 :mod:`pufkit.config`。

保留本文件只为让旧的 ``from project_config import ...`` 继续可用；
新代码请写 ``from pufkit.config import load_project_config, resolve_project_path``。
"""

from pufkit.config import load_project_config, resolve_project_path

__all__ = ["load_project_config", "resolve_project_path"]