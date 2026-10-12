"""Central project path configuration, rooted at the selected TOML file."""

from __future__ import annotations

import os
from pathlib import Path
import tomllib


def load_project_config(config_path: str | Path | None = None) -> tuple[Path, dict]:
    """Return (project_root, config); PUF_CONFIG overrides cwd discovery."""
    selected = config_path or os.environ.get("PUF_CONFIG")
    if selected:
        path = Path(selected).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"配置文件不存在：{path}")
    else:
        path = next((parent / "puf.toml" for parent in (Path.cwd(), *Path.cwd().parents)
                     if (parent / "puf.toml").is_file()), None)
        if path is None:
            raise FileNotFoundError("找不到 puf.toml；请从项目目录运行或设置 PUF_CONFIG")
        path = path.resolve()
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    paths = config.get("paths", {})
    resolved = {}
    for key in ("data_dir", "runs_dir", "default_crp", "default_challenges", "default_responses"):
        value = paths.get(key)
        if value is not None:
            item = Path(value).expanduser()
            resolved[key] = item.resolve() if item.is_absolute() else (root / item).resolve()
    return root, resolved


def resolve_project_path(value: str | Path, project_root: Path) -> Path:
    """Resolve a user path relative to the selected config, not the shell cwd."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()
