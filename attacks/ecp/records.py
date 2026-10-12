"""Read completed experiment records and verify their artifact contents."""

import json
from pathlib import Path

from .common import file_sha256


REQUIRED_ARTIFACTS = {
    "config.json", "split.npz", "initial.npz", "initial_check.npz",
    "history.csv", "model.npz", "lr_model.npz", "metrics.json", "predictions.npz",
}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} 必须为 JSON 对象")
    return value


def load_record(directory: str | Path) -> tuple[Path, dict, dict, dict]:
    directory = Path(directory).resolve()
    manifest = read_json(directory / "manifest.json")
    if manifest.get("record_version") != 1 or manifest.get("status") != "completed":
        raise ValueError("只支持已完成的 version 1 实验记录")
    hashes = manifest.get("artifact_sha256", {})
    if not REQUIRED_ARTIFACTS.issubset(hashes):
        raise ValueError("实验记录缺少必需的产物摘要")
    for name, expected in hashes.items():
        if Path(name).name != name or file_sha256(directory / name) != expected:
            raise ValueError(f"实验产物已改变或损坏：{name}")
    config = read_json(directory / "config.json")
    if config.get("record_version") != 1 or config.get("model") != "single_apuf_ecp":
        raise ValueError("模型配置与单 APUF ECP 协议不符")
    return directory, manifest, config, read_json(directory / "metrics.json")
