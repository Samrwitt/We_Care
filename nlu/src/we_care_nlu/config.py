from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg = Path(path) if path else package_root() / "configs" / "default.yaml"
    with cfg.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    return data
