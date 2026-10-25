"""YAML configuration loading with project-root-relative path resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _is_mt_root(candidate: Path) -> bool:
    return (candidate / "src" / "we_care_mt").is_dir() and (candidate / "configs" / "default.yaml").is_file()


def _project_root_from(start: Path | None = None) -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if _is_mt_root(candidate):
            return candidate
    cur = (start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        if _is_mt_root(candidate):
            return candidate
    return cur


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


@dataclass
class AppConfig:
    raw: dict[str, Any]
    root: Path
    models: dict[str, Any] = field(default_factory=dict)

    @property
    def paths(self) -> dict[str, Any]:
        return self.raw.get("paths", {})

    def path(self, key: str) -> Path:
        return resolve_path(self.paths[key], self.root)

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name, {}) or {})

    def model_entry(self, model_key: str) -> dict[str, Any]:
        models = self.models.get("models", {})
        if model_key in models:
            return dict(models[model_key])
        # Allow passing a raw Hugging Face id.
        return {
            "hf_id": model_key,
            "display_name": model_key,
            "license": "unknown",
            "clinically_validated": False,
            "max_input_tokens": 256,
            "cpu_ok": True,
            "recommended_vram_gb": 6,
        }

    def default_model_key(self) -> str:
        return str(self.section("inference").get("default_model", "dsfsi_600m"))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_config(
    config_path: str | Path | None = None,
    models_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    root = _project_root_from()
    cfg_file = Path(config_path) if config_path else root / "configs" / "default.yaml"
    if not cfg_file.is_absolute():
        cfg_file = (root / cfg_file).resolve() if not cfg_file.exists() else cfg_file.resolve()
    models_file = Path(models_path) if models_path else root / "configs" / "models.yaml"
    if not models_file.is_absolute():
        models_file = (root / models_file).resolve() if not models_file.exists() else models_file.resolve()

    raw = load_yaml(cfg_file)
    models = load_yaml(models_file) if models_file.exists() else {}
    if overrides:
        raw = _deep_merge(raw, overrides)
    raw["project_root"] = str(root)
    return AppConfig(raw=raw, root=root, models=models)


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
