"""Disk cache for chunk embedding matrices (avoids re-encoding ~15k chunks)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from we_care_nlu.config import package_root


def cache_dir() -> Path:
    root = package_root() / "data" / "cache" / "embeddings"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _key(model_key: str, hf_id: str | None, texts: list[str]) -> str:
    h = hashlib.sha256()
    h.update((model_key or "").encode())
    h.update(b"\0")
    h.update((hf_id or "").encode())
    h.update(b"\0")
    # Hash content without holding a giant joined string in memory.
    for t in texts:
        h.update(t.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()[:40]


def load_cached(model_key: str, hf_id: str | None, texts: list[str]) -> np.ndarray | None:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    stem = f"{model_key}_{_key(model_key, hf_id, texts)}"
    path = cache_dir() / f"{stem}.npy"
    meta = cache_dir() / f"{stem}.json"
    if not path.exists() or not meta.exists():
        return None
    try:
        info = json.loads(meta.read_text(encoding="utf-8"))
        if int(info.get("n") or 0) != len(texts):
            return None
        arr = np.load(path)
        if arr.shape[0] != len(texts):
            return None
        return np.asarray(arr, dtype=np.float32)
    except Exception:
        return None


def save_cached(model_key: str, hf_id: str | None, texts: list[str], matrix: np.ndarray) -> Path:
    stem = f"{model_key}_{_key(model_key, hf_id, texts)}"
    path = cache_dir() / f"{stem}.npy"
    meta = cache_dir() / f"{stem}.json"
    # np.save adds .npy if missing; pass path without relying on that.
    with path.open("wb") as fh:
        np.save(fh, np.asarray(matrix, dtype=np.float32))
    meta.write_text(
        json.dumps(
            {
                "model_key": model_key,
                "hf_id": hf_id,
                "n": len(texts),
                "dim": int(matrix.shape[1]) if matrix.ndim == 2 and matrix.size else 0,
            }
        ),
        encoding="utf-8",
    )
    return path


def embed_texts(embedder: Any, texts: list[str], *, use_cache: bool = True) -> np.ndarray:
    """Encode texts, optionally reusing a content-addressed numpy cache."""
    if not texts:
        dim = int(getattr(embedder, "dim", 256) or 256)
        return np.zeros((0, dim), dtype=np.float32)
    model_key = str(getattr(embedder, "model_key", "unknown"))
    hf_id = getattr(embedder, "hf_id", None)
    if use_cache:
        cached = load_cached(model_key, hf_id, texts)
        if cached is not None:
            print(f"we-care-nlu: embedding cache hit n={len(texts)} model={model_key}", flush=True)
            return cached
    matrix = embedder.encode(texts)
    if use_cache:
        try:
            save_cached(model_key, hf_id, texts, matrix)
            print(f"we-care-nlu: embedding cache saved n={len(texts)} model={model_key}", flush=True)
        except Exception as exc:
            print(f"we-care-nlu: embedding cache save failed: {exc}", flush=True)
    return np.asarray(matrix, dtype=np.float32)
