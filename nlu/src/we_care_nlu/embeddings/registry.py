"""Embedding backends. AfroXLMR uses mean pooling; it is not an STS model."""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

import numpy as np


class Embedder(Protocol):
    model_key: str
    hf_id: str | None
    dim: int
    amharic_sentence_encoder: bool

    def encode(self, texts: list[str]) -> np.ndarray: ...


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return mat / denom


class HashEmbedder:
    """Deterministic character n-gram hash. For tests only. Not semantic."""

    def __init__(self, dim: int = 256, model_key: str = "hash"):
        self.model_key = model_key
        self.hf_id = None
        self.dim = dim
        self.amharic_sentence_encoder = False

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            padded = f" {text} "
            for i in range(len(padded) - 2):
                gram = padded[i : i + 3]
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            rows.append(vec)
        return l2_normalize(np.vstack(rows) if rows else np.zeros((0, self.dim)))


class SentenceTransformerEmbedder:
    def __init__(self, model_key: str, spec: dict[str, Any]):
        import os

        from sentence_transformers import SentenceTransformer

        self.model_key = model_key
        self.hf_id = spec["hf_id"]
        self.dim = int(spec.get("dim") or 768)
        self.amharic_sentence_encoder = bool(spec.get("amharic_sentence_encoder", True))
        self._batch_size = int(spec.get("batch_size") or 32)
        cache = os.environ.get("SENTENCE_TRANSFORMERS_HOME") or os.environ.get("HF_HOME")
        kwargs: dict[str, Any] = {}
        if cache:
            kwargs["cache_folder"] = cache
        self._model = SentenceTransformer(self.hf_id, **kwargs)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        arr = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 200,
        )
        return np.asarray(arr, dtype=np.float32)


class MeanPoolEncoderEmbedder:
    """Feature extraction from an MLM encoder. Not a trained sentence embedding."""

    def __init__(self, model_key: str, spec: dict[str, Any]):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_key = model_key
        self.hf_id = spec["hf_id"]
        self.dim = int(spec.get("dim") or 768)
        self.amharic_sentence_encoder = False
        self.max_length = int(spec.get("max_length") or 256)
        self._tok = AutoTokenizer.from_pretrained(self.hf_id)
        self._model = AutoModel.from_pretrained(self.hf_id)
        self._model.eval()
        self._torch = torch

    def encode(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        enc = self._tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            out = self._model(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            summed = (hidden * mask).sum(1)
            counts = mask.sum(1).clamp(min=1)
            pooled = summed / counts
        vec = pooled.cpu().numpy().astype(np.float32)
        return l2_normalize(vec)


def load_embedder(model_key: str, spec: dict[str, Any] | None = None) -> Embedder:
    spec = spec or {}
    kind = spec.get("kind") or ("hash" if model_key == "hash" else "sentence_transformer")
    if kind == "hash" or model_key == "hash":
        return HashEmbedder(dim=int(spec.get("dim") or 256), model_key=model_key)
    if kind == "encoder_mean_pool":
        return MeanPoolEncoderEmbedder(model_key, spec)
    if kind == "sentence_transformer":
        return SentenceTransformerEmbedder(model_key, spec)
    raise ValueError(f"Unknown embedder kind {kind}")


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    return a @ b.T
