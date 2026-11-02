"""SQLite cache of identical source-segment translations."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path


def _key(model_id: str, source: str, gen_sig: str) -> str:
    payload = f"{model_id}\n{gen_sig}\n{source}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class TranslationCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS translations (
                cache_key TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                source TEXT NOT NULL,
                target TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, model_id: str, source: str, gen_sig: str) -> str | None:
        key = _key(model_id, source, gen_sig)
        with self._lock:
            cur = self._conn.execute("SELECT target FROM translations WHERE cache_key = ?", (key,))
            row = cur.fetchone()
        return None if row is None else row[0]

    def put(self, model_id: str, source: str, gen_sig: str, target: str) -> None:
        key = _key(model_id, source, gen_sig)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO translations(cache_key, model_id, source, target) VALUES (?,?,?,?)",
                (key, model_id, source, target),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
