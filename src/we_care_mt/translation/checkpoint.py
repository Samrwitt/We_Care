"""JSONL checkpoints so long documents can resume after failure."""

from __future__ import annotations

import json
from pathlib import Path

from we_care_mt.schema import Segment


class CheckpointStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._done: dict[str, dict] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._done[rec["segment_id"]] = rec

    def has(self, segment_id: str) -> bool:
        return segment_id in self._done

    def apply(self, segments: list[Segment]) -> int:
        n = 0
        for seg in segments:
            rec = self._done.get(seg.segment_id)
            if not rec:
                continue
            seg.translated_text = rec.get("translated_text") or ""
            seg.status = rec.get("status") or "translated"
            seg.model_id = rec.get("model_id") or seg.model_id
            seg.error = rec.get("error") or ""
            n += 1
        return n

    def save_many(self, segments: list[Segment]) -> None:
        if not segments:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            for seg in segments:
                rec = {
                    "segment_id": seg.segment_id,
                    "translated_text": seg.translated_text,
                    "status": seg.status,
                    "model_id": seg.model_id,
                    "error": seg.error,
                    "page_number": seg.page_number,
                    "source_text": seg.source_text,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self._done[seg.segment_id] = rec
