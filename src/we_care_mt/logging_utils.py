"""Console and rotating-file logging."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(
    log_dir: Path | None = None,
    level: str = "INFO",
    json_logs: bool = False,
    name: str = "we_care_mt",
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    stream = logging.StreamHandler(sys.stderr)
    if json_logs:
        stream.setFormatter(JsonFormatter())
    else:
        stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(stream)

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"we_care_mt_{datetime.now().strftime('%Y%m%d')}.log"
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        logger.addHandler(fh)
    return logger
