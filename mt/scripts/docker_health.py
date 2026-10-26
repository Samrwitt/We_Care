"""Docker HEALTHCHECK: HTTP server up (200 ready or 503 still loading)."""

from __future__ import annotations

import sys
import urllib.error
import urllib.request


def main() -> int:
    try:
        urllib.request.urlopen("http://127.0.0.1:8090/health", timeout=3)
    except urllib.error.HTTPError as exc:
        return 0 if exc.code in {200, 503} else 1
    except OSError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
