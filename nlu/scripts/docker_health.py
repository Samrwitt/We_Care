"""Docker HEALTHCHECK for the NLU API.

HTTP 200 with status loading/ok both mean the server is bound.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


def main() -> int:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8088/health", timeout=3) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return 0 if exc.code == 200 else 1
    except OSError:
        return 1
    return 0 if body.get("status") in {"ok", "loading"} else 1


if __name__ == "__main__":
    raise SystemExit(main())