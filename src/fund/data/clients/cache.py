"""Shared TTL disk cache for the free data stack.

Same idea stolen from Dexter: cache API responses on disk as JSON keyed by
endpoint+params hash, with per-source TTLs. No secrets ever touch the cache.
Filings are immutable once published, so the SEC client uses long TTLs.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


class TTLCache:
    def __init__(self, cache_dir: str | Path):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, params: dict) -> Path:
        digest = hashlib.sha256(
            json.dumps({"n": namespace, "p": params}, sort_keys=True,
                       default=str).encode()
        ).hexdigest()[:16]
        safe = "".join(c if c.isalnum() or c in "-_" else "_"
                       for c in namespace) or "root"
        return self._dir / f"{safe}_{digest}.json"

    def get(self, namespace: str, params: dict, ttl: int):
        """Return cached data or None. Corrupt/expired entries are ignored."""
        cp = self._path(namespace, params)
        if not cp.exists():
            return None
        try:
            blob = json.loads(cp.read_text())
            if time.time() - blob["ts"] < ttl:
                return blob["data"]
        except (json.JSONDecodeError, KeyError, OSError, ValueError):
            pass
        return None

    def put(self, namespace: str, params: dict, data) -> None:
        cp = self._path(namespace, params)
        try:
            cp.write_text(json.dumps({"ts": time.time(), "data": data}))
        except OSError:
            pass  # cache is best-effort; the fetch already succeeded
