"""Content-addressed cache.

Keyed on the full request content (model, sampling settings, system prompt, user
prompt, schema, sample index), so re-running a study costs nothing and changing
any input misses the cache rather than silently reusing a stale verdict.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from ..providers.base import LLMResponse, RawUsage

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key         TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    payload     TEXT,
    text        TEXT,
    usage       TEXT NOT NULL,
    stop_reason TEXT,
    created_at  REAL NOT NULL DEFAULT (strftime('%s','now'))
);
"""


class ResponseCache:
    """SQLite-backed store of provider responses. Thread-safe, process-local."""

    def __init__(self, directory: Path | str, enabled: bool = True):
        self.enabled = enabled
        self.path = Path(directory) / "responses.sqlite3"
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self.hits = 0
        self.misses = 0
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def get(self, key: str) -> LLMResponse | None:
        if not self.enabled or self._conn is None:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT model, payload, text, usage, stop_reason FROM responses WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        model, payload, text, usage, stop_reason = row
        return LLMResponse(
            payload=json.loads(payload) if payload is not None else None,
            text=text or "",
            usage=RawUsage(**json.loads(usage)),
            model=model,
            stop_reason=stop_reason,
        )

    def put(self, key: str, response: LLMResponse) -> None:
        if not self.enabled or self._conn is None:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, model, payload, text, usage, stop_reason) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    key,
                    response.model,
                    json.dumps(response.payload) if response.payload is not None else None,
                    response.text,
                    json.dumps(vars(response.usage)),
                    response.stop_reason,
                ),
            )
            self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            with self._lock:
                self._conn.close()
                self._conn = None
