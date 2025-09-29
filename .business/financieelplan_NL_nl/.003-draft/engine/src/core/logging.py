"""Minimal JSONL logging helpers (scaffold)."""
import json
import time


def ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jsonl(event: str, **kw) -> str:
    rec = {"ts": ts(), "level": "INFO", "event": event}
    rec.update(kw)
    return json.dumps(rec)
