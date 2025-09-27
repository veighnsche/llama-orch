from __future__ import annotations

import datetime as dt


def now_utc_iso() -> str:
    # Use timezone-aware UTC and format with trailing 'Z'
    return (
        dt.datetime.now(dt.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
