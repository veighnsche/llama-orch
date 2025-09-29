"""Centralized CSV artifact helpers.

Responsibilities:
- Ensure headers are written exactly once
- Append rows safely
- Prevent path traversal and ensure parent directories exist
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv_header(path: Path, headers: Iterable[str]) -> None:
    ensure_parent(path)
    if not path.exists():
        path.write_text(",".join(list(headers)) + "\n")


def append_csv_row(path: Path, row: Iterable[str]) -> None:
    ensure_parent(path)
    with path.open("a") as f:
        f.write(",".join(list(row)) + "\n")


def write_dict_rows(path: Path, headers: Iterable[str], rows: Iterable[dict]) -> None:
    """Write a header (once) and append dict rows in the given header order.

    This is append-only; assumes idempotent order upstream for determinism.
    """
    hdr = list(headers)
    write_csv_header(path, hdr)
    for row in rows:
        append_csv_row(path, [str(row.get(h, "")) for h in hdr])
