from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Iterable
import csv


def read_csv_rows(p: Path) -> List[Dict[str, str]]:
    if not p.exists():
        return []
    with p.open() as f:
        rdr = csv.DictReader(f)
        return [row for row in rdr]


def write_rows(path: Path, header: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))
    return path.name
