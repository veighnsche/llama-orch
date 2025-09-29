"""Writers and materialization helpers for runner orchestration."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ..core.orderings import sorted_rows
from ..core.artifacts import write_csv_header, append_csv_row, write_dict_rows


class CSVWriter:
    """Append-only CSV writer with header caching per table name."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self._header_by_name: Dict[str, List[str]] = {}

    def write_table(self, name: str, header: Iterable[str], rows: Iterable[dict]) -> None:
        hdr = list(header)
        if name not in self._header_by_name:
            self._header_by_name[name] = hdr
        # Always write using canonical header order (first seen for this table)
        hdr_used = self._header_by_name[name]
        p = self.out_dir / f"{name}.csv"
        rows_sorted = sorted_rows(name, list(rows))
        write_dict_rows(p, hdr_used, rows_sorted)

    def artifacts(self) -> List[str]:
        return [f"{name}.csv" for name in sorted(self._header_by_name.keys())]


def write_scaling_events(out_dir: Path, events: List[dict]) -> str:
    """Append autoscaling events CSV and return filename."""
    path = Path(out_dir) / "public_tap_scaling_events.csv"
    if not path.exists():
        write_csv_header(
            path,
            [
                "timestamp_s",
                "model",
                "gpu",
                "demand_tokens_per_hour",
                "effective_capacity",
                "replicas_prev",
                "replicas_new",
                "reason",
                "util_pct",
            ],
        )
    for e in events:
        append_csv_row(
            path,
            [
                str(e.get("timestamp_s", "")),
                str(e.get("model", "")),
                str(e.get("gpu", "")),
                f"{e.get('demand_tokens_per_hour', '')}",
                f"{e.get('effective_capacity', '')}",
                str(e.get("replicas_prev", "")),
                str(e.get("replicas_new", "")),
                str(e.get("reason", "")),
                f"{e.get('util_pct', '')}",
            ],
        )
    return path.name


def materialize_pipeline_tables(
    out_dir: Path,
    results: List[Tuple[int, int, int, Dict[str, tuple[list[str], list[dict]]]]],
    csv_writer: CSVWriter,
) -> List[str]:
    """Write tables from results for a single pipeline using CSVWriter.

    results: list of (gi, ri, mi, tables)
    Returns list of artifact filenames written (CSV tables).
    """
    for gi, ri, mi, tables in sorted(results, key=lambda t: (t[0], t[1], t[2])):
        if not tables:
            continue
        for name, (hdr, rows) in tables.items():
            # Tag all rows with (grid_index, replicate_index, mc_index)
            tag_cols = ["grid_index", "replicate_index", "mc_index"]
            if not any(c in hdr for c in tag_cols):
                hdr = tag_cols + list(hdr)
            tagged_rows = []
            for r in rows:
                r2 = dict(r)
                r2.setdefault("grid_index", str(gi))
                r2.setdefault("replicate_index", str(ri))
                r2.setdefault("mc_index", str(mi))
                tagged_rows.append(r2)
            csv_writer.write_table(name, hdr, tagged_rows)
    return csv_writer.artifacts()
