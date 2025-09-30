"""Writers and materialization helpers for runner orchestration."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Set

from core.orderings import sorted_rows
from core.artifacts import write_csv_header, append_csv_row, write_dict_rows


class CSVWriter:
    """Append-only CSV writer with header caching per table name."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self._header_by_name: Dict[str, List[str]] = {}
        # Track content-based keys we've already written per table to avoid duplicates
        self._seen_keys_by_name: Dict[str, Set[Tuple[str, ...]]] = {}

    def write_table(self, name: str, header: Iterable[str], rows: Iterable[dict]) -> None:
        hdr = list(header)
        if name not in self._header_by_name:
            self._header_by_name[name] = hdr
        # Always write using canonical header order (first seen for this table)
        hdr_used = self._header_by_name[name]
        p = self.out_dir / f"{name}.csv"
        rows_sorted = sorted_rows(name, list(rows))
        # Prepare dedup cache
        seen = self._seen_keys_by_name.setdefault(name, set())
        # Tag columns (if present) are ignored for dedup so identical content isn't multiplied by job tags
        tag_cols = {"grid_index", "replicate_index", "mc_index"}
        content_hdr = [h for h in hdr_used if h not in tag_cols]
        # Ensure header exists once
        write_csv_header(p, hdr_used)
        for r in rows_sorted:
            key = tuple(str(r.get(h, "")) for h in content_hdr)
            if key in seen:
                continue
            seen.add(key)
            append_csv_row(p, [str(r.get(h, "")) for h in hdr_used])

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
    heavy_tables = {
        "public_tap_scenarios",
        "public_tap_customers_by_month",
        "public_tap_capacity_by_month",
        "private_tap_customers_by_month",
        "private_tap_costs_by_month",
    }
    for gi, ri, mi, tables in sorted(results, key=lambda t: (t[0], t[1], t[2])):
        if not tables:
            continue
        # Compute and append lightweight per-sample totals to small files
        try:
            if "public_tap_scenarios" in tables:
                _, scen_rows = tables["public_tap_scenarios"]
                # Sum revenue for base scenario only, mirroring analysis behavior
                total_pub = 0.0
                # Also compute monthly totals for public (base scenario)
                monthly_pub_rev: Dict[int, float] = {}
                monthly_pub_cost: Dict[int, float] = {}
                for r in scen_rows:
                    if (str(r.get("scenario", "")).strip().lower() == "base"):
                        try:
                            m = int(float(r.get("month", 0) or 0))
                            rev = float(r.get("revenue_eur", 0) or 0)
                            cost = float(r.get("cost_eur", 0) or 0)
                            total_pub += rev
                            monthly_pub_rev[m] = monthly_pub_rev.get(m, 0.0) + rev
                            monthly_pub_cost[m] = monthly_pub_cost.get(m, 0.0) + cost
                        except Exception:
                            pass
                write_csv_header(Path(out_dir) / "public_sample_totals.csv", ["grid_index","replicate_index","mc_index","total_revenue_eur"])
                append_csv_row(Path(out_dir) / "public_sample_totals.csv", [str(gi), str(ri), str(mi), f"{total_pub}"])
                # Write per-month totals compactly
                write_csv_header(Path(out_dir) / "public_monthly_totals.csv", [
                    "grid_index","replicate_index","mc_index","month","revenue_eur","cost_eur"
                ])
                for m in sorted(set(list(monthly_pub_rev.keys()) + list(monthly_pub_cost.keys()))):
                    append_csv_row(
                        Path(out_dir) / "public_monthly_totals.csv",
                        [str(gi), str(ri), str(mi), str(m), f"{monthly_pub_rev.get(m, 0.0)}", f"{monthly_pub_cost.get(m, 0.0)}"],
                    )
        except Exception:
            pass
        try:
            if "private_tap_customers_by_month" in tables:
                _, cust_rows = tables["private_tap_customers_by_month"]
                total_prv = 0.0
                monthly_prv_rev: Dict[int, float] = {}
                for r in cust_rows:
                    try:
                        m = int(float(r.get("month", 0) or 0))
                        rev = float(r.get("revenue_eur", 0) or 0)
                        total_prv += rev
                        monthly_prv_rev[m] = monthly_prv_rev.get(m, 0.0) + rev
                    except Exception:
                        pass
                write_csv_header(Path(out_dir) / "private_sample_totals.csv", ["grid_index","replicate_index","mc_index","total_revenue_eur"])
                append_csv_row(Path(out_dir) / "private_sample_totals.csv", [str(gi), str(ri), str(mi), f"{total_prv}"])
                # Pair with costs per month when present
                monthly_prv_cost: Dict[int, float] = {}
                if "private_tap_costs_by_month" in tables:
                    _, cost_rows = tables["private_tap_costs_by_month"]
                    for r in cost_rows:
                        try:
                            m = int(float(r.get("month", 0) or 0))
                            c = float(r.get("cost_eur", 0) or 0)
                            monthly_prv_cost[m] = monthly_prv_cost.get(m, 0.0) + c
                        except Exception:
                            pass
                write_csv_header(Path(out_dir) / "private_monthly_totals.csv", [
                    "grid_index","replicate_index","mc_index","month","revenue_eur","cost_eur"
                ])
                for m in sorted(set(list(monthly_prv_rev.keys()) + list(monthly_prv_cost.keys()))):
                    append_csv_row(
                        Path(out_dir) / "private_monthly_totals.csv",
                        [str(gi), str(ri), str(mi), str(m), f"{monthly_prv_rev.get(m, 0.0)}", f"{monthly_prv_cost.get(m, 0.0)}"],
                    )
        except Exception:
            pass

        for name, (hdr, rows) in tables.items():
            # To keep output size manageable, write heavy per-sample monthly tables only for the first sample
            if name in heavy_tables and not (gi == 0 and ri == 0 and mi == 0):
                continue
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
