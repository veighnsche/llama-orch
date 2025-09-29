"""Percentiles computation (minimal)."""
from __future__ import annotations
from typing import Any, Dict, List
from pathlib import Path
import csv


def _sample_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("grid_index", "")),
        str(row.get("replicate_index", "")),
        str(row.get("mc_index", "")),
    )


def _read_sample_totals(p: Path, value_col: str, group_col: str | None = None) -> List[float]:
    """Sum values per (grid,replicate,mc) sample; optionally filter/group by group_col.

    Returns a list of totals, one per sample, to compute percentiles across samples (not pooled rows).
    """
    totals: dict[tuple[str, str, str], float] = {}
    try:
        with p.open() as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                # If a group_col is provided, only include rows with that col present (e.g., scenario=="base")
                if group_col:
                    if (row.get(group_col) or "").strip().lower() != "base":
                        continue
                key = _sample_key(row)
                try:
                    v = float(row.get(value_col, 0) or 0)
                except Exception:
                    v = 0.0
                totals[key] = totals.get(key, 0.0) + v
    except FileNotFoundError:
        pass
    return list(totals.values())


def _pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if q <= 0:
        return xs[0]
    if q >= 100:
        return xs[-1]
    k = (q / 100.0) * (len(xs) - 1)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def compute_percentiles(tables: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Compute percentiles for monthly public/private revenue from outputs.

    Expects `context` to include `out_dir` and optional `simulation` with
    `stochastic.percentiles` list.
    """
    out_dir = context.get("out_dir")
    if not out_dir:
        return {}
    out = Path(out_dir)
    perc_list: List[int] = [10, 50, 90]
    sim = context.get("simulation") or {}
    try:
        perc_list = list(sim.get("stochastic", {}).get("percentiles", perc_list))
    except Exception:
        pass

    # Compute totals per sample (grid,replicate,mc) and then percentiles across samples
    pub_totals_p = out / "public_sample_totals.csv"
    prv_totals_p = out / "private_sample_totals.csv"
    pub_rev: List[float] = []
    prv_rev: List[float] = []
    if pub_totals_p.exists():
        # Read compact totals
        import csv
        with pub_totals_p.open() as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                try:
                    pub_rev.append(float(row.get("total_revenue_eur", 0) or 0))
                except Exception:
                    continue
    else:
        pub_rev = _read_sample_totals(out / "public_tap_scenarios.csv", "revenue_eur", group_col="scenario")
    if prv_totals_p.exists():
        import csv
        with prv_totals_p.open() as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                try:
                    prv_rev.append(float(row.get("total_revenue_eur", 0) or 0))
                except Exception:
                    continue
    else:
        prv_rev = _read_sample_totals(out / "private_tap_customers_by_month.csv", "revenue_eur")

    result: Dict[str, Any] = {"public_revenue": {}, "private_revenue": {}}
    for p in perc_list:
        result["public_revenue"][str(p)] = _pct(pub_rev, float(p))
        result["private_revenue"][str(p)] = _pct(prv_rev, float(p))

    return result
