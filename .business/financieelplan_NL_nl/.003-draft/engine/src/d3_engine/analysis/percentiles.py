"""Percentiles computation (minimal)."""
from __future__ import annotations
from typing import Any, Dict, List
from pathlib import Path
import csv


def _read_col(p: Path, col: str) -> List[float]:
    vals: List[float] = []
    try:
        with p.open() as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                try:
                    vals.append(float(row.get(col, 0) or 0))
                except Exception:
                    vals.append(0.0)
    except FileNotFoundError:
        pass
    return vals


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

    pub_rev = _read_col(out / "public_tap_scenarios.csv", "revenue_eur")
    prv_rev = _read_col(out / "private_tap_customers_by_month.csv", "revenue_eur")

    result: Dict[str, Any] = {"public_revenue": {}, "private_revenue": {}}
    for p in perc_list:
        result["public_revenue"][str(p)] = _pct(pub_rev, float(p))
        result["private_revenue"][str(p)] = _pct(prv_rev, float(p))

    return result
