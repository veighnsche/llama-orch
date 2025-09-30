"""Sensitivity analysis (minimal).

Computes simple dispersion metrics over monthly revenue for public/private
based on produced outputs. Serves as a placeholder until full grid/replicate
MC analysis is wired.
"""
from __future__ import annotations
from typing import Any, Dict, List
from pathlib import Path
import csv
import math


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


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if not xs:
        return 0.0
    mu = _mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs)
    return math.sqrt(var)


def compute_sensitivity(tables: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = context.get("out_dir")
    if not out_dir:
        return {}
    out = Path(out_dir)
    pub_rev = _read_col(out / "public_tap_scenarios.csv", "revenue_eur")
    prv_rev = _read_col(out / "private_tap_customers_by_month.csv", "revenue_eur")

    return {
        "public_revenue": {
            "min": min(pub_rev) if pub_rev else 0.0,
            "max": max(pub_rev) if pub_rev else 0.0,
            "mean": _mean(pub_rev),
            "std": _std(pub_rev),
        },
        "private_revenue": {
            "min": min(prv_rev) if prv_rev else 0.0,
            "max": max(prv_rev) if prv_rev else 0.0,
            "mean": _mean(prv_rev),
            "std": _std(prv_rev),
        },
    }
