"""Acceptance checks (expanded).

Responsibilities (spec refs: 40_testing.md §5, 20_simulations.md §7.1):
- Monotone groei (public/private) en optionele min MoM groei (public)
- Private marge‑drempel per GPU‑klasse
- Capaciteitsviolaties (instances_needed ≤ autoscaling.max_instances_per_model)
- Autoscaling p95(util) binnen tolerantie en geen SLA‑violations

Return structured results for run_summary.
"""
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import csv


def _read_csv_column(p: Path, col: str) -> List[float]:
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


def _read_csv_bool_col(p: Path, col: str) -> List[bool]:
    vals: List[bool] = []
    try:
        with p.open() as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                s = (row.get(col) or "").strip().lower()
                vals.append(s in ("true", "1", "yes"))
    except FileNotFoundError:
        pass
    return vals


def _is_monotone_non_decreasing(xs: List[float]) -> bool:
    return all(xs[i] <= xs[i+1] for i in range(len(xs)-1)) if len(xs) > 1 else True


def _min_mom_growth(xs: List[float]) -> float:
    mins: List[float] = []
    for i in range(len(xs)-1):
        prev = xs[i]
        cur = xs[i+1]
        denom = prev if prev > 0 else 1.0
        mins.append((cur - prev) / denom)
    return min(mins) if mins else 0.0


def check_acceptance(tables: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
    autos = tables.get("autoscaling_summary") or {}
    outputs_dir = tables.get("outputs_dir")  # may be Path
    out_dir = Path(outputs_dir) if outputs_dir else None

    target_util = float(targets.get("target_utilization_pct", 75.0))
    tol = float(targets.get("autoscaling_util_tolerance_pct", 25.0))
    priv_margin_thr = float(targets.get("private_margin_threshold_pct", 20.0))
    min_public_mom_pct = targets.get("public_growth_min_mom_pct")
    min_public_mom_frac = None if min_public_mom_pct is None else float(min_public_mom_pct) / 100.0

    # Autoscaling checks
    p95 = autos.get("p95_util_pct")
    violations = autos.get("sla_violations")
    p95_ok = False if p95 is None else (target_util - tol <= float(p95) <= target_util + tol)
    no_violations = False if violations is None else (int(violations) == 0)

    # Monotonicity and min MoM (read public/private monthly tables)
    public_monotone_ok = None
    public_min_mom_ok = None
    private_monotone_ok = None
    capacity_ok = None
    private_margin_ok = None

    if out_dir:
        pub_active = _read_csv_column(out_dir / "public_tap_customers_by_month.csv", "active_customers")
        if pub_active:
            public_monotone_ok = _is_monotone_non_decreasing(pub_active)
            if min_public_mom_frac is not None:
                min_growth = _min_mom_growth(pub_active)
                public_min_mom_ok = (min_growth >= min_public_mom_frac)

        prv_active = _read_csv_column(out_dir / "private_tap_customers_by_month.csv", "active_clients")
        if prv_active:
            private_monotone_ok = _is_monotone_non_decreasing(prv_active)

        cap_flags = _read_csv_bool_col(out_dir / "public_tap_capacity_plan.csv", "capacity_violation")
        if cap_flags:
            capacity_ok = (not any(cap_flags))

        # Private margin threshold from economics table (per GPU)
        prv_margin_pct = _read_csv_column(out_dir / "private_tap_economics.csv", "margin_pct")
        if prv_margin_pct:
            private_margin_ok = (min(prv_margin_pct) >= priv_margin_thr)

    # Aggregate acceptance
    accepted = True
    # Autoscaling must be OK when summary present
    if autos:
        accepted = accepted and bool(p95_ok and no_violations)
    # Monotonic constraints where present
    if public_monotone_ok is not None:
        accepted = accepted and public_monotone_ok
    if private_monotone_ok is not None:
        accepted = accepted and private_monotone_ok
    if public_min_mom_ok is not None:
        accepted = accepted and public_min_mom_ok
    if capacity_ok is not None:
        accepted = accepted and capacity_ok
    if private_margin_ok is not None:
        accepted = accepted and private_margin_ok

    return {
        "accepted": bool(accepted),
        "autoscaling": {
            "p95_ok": p95_ok,
            "no_sla_violations": no_violations,
            "target_utilization_pct": target_util,
            "tolerance_pct": tol,
        },
        "public_monotonic_ok": public_monotone_ok,
        "public_min_mom_ok": public_min_mom_ok,
        "private_monotonic_ok": private_monotone_ok,
        "private_margin_threshold_pct": priv_margin_thr,
        "private_margin_ok": private_margin_ok,
        "capacity_ok": capacity_ok,
    }
