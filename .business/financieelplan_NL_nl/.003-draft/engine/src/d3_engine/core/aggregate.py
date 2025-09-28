"""Aggregation & KPIs (scaffold).

Responsibilities (spec refs: 20_simulations.md §6, 31_engine.md §9):
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import csv
import json


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except Exception:
        return 0.0


def _read_csv_rows(p: Path) -> List[Dict[str, str]]:
    if not p.exists():
        return []
    with p.open() as f:
        rdr = csv.DictReader(f)
        return [row for row in rdr]


def write_consolidated_outputs(out_dir: Path, context: Dict[str, Any]) -> List[str]:
    """Compute minimal consolidated KPIs and write summary files.

    Returns list of written filenames.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    # Read inputs
    pub_scen = _read_csv_rows(out_dir / "public_tap_scenarios.csv")
    prv_cust = _read_csv_rows(out_dir / "private_tap_customers_by_month.csv")

    # KPIs
    pub_rev = sum(_safe_float(r.get("revenue_eur", 0)) for r in pub_scen)
    pub_cost = sum(_safe_float(r.get("cost_eur", 0)) for r in pub_scen)
    pub_margin = pub_rev - pub_cost

    prv_rev = sum(_safe_float(r.get("revenue_eur", 0)) for r in prv_cust)
    prv_cost = 0.0  # not modeled yet in monthly table
    prv_margin = prv_rev - prv_cost

    total_rev = pub_rev + prv_rev
    total_cost = pub_cost + prv_cost
    total_margin = total_rev - total_cost

    # Overhead allocation (simple): sum monthly fixed costs from general finance; allocate by driver
    overhead_driver: Optional[str] = None
    try:
        overhead_driver = context.get("simulation", {}).get("consolidation", {}).get("overhead_allocation_driver")
    except Exception:
        overhead_driver = None
    overhead_total = 0.0
    try:
        gen_fin = context.get("general_finance", {})
        fixed = gen_fin.get("fixed_costs_monthly_eur", {})
        if isinstance(fixed, dict):
            overhead_total = sum(float(v) for v in fixed.values())
    except Exception:
        overhead_total = 0.0

    # metrics for allocation keys
    pub_tokens = sum(_safe_float(r.get("tokens", 0)) for r in pub_scen)
    prv_hours = sum(_safe_float(r.get("hours", 0)) for r in prv_cust)

    alloc_pub = 0.0
    alloc_prv = 0.0
    if overhead_total > 0 and overhead_driver in ("revenue", "gpu_hours", "tokens"):
        if overhead_driver == "revenue":
            total_basis = total_rev
            pub_basis = pub_rev
            prv_basis = prv_rev
        elif overhead_driver == "gpu_hours":
            total_basis = max(prv_hours, 0.0)
            pub_basis = 0.0
            prv_basis = prv_hours
        else:  # tokens
            total_basis = max(pub_tokens, 0.0)
            pub_basis = pub_tokens
            prv_basis = 0.0
        if total_basis > 0:
            alloc_pub = overhead_total * (pub_basis / total_basis)
            alloc_prv = overhead_total * (prv_basis / total_basis)

    pub_cost_alloc = pub_cost + alloc_pub
    prv_cost_alloc = prv_cost + alloc_prv
    total_cost_alloc = pub_cost_alloc + prv_cost_alloc
    pub_margin_alloc = pub_rev - pub_cost_alloc
    prv_margin_alloc = prv_rev - prv_cost_alloc
    total_margin_alloc = total_rev - total_cost_alloc

    # Write consolidated_kpis.csv
    kpis_path = out_dir / "consolidated_kpis.csv"
    with kpis_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["section", "revenue_eur", "cost_eur", "margin_eur", "overhead_allocated_eur"])
        w.writerow(["public", f"{pub_rev}", f"{pub_cost_alloc}", f"{pub_margin_alloc}", f"{alloc_pub}"])
        w.writerow(["private", f"{prv_rev}", f"{prv_cost_alloc}", f"{prv_margin_alloc}", f"{alloc_prv}"])
        w.writerow(["total", f"{total_rev}", f"{total_cost_alloc}", f"{total_margin_alloc}", f"{overhead_total}"])
    written.append(kpis_path.name)

    # Prepare consolidated_summary
    summary = {
        "public": {"revenue_eur": pub_rev, "cost_eur": pub_cost, "margin_eur": pub_margin},
        "private": {"revenue_eur": prv_rev, "cost_eur": prv_cost, "margin_eur": prv_margin},
        "total": {"revenue_eur": total_rev, "cost_eur": total_cost, "margin_eur": total_margin},
    }
    # Merge in autoscaling summary if provided
    autos = context.get("autoscaling_summary") or {}
    if autos:
        summary["autoscaling"] = autos
    # Merge percentiles if provided
    percs = context.get("percentiles") or {}
    if percs:
        summary["percentiles"] = percs
    # Record overhead allocation if applied
    if overhead_total > 0 and overhead_driver:
        summary["overhead_allocation"] = {
            "driver": overhead_driver,
            "total_overhead_eur_per_month": overhead_total,
            "allocated": {"public": alloc_pub, "private": alloc_prv},
        }

    # Write JSON
    (out_dir / "consolidated_summary.json").write_text(json.dumps(summary, indent=2))
    written.append("consolidated_summary.json")

    # Write Markdown (brief)
    md = [
        "# Consolidated Summary",
        "",
        f"Public: revenue €{pub_rev:.2f}, cost €{pub_cost:.2f}, margin €{pub_margin:.2f}",
        f"Private: revenue €{prv_rev:.2f}, cost €{prv_cost:.2f}, margin €{prv_margin:.2f}",
        f"Total: revenue €{total_rev:.2f}, cost €{total_cost:.2f}, margin €{total_margin:.2f}",
    ]
    if autos:
        md.append(
            f"Autoscaling: p95_util={autos.get('p95_util_pct')}, violations={autos.get('sla_violations')}"
        )
    (out_dir / "consolidated_summary.md").write_text("\n".join(md) + "\n")
    written.append("consolidated_summary.md")

    return written
