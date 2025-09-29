from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import json

from .io_utils import read_csv_rows, write_rows
from .series import (
    sum_by_month,
    percentiles_sum_by_month,
    pad_series,
    fixed_opex_series,
    depreciation_series,
    revenue_and_cogs,
    vat_cash_series,
    working_capital_deltas,
    ebitda_series,
    ebt_tax_net,
)
from .loan import build_loan_schedule
from .pnl_cashflow import write_pnl, write_cashflow
from .kpis import compute_kpis


def write_consolidated_outputs(out_dir: Path, context: Dict[str, Any]) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    sim = context.get("simulation", {}) if isinstance(context, dict) else {}
    horizon = int(sim.get("targets", {}).get("horizon_months", 12)) if isinstance(sim, dict) else 12
    # Consolidation percentiles and representative cut (closest to 50)
    try:
        report_percs = list(sim.get("consolidation", {}).get("report_percentiles", [50]))
        report_percs = [float(p) for p in report_percs if isinstance(p, (int, float))]
        report_percs = report_percs or [50.0]
    except Exception:
        report_percs = [50.0]
    sel_perc = min(report_percs, key=lambda p: abs(float(p) - 50.0))
    sel_key = str(int(sel_perc))
    operator_general = context.get("operator_general", {}) if isinstance(context, dict) else {}
    finance = operator_general.get("finance", {}) if isinstance(operator_general, dict) else {}
    tax = operator_general.get("tax", {}) if isinstance(operator_general, dict) else {}
    wc = operator_general.get("working_capital", {}) if isinstance(operator_general, dict) else {}
    loan = operator_general.get("loan", {}) if isinstance(operator_general, dict) else {}
    dep_sched = (operator_general.get("depreciation_schedule", {}) or {}).get("assets", [])
    dep_const = float(finance.get("depreciation_assets_eur", 0.0) or 0.0)

    # Read inputs produced by pipelines
    pub_scen = read_csv_rows(out_dir / "public_tap_scenarios.csv")
    prv_cust = read_csv_rows(out_dir / "private_tap_customers_by_month.csv")
    prv_costs = read_csv_rows(out_dir / "private_tap_costs_by_month.csv")

    # Monthly series consolidated across runs (use selected percentile). Public filtered to base scenario.
    pub_rev_p = percentiles_sum_by_month(pub_scen, "month", "revenue_eur", report_percs, scenario="base")
    pub_cost_p = percentiles_sum_by_month(pub_scen, "month", "cost_eur", report_percs, scenario="base")
    prv_rev_p = percentiles_sum_by_month(prv_cust, "month", "revenue_eur", report_percs)
    prv_cost_p = percentiles_sum_by_month(prv_costs, "month", "cost_eur", report_percs)
    pub_rev_m = pub_rev_p.get(sel_key, [])
    pub_cost_m = pub_cost_p.get(sel_key, [])
    prv_rev_m = prv_rev_p.get(sel_key, [])
    prv_cost_m = prv_cost_p.get(sel_key, [])
    n_m = max(horizon, len(pub_rev_m), len(prv_rev_m), len(pub_cost_m), len(prv_cost_m))
    pub_rev_m, pub_cost_m, prv_rev_m, prv_cost_m = (
        pad_series(pub_rev_m, n_m), pad_series(pub_cost_m, n_m), pad_series(prv_rev_m, n_m), pad_series(prv_cost_m, n_m)
    )

    # Derived series
    opex_fixed_m = fixed_opex_series(finance, n_m)
    dep_m = depreciation_series(dep_sched, dep_const, n_m)
    revenue_m, cogs_m = revenue_and_cogs(pub_rev_m, prv_rev_m, pub_cost_m, prv_cost_m)
    ebitda_m = ebitda_series(revenue_m, cogs_m, opex_fixed_m)

    loan_sched = build_loan_schedule(loan, n_m)
    interest_m = [row["interest"] for row in loan_sched]
    principal_m = [row["principal_repayment"] for row in loan_sched]

    vat_cash_m = vat_cash_series(tax, wc, revenue_m, cogs_m)
    wc_delta_m = working_capital_deltas(wc, revenue_m, cogs_m)
    ebt_m, tax_m, net_income_m = ebt_tax_net(
        ebitda_m, dep_m, interest_m, float((tax or {}).get("corporate_income_pct", 0.0) or 0.0)
    )

    # Cashflow series
    starting_cash_m = [0.0] * n_m
    loan_amt = float((loan or {}).get("amount_eur", 0.0) or 0.0)
    equity_inj = float((loan or {}).get("equity_injection_eur", 0.0) or 0.0)
    if n_m > 0:
        starting_cash_m[0] = loan_amt + equity_inj
    capex_m = [0.0] * n_m
    if isinstance(dep_sched, list):
        for a in dep_sched:
            try:
                amt = float(a.get("amount_eur", 0) or 0)
                start = int(a.get("start_month_index", 0) or 0)
                if amt > 0 and 0 <= start < n_m:
                    capex_m[start] += amt
            except Exception:
                continue
    cash_from_ops_m = [ebitda_m[i] - tax_m[i] - wc_delta_m[i] - vat_cash_m[i] for i in range(n_m)]
    ending_cash_m: List[float] = []
    cash = 0.0
    for i in range(n_m):
        cash += starting_cash_m[i]
        cash += cash_from_ops_m[i]
        cash -= capex_m[i]
        cash -= interest_m[i]
        cash -= principal_m[i]
        ending_cash_m.append(cash)

    # Write P&L and Cashflow
    written.append(
        write_pnl(
            out_dir, pub_rev_m, prv_rev_m, pub_cost_m, prv_cost_m, opex_fixed_m, dep_m, ebitda_m, interest_m, ebt_m, tax_m, net_income_m
        )
    )
    written.append(
        write_cashflow(
            out_dir, starting_cash_m, cash_from_ops_m, wc_delta_m, vat_cash_m, capex_m, interest_m, principal_m, ending_cash_m
        )
    )

    # Loan schedule CSV
    header = ["month", "principal_opening", "interest", "principal_repayment", "payment", "principal_closing"]
    rows = (
        [f"{r['month']}", f"{r['principal_opening']}", f"{r['interest']}", f"{r['principal_repayment']}", f"{r['payment']}", f"{r['principal_closing']}"]
        for r in loan_sched
    )
    written.append(write_rows(out_dir / "loan_schedule.csv", header, rows))

    # KPIs
    kpis = compute_kpis(ebitda_m, interest_m, principal_m, ending_cash_m)
    (out_dir / "kpi_summary.json").write_text(json.dumps(kpis, indent=2))
    written.append("kpi_summary.json")

    # Consolidated KPIs summary CSV (simple overhead to private for now)
    pub_rev = sum(pub_rev_m)
    pub_cost = sum(pub_cost_m)
    prv_rev = sum(prv_rev_m)
    prv_cost = sum(prv_cost_m)
    total_rev = pub_rev + prv_rev
    overhead_total = sum(v for v in fixed_opex_series(finance, 1))
    alloc_pub = 0.0
    alloc_prv = overhead_total
    pub_cost_alloc = pub_cost + alloc_pub
    prv_cost_alloc = prv_cost + alloc_prv
    pub_margin_alloc = pub_rev - pub_cost_alloc
    prv_margin_alloc = prv_rev - prv_cost_alloc
    header = ["section", "revenue_eur", "cost_eur", "margin_eur", "overhead_allocated_eur"]
    rows = [
        ["public", f"{pub_rev}", f"{pub_cost_alloc}", f"{pub_margin_alloc}", f"{alloc_pub}"],
        ["private", f"{prv_rev}", f"{prv_cost_alloc}", f"{prv_margin_alloc}", f"{alloc_prv}"],
        ["total", f"{total_rev}", f"{pub_cost_alloc + prv_cost_alloc}", f"{(pub_rev + prv_rev) - (pub_cost_alloc + prv_cost_alloc)}", f"{overhead_total}"],
    ]
    written.append(write_rows(out_dir / "consolidated_kpis.csv", header, rows))

    # Autoscaling summary from capacity-by-month
    def _pct(xs: List[float], p: float) -> float:
        if not xs:
            return 0.0
        ys = sorted(xs)
        import math
        k = int(math.ceil(max(0.0, min(100.0, float(p))) / 100.0 * len(ys)))
        k = max(1, min(k, len(ys)))
        return ys[k - 1]

    autoscaling_rows = read_csv_rows(out_dir / "public_tap_capacity_by_month.csv")
    autos_summary: Dict[str, Any] = {}
    if autoscaling_rows:
        utils: List[float] = []
        violations = 0
        HOURS_IN_MONTH = 24.0 * 30.0
        for r in autoscaling_rows:
            try:
                avg_tph = float(r.get("avg_tokens_per_hour", 0.0) or 0.0)
                cap_per_inst = float(r.get("cap_tokens_per_hour_per_instance", 0.0) or 0.0)
                instance_hours = float(r.get("instance_hours", 0.0) or 0.0)
                viol = (str(r.get("capacity_violation", "False")).strip().lower() == "true")
            except Exception:
                continue
            avg_replicas = instance_hours / HOURS_IN_MONTH if HOURS_IN_MONTH > 0 else 0.0
            supply_tph = cap_per_inst * avg_replicas
            util = 0.0 if supply_tph <= 0 else max(0.0, min(1.0, avg_tph / supply_tph))
            utils.append(util * 100.0)
            if viol:
                violations += 1
        autos_summary = {
            "avg_util_pct": (sum(utils) / len(utils)) if utils else 0.0,
            "p95_util_pct": _pct(utils, 95.0),
            "sla_violations": violations,
        }

    # consolidated_summary.json and .md
    summary = {
        "public": {"revenue_eur": pub_rev, "cost_eur": pub_cost, "margin_eur": pub_rev - pub_cost},
        "private": {"revenue_eur": prv_rev, "cost_eur": prv_cost, "margin_eur": prv_rev - prv_cost},
        "total": {"revenue_eur": total_rev, "cost_eur": pub_cost + prv_cost, "margin_eur": (pub_rev + prv_rev) - (pub_cost + prv_cost)},
    }
    autos = autos_summary or context.get("autoscaling_summary") or {}
    if autos:
        summary["autoscaling"] = autos
    percs = {"report_percentiles": report_percs, "selected_percentile": sel_perc}
    if percs:
        summary["percentiles"] = percs
    sens = context.get("sensitivity") or {}
    if sens:
        summary["sensitivity"] = sens
    (out_dir / "consolidated_summary.json").write_text(json.dumps(summary, indent=2))
    written.append("consolidated_summary.json")
    md = [
        "# Consolidated Summary",
        "",
        f"Public: revenue €{pub_rev:.2f}, cost €{pub_cost:.2f}, margin €{(pub_rev - pub_cost):.2f}",
        f"Private: revenue €{prv_rev:.2f}, cost €{prv_cost:.2f}, margin €{(prv_rev - prv_cost):.2f}",
        f"Total: revenue €{(pub_rev + prv_rev):.2f}, cost €{(pub_cost + prv_cost):.2f}, margin €{((pub_rev + prv_rev) - (pub_cost + prv_cost)):.2f}",
    ]
    if autos:
        md.append(f"Autoscaling: p95_util={autos.get('p95_util_pct')}, violations={autos.get('sla_violations')}")
    (out_dir / "consolidated_summary.md").write_text("\n".join(md) + "\n")
    written.append("consolidated_summary.md")

    # LENDER_PACK.md (basic assembly)
    pack_lines = [
        "# Lender Pack",
        "",
        "## Executive Summary",
        f"Min DSCR: {kpis['dscr_min']}",
        f"Min Cash: {kpis['cash_min']}",
        "",
        "## KPIs",
        json.dumps(summary, indent=2),
        "",
        "## Artifacts",
        "- pnl_by_month.csv",
        "- cashflow_by_month.csv",
        "- loan_schedule.csv",
        "- kpi_summary.json",
        "- consolidated_summary.json",
        "- SHA256SUMS",
    ]
    (out_dir / "LENDER_PACK.md").write_text("\n".join(pack_lines) + "\n")
    written.append("LENDER_PACK.md")

    return written
