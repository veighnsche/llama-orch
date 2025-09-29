from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import csv
import json
import math


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


def _read_csv_rows(p: Path) -> List[Dict[str, str]]:
    if not p.exists():
        return []
    with p.open() as f:
        rdr = csv.DictReader(f)
        return [row for row in rdr]


def _sum_by_month(rows: List[Dict[str, str]], month_col: str, val_col: str, scenario: Optional[str] = None) -> List[float]:
    months: Dict[int, float] = {}
    for r in rows:
        if scenario is not None:
            if (r.get("scenario") or "").strip().lower() != scenario:
                continue
        try:
            m = int(float(r.get(month_col, 0) or 0))
            v = float(r.get(val_col, 0) or 0)
        except Exception:
            continue
        months[m] = months.get(m, 0.0) + v
    if not months:
        return []
    n = max(months.keys()) + 1
    return [months.get(i, 0.0) for i in range(n)]


def _build_loan_schedule(loan: Dict[str, Any], horizon: int) -> List[Dict[str, float]]:
    amt = float(loan.get("amount_eur", 0.0) or 0.0)
    term = int(loan.get("term_months", 0) or 0)
    rate_annual = float(loan.get("interest_rate_pct_flat", 0.0) or 0.0) / 100.0
    r = rate_annual / 12.0
    grace = int(loan.get("grace_period_months", 0) or 0)
    typ = str(loan.get("repayment_type", "annuity")).strip().lower()
    sched: List[Dict[str, float]] = []
    bal = amt
    pay = 0.0
    if typ == "annuity" and term > grace and r > 0:
        # Remaining term after grace
        n = max(term - grace, 1)
        pay = bal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    for m in range(horizon):
        interest = bal * r
        principal = 0.0
        if m < grace:
            # Interest-only period
            principal = 0.0
            payment = interest
        else:
            if typ == "annuity" and pay > 0:
                payment = pay
                principal = max(0.0, payment - interest)
                principal = min(principal, bal)
            elif typ == "flat" and term > 0:
                principal = amt / term
                payment = principal + interest
            elif typ == "bullet":
                principal = amt if m == term - 1 else 0.0
                payment = interest + (principal if principal > 0 else 0.0)
            else:
                payment = interest
                principal = 0.0
        bal_next = max(0.0, bal - principal)
        sched.append({
            "month": float(m),
            "principal_opening": bal,
            "interest": interest,
            "principal_repayment": principal,
            "principal_closing": bal_next,
            "payment": payment,
        })
        bal = bal_next
    return sched


def write_consolidated_outputs(out_dir: Path, context: Dict[str, Any]) -> List[str]:
    """Compute consolidated financials, loan schedule, KPIs, and lender pack.

    Returns list of written filenames.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    # Context and inputs
    sim = context.get("simulation", {}) if isinstance(context, dict) else {}
    horizon = int(sim.get("targets", {}).get("horizon_months", 12)) if isinstance(sim, dict) else 12
    operator_general = context.get("operator_general", {}) if isinstance(context, dict) else {}
    finance = operator_general.get("finance", {}) if isinstance(operator_general, dict) else {}
    tax = operator_general.get("tax", {}) if isinstance(operator_general, dict) else {}
    wc = operator_general.get("working_capital", {}) if isinstance(operator_general, dict) else {}
    loan = operator_general.get("loan", {}) if isinstance(operator_general, dict) else {}
    dep_sched = (operator_general.get("depreciation_schedule", {}) or {}).get("assets", [])
    dep_const = float(finance.get("depreciation_assets_eur", 0.0) or 0.0)

    pub_scen = _read_csv_rows(out_dir / "public_tap_scenarios.csv")
    prv_cust = _read_csv_rows(out_dir / "private_tap_customers_by_month.csv")
    prv_costs = _read_csv_rows(out_dir / "private_tap_costs_by_month.csv")
    pub_cap = _read_csv_rows(out_dir / "public_tap_capacity_by_month.csv")

    # Monthly series (base scenario for public)
    pub_rev_m = _sum_by_month(pub_scen, "month", "revenue_eur", scenario="base")
    pub_cost_m = _sum_by_month(pub_scen, "month", "cost_eur", scenario="base")
    prv_rev_m = _sum_by_month(prv_cust, "month", "revenue_eur")
    prv_cost_m = _sum_by_month(prv_costs, "month", "cost_eur")
    n_m = max(horizon, len(pub_rev_m), len(prv_rev_m), len(pub_cost_m), len(prv_cost_m))
    def pad(xs: List[float]) -> List[float]:
        return xs + [0.0] * (n_m - len(xs))
    pub_rev_m, pub_cost_m, prv_rev_m, prv_cost_m = map(pad, [pub_rev_m, pub_cost_m, prv_rev_m, prv_cost_m])

    # Fixed opex per month
    opex_fixed = 0.0
    if isinstance(finance.get("fixed_costs_monthly_eur"), dict):
        try:
            opex_fixed = sum(float(v) for v in finance.get("fixed_costs_monthly_eur", {}).values())
        except Exception:
            opex_fixed = 0.0
    opex_fixed_m = [opex_fixed] * n_m

    # Depreciation per month from schedule or constant
    dep_m = [dep_const] * n_m
    if isinstance(dep_sched, list) and dep_sched:
        dep_m = [0.0] * n_m
        for a in dep_sched:
            try:
                amt = float(a.get("amount_eur", 0) or 0)
                months = int(a.get("months", 0) or 0)
                start = int(a.get("start_month_index", 0) or 0)
                if amt > 0 and months > 0 and start >= 0:
                    per = amt / months
                    for i in range(start, min(n_m, start + months)):
                        dep_m[i] += per
            except Exception:
                continue

    # Revenue & COGS
    revenue_m = [pub_rev_m[i] + prv_rev_m[i] for i in range(n_m)]
    cogs_m = [pub_cost_m[i] + prv_cost_m[i] for i in range(n_m)]
    ebitda_m = [revenue_m[i] - cogs_m[i] - opex_fixed_m[i] for i in range(n_m)]

    # Loan schedule and debt service
    loan_sched = _build_loan_schedule(loan, n_m)
    interest_m = [row["interest"] for row in loan_sched]
    principal_m = [row["principal_repayment"] for row in loan_sched]

    # Taxes & VAT
    vat_pct = float(tax.get("vat_pct", 0.0) or 0.0) / 100.0
    corp_pct = float(tax.get("corporate_income_pct", 0.0) or 0.0) / 100.0
    vat_lag = int(wc.get("vat_payment_lag_months", 0) or 0)
    vat_in_m = [revenue * vat_pct for revenue in revenue_m]
    vat_on_purchases_m = [cogs * vat_pct for cogs in cogs_m]
    vat_payable_m = [vat_in_m[i] - vat_on_purchases_m[i] for i in range(n_m)]
    vat_cash_m = [0.0] * n_m
    for i in range(n_m):
        j = i + vat_lag
        if 0 <= j < n_m:
            vat_cash_m[j] += max(0.0, vat_payable_m[i])

    # Working capital: AR/AP days
    ar_days = float(wc.get("ar_days", 0.0) or 0.0)
    ap_days = float(wc.get("ap_days", 0.0) or 0.0)
    def wc_level(revenue: float, cogs: float) -> float:
        ar = revenue * (ar_days / 30.0)
        ap = cogs * (ap_days / 30.0)
        return ar - ap
    wc_levels = [wc_level(revenue_m[i], cogs_m[i]) for i in range(n_m)]
    wc_delta_m = [0.0] + [wc_levels[i] - wc_levels[i-1] for i in range(1, n_m)]

    # EBT, Tax, NetIncome
    ebt_m = [ebitda_m[i] - dep_m[i] - interest_m[i] for i in range(n_m)]
    tax_m = [max(0.0, ebt_m[i]) * corp_pct for i in range(n_m)]
    net_income_m = [ebt_m[i] - tax_m[i] for i in range(n_m)]

    # Cashflow (simple)
    # Financing inflow month 0
    loan_amt = float(loan.get("amount_eur", 0.0) or 0.0)
    equity_inj = float(loan.get("equity_injection_eur", 0.0) or 0.0)
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
    starting_cash_m = [0.0] * n_m
    starting_cash_m[0] = loan_amt + equity_inj
    debt_service_interest_m = interest_m
    debt_service_principal_m = principal_m
    cash_from_ops_m = [ebitda_m[i] - tax_m[i] - wc_delta_m[i] - vat_cash_m[i] for i in range(n_m)]
    ending_cash_m: List[float] = []
    cash = 0.0
    for i in range(n_m):
        cash += starting_cash_m[i]
        cash += cash_from_ops_m[i]
        cash -= capex_m[i]
        cash -= debt_service_interest_m[i]
        cash -= debt_service_principal_m[i]
        ending_cash_m.append(cash)

    # P&L by month CSV
    pnl_path = out_dir / "pnl_by_month.csv"
    with pnl_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["month","revenue_public","revenue_private","cogs_public","cogs_private","opex_fixed","depreciation","EBITDA","interest","EBT","tax","NetIncome"])
        for i in range(n_m):
            w.writerow([
                str(i), f"{pub_rev_m[i]}", f"{prv_rev_m[i]}", f"{pub_cost_m[i]}", f"{prv_cost_m[i]}", f"{opex_fixed_m[i]}", f"{dep_m[i]}", f"{ebitda_m[i]}", f"{interest_m[i]}", f"{ebt_m[i]}", f"{tax_m[i]}", f"{net_income_m[i]}"
            ])
    written.append(pnl_path.name)

    # Cashflow by month CSV
    cfw_path = out_dir / "cashflow_by_month.csv"
    with cfw_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["month","starting_cash","cash_from_ops","working_capital_delta","vat_cash","capex","debt_service_interest","debt_service_principal","ending_cash"])
        for i in range(n_m):
            w.writerow([
                str(i), f"{starting_cash_m[i]}", f"{cash_from_ops_m[i]}", f"{wc_delta_m[i]}", f"{vat_cash_m[i]}", f"{capex_m[i]}", f"{debt_service_interest_m[i]}", f"{debt_service_principal_m[i]}", f"{ending_cash_m[i]}"
            ])
    written.append(cfw_path.name)

    # Loan schedule CSV
    loan_path = out_dir / "loan_schedule.csv"
    with loan_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["month","principal_opening","interest","principal_repayment","payment","principal_closing"])
        for row in loan_sched:
            w.writerow([
                f"{row['month']}", f"{row['principal_opening']}", f"{row['interest']}", f"{row['principal_repayment']}", f"{row['payment']}", f"{row['principal_closing']}"
            ])
    written.append(loan_path.name)

    # KPIs summary JSON
    dscr_m = []
    for i in range(n_m):
        debt_service = debt_service_interest_m[i] + debt_service_principal_m[i]
        dscr = float('inf') if debt_service <= 0 else (ebitda_m[i] / max(debt_service, 1e-9))
        dscr_m.append(dscr)
    kpis = {
        "dscr_min": (min(dscr_m) if dscr_m else None),
        "icr_min": (min([float('inf') if interest_m[i] <= 0 else (ebitda_m[i] / max(interest_m[i], 1e-9)) for i in range(n_m)]) if n_m > 0 else None),
        "cash_min": (min(ending_cash_m) if ending_cash_m else None),
        "runway_months": next((i for i, c in enumerate(ending_cash_m) if c < 0), n_m),
    }
    kpi_path = out_dir / "kpi_summary.json"
    kpi_path.write_text(json.dumps(kpis, indent=2))
    written.append(kpi_path.name)

    # Consolidated quick KPIs (legacy minimal)
    pub_rev = sum(pub_rev_m)
    pub_cost = sum(pub_cost_m)
    prv_rev = sum(prv_rev_m)
    prv_cost = sum(prv_cost_m)
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
    # Merge sensitivity if provided
    sens = context.get("sensitivity") or {}
    if sens:
        summary["sensitivity"] = sens
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
