from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import json
import math
import datetime as _dt

from aggregate.loan import build_loan_schedule
from core import loader as input_loader

# Jinja2 is optional; if missing, we emit a friendly error via caller.
try:
    from jinja2 import Environment, FileSystemLoader
except Exception:  # pragma: no cover
    Environment = None  # type: ignore
    FileSystemLoader = None  # type: ignore


def _read_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    try:
        with path.open("r", newline="") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                rows.append(dict(r))
    except FileNotFoundError:
        pass
    return rows


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if not s or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default


def _sum(xs: List[float]) -> float:
    return float(sum(xs))


def _unique(xs: List[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for v in xs:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _pct(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    ys = sorted(vals)
    k = int(math.ceil(max(0.0, min(100.0, p)) / 100.0 * len(ys)))
    k = max(1, min(k, len(ys)))
    return ys[k - 1]


def _deep_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def _deep_find_key(d: Dict[str, Any], key: str) -> Optional[Any]:
    # find first occurrence of key in nested dicts
    if not isinstance(d, dict):
        return None
    if key in d:
        return d[key]
    for v in d.values():
        if isinstance(v, dict):
            found = _deep_find_key(v, key)
            if found is not None:
                return found
    return None


def _build_model_price_rows(prices_csv: Path) -> Tuple[str, float, Dict[str, float]]:
    rows = _read_csv(prices_csv)
    lines: List[str] = []
    sell_per_1k: List[float] = []
    price_map: Dict[str, float] = {}
    for r in rows:
        model = (r.get("model") or "").strip()
        gpu = (r.get("gpu") or "").strip()
        cost_1m = _to_float(r.get("cost_eur_per_1M"))
        sell_1k = _to_float(r.get("sell_eur_per_1k"))
        sell_per_1k.append(sell_1k)
        if model:
            price_map[model] = sell_1k
        sell_1m = sell_1k * 1000.0
        gross_1m = sell_1m - cost_1m
        gross_pct = (0.0 if sell_1m <= 0 else (gross_1m / sell_1m) * 100.0)
        line = f"| {model} | {gpu} | {cost_1m:,.2f} | {cost_1m:,.2f} | {cost_1m:,.2f} | {sell_1m:,.2f} | {gross_1m:,.2f} | {gross_pct:,.1f}% |"
        lines.append(line)
    blended = (sum(sell_per_1k) / len(sell_per_1k)) if sell_per_1k else 0.0
    return ("\n".join(lines), blended, price_map)


def _build_private_gpu_econ_rows(prv_csv: Path) -> str:
    rows = _read_csv(prv_csv)
    lines: List[str] = []
    for r in rows:
        gpu = (r.get("gpu") or "").strip()
        prov = _to_float(r.get("provider_eur_hr_med"))
        markup = _to_float(r.get("markup_pct"))
        sell = _to_float(r.get("sell_eur_hr"))
        gm_eur = _to_float(r.get("margin_eur_hr"))
        line = f"| {gpu} | {prov:,.2f} | {markup:,.1f}% | {sell:,.2f} | {gm_eur:,.2f} |"
        lines.append(line)
    return "\n".join(lines)


def _build_loan_schedule_rows_full(loan: Dict[str, Any]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    term = int(_to_float(loan.get("term_months"), 0))
    if term <= 0:
        term = 60
    sched = build_loan_schedule(loan, term)
    total_interest = _sum([_to_float(r.get("interest")) for r in sched])
    total_payment = _sum([_to_float(r.get("payment")) for r in sched])
    monthly_payment = _to_float(sched[min(len(sched)-1, max(0, int(loan.get("grace_period_months", 0))))].get("payment")) if sched else 0.0
    return sched, {
        "monthly_payment_eur": monthly_payment,
        "total_interest_eur": total_interest,
        "total_repayment_eur": total_payment,
    }


def _build_scenarios(outputs_dir: Path, marketing_pct: float, fixed_total_with_loan: float) -> Dict[str, Any]:
    # Use public_tap_scenarios.csv for baseline monthly values; if not present, return minimal skeleton.
    rows = _read_csv(outputs_dir / "public_tap_scenarios.csv")
    # Pick month 0 rows across runs (if present)
    m0 = [r for r in rows if str(r.get("month", "0")).strip() == "0"]
    # Value lists across runs
    revs = [_to_float(r.get("revenue_eur")) for r in m0]
    cogs = [_to_float(r.get("cost_eur")) for r in m0]
    toks = [_to_float(r.get("tokens")) for r in m0]
    if not m0 and rows:
        # fallback first row
        revs = [_to_float(rows[0].get("revenue_eur"))]
        cogs = [_to_float(rows[0].get("cost_eur"))]
        toks = [_to_float(rows[0].get("tokens"))]
    def pack(rev: float, c: float) -> Dict[str, Any]:
        gm = rev - c
        marketing = max(0.0, rev * (marketing_pct / 100.0))
        net = gm - fixed_total_with_loan - marketing
        m_tokens = (toks[0] / 1_000_000.0) if toks else 0.0
        return {
            "m_tokens": f"{m_tokens:,.2f}",
            "revenue_eur": f"{rev:,.2f}",
            "cogs_eur": f"{c:,.2f}",
            "gross_margin_eur": f"{gm:,.2f}",
            "gross_margin_pct": ("N/A" if rev <= 0 else f"{(gm/rev)*100.0:,.1f}%"),
            "marketing_reserved_eur": f"{marketing:,.2f}",
            "net_eur": f"{net:,.2f}",
        }
    if revs:
        worst = pack(_pct(revs, 10.0), _pct(cogs, 10.0))
        base = pack(_pct(revs, 50.0), _pct(cogs, 50.0))
        best = pack(_pct(revs, 90.0), _pct(cogs, 90.0))
    else:
        worst = pack(0.0, 0.0)
        base = pack(0.0, 0.0)
        best = pack(0.0, 0.0)
    # Yearly / 60m snapshots as simple multiples (approximation; improve if full projections are available)
    yearly = {
        "fixed_total": f"{fixed_total_with_loan * 12:,.2f}",
        "worst": {"total_revenue": worst["revenue_eur"], "cogs": worst["cogs_eur"], "gross": worst["gross_margin_eur"], "marketing": f"{_to_float(worst["marketing_reserved_eur"].replace(',', ''))*12:,.2f}", "net": f"{_to_float(worst["net_eur"].replace(',', ''))*12:,.2f}"},
        "base":  {"total_revenue": base["revenue_eur"],  "cogs": base["cogs_eur"],  "gross": base["gross_margin_eur"],  "marketing": f"{_to_float(base["marketing_reserved_eur"].replace(',', ''))*12:,.2f}",  "net": f"{_to_float(base["net_eur"].replace(',', ''))*12:,.2f}"},
        "best":  {"total_revenue": best["revenue_eur"],  "cogs": best["cogs_eur"],  "gross": best["gross_margin_eur"],  "marketing": f"{_to_float(best["marketing_reserved_eur"].replace(',', ''))*12:,.2f}",  "net": f"{_to_float(best["net_eur"].replace(',', ''))*12:,.2f}"},
    }
    sixty = {
        "fixed_total": f"{fixed_total_with_loan * 60:,.2f}",
        "worst": {"total_revenue": f"{_to_float(worst["revenue_eur"].replace(',', ''))*60:,.2f}", "cogs": f"{_to_float(worst["cogs_eur"].replace(',', ''))*60:,.2f}", "gross": f"{_to_float(worst["gross_margin_eur"].replace(',', ''))*60:,.2f}", "marketing": f"{_to_float(worst["marketing_reserved_eur"].replace(',', ''))*60:,.2f}", "net": f"{_to_float(worst["net_eur"].replace(',', ''))*60:,.2f}"},
        "base":  {"total_revenue": f"{_to_float(base["revenue_eur"].replace(',', ''))*60:,.2f}",  "cogs": f"{_to_float(base["cogs_eur"].replace(',', ''))*60:,.2f}",  "gross": f"{_to_float(base["gross_margin_eur"].replace(',', ''))*60:,.2f}",  "marketing": f"{_to_float(base["marketing_reserved_eur"].replace(',', ''))*60:,.2f}",  "net": f"{_to_float(base["net_eur"].replace(',', ''))*60:,.2f}"},
        "best":  {"total_revenue": f"{_to_float(best["revenue_eur"].replace(',', ''))*60:,.2f}",  "cogs": f"{_to_float(best["cogs_eur"].replace(',', ''))*60:,.2f}",  "gross": f"{_to_float(best["gross_margin_eur"].replace(',', ''))*60:,.2f}",  "marketing": f"{_to_float(best["marketing_reserved_eur"].replace(',', ''))*60:,.2f}",  "net": f"{_to_float(best["net_eur"].replace(',', ''))*60:,.2f}"},
    }
    return {"worst": worst, "base": base, "best": best, "yearly": yearly, "60m": sixty}


def render_markdown_template(inputs_dir: Path, outputs_dir: Path, analyzed_dir: Path, state: Dict[str, Any], template_filename: str = "template.md") -> str:
    analyzed_dir = Path(analyzed_dir)
    analyzed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = Path(outputs_dir)
    inputs_dir = Path(inputs_dir)

    # 1) Load outputs
    cons = json.loads((outputs_dir / "consolidated_summary.json").read_text()) if (outputs_dir / "consolidated_summary.json").exists() else {}
    kpis = json.loads((outputs_dir / "kpi_summary.json").read_text()) if (outputs_dir / "kpi_summary.json").exists() else {}
    loan_csv = _read_csv(outputs_dir / "loan_schedule.csv")

    # 2) Extract inputs from state
    op_gen = state.get("operator", {}).get("general", {}) if isinstance(state, dict) else {}
    op_pub = state.get("operator", {}).get("public_tap", {}) if isinstance(state, dict) else {}
    op_prv = state.get("operator", {}).get("private_tap", {}) if isinstance(state, dict) else {}
    finance = op_gen.get("finance", {}) if isinstance(op_gen, dict) else {}
    tax = op_gen.get("tax", {}) if isinstance(op_gen, dict) else {}
    loan = op_gen.get("loan", {}) if isinstance(op_gen, dict) else {}

    # Loan enrich
    # normalize interest key for template
    if "interest_rate_pct" not in loan and "interest_rate_pct_flat" in loan:
        try:
            loan["interest_rate_pct"] = float(loan.get("interest_rate_pct_flat") or 0.0)
        except Exception:
            loan["interest_rate_pct"] = 0.0
    # compute full term schedule
    sched_full, loan_totals = _build_loan_schedule_rows_full(loan)

    # Fixed costs breakdown
    fixed_map = (finance or {}).get("fixed_costs_monthly_eur", {})
    fixed_personal = _to_float(_deep_find_key({"fixed": fixed_map}, "personal")) or 0.0
    # business = sum of others
    fixed_business = 0.0
    if isinstance(fixed_map, dict):
        for k, v in fixed_map.items():
            if str(k).strip().lower() != "personal":
                fixed_business += _to_float(v)
    fixed_total_with_loan = fixed_personal + fixed_business + loan_totals.get("monthly_payment_eur", 0.0)

    # Pricing
    target_gm_pct = _deep_find_key(op_pub, "target_gross_margin_pct")
    fx_buffer_pct = _deep_find_key(op_pub, "fx_buffer_pct")
    if fx_buffer_pct is None:
        fx_buffer_pct = _deep_find_key(state.get("facts", {}).get("market_env", {}), "fx_buffer_pct")

    # Prepaid policy (best-effort)
    prepaid = op_gen.get("prepaid", {}) if isinstance(op_gen, dict) else {}
    if not prepaid:
        prepaid = op_pub.get("prepaid", {}) if isinstance(op_pub, dict) else {}

    # Catalog
    curated_models = state.get("curated", {}).get("public_models", [])
    allowed_models = _unique([(m.get("Model") or m.get("model") or "").strip() for m in curated_models if (m.get("Model") or m.get("model"))])
    curated_gpu = state.get("curated", {}).get("gpu_rentals", [])
    allowed_gpus = _unique([(g.get("gpu") or "").strip() for g in curated_gpu if g.get("gpu")])

    # Public pricing rows + blended price
    model_price_table, blended_1k, model_price_map = _build_model_price_rows(outputs_dir / "public_tap_prices_per_model.csv")

    # Private GPU economics rows
    private_gpu_econ_table = _build_private_gpu_econ_rows(outputs_dir / "private_tap_economics.csv")

    # Scenarios (monthly snapshot worst/base/best) and yearly/60m approximations
    marketing_pct = _to_float(_deep_find_key(finance, "marketing_allocation_pct_of_inflow"), 0.0)
    scenarios = _build_scenarios(outputs_dir, marketing_pct, fixed_total_with_loan)

    # 3) Build context for template
    ctx: Dict[str, Any] = {
        "loan": {
            "amount_eur": _to_float(loan.get("amount_eur"), _to_float(loan_csv[0].get("principal_opening") if loan_csv else 0.0)),
            "term_months": int(_to_float(loan.get("term_months"), len(sched_full))),
            "interest_rate_pct": _to_float(loan.get("interest_rate_pct"), 0.0),
            "monthly_payment_eur": f"{loan_totals.get('monthly_payment_eur', 0.0):,.2f}",
            "total_repayment_eur": f"{loan_totals.get('total_repayment_eur', 0.0):,.2f}",
            "total_interest_eur": f"{loan_totals.get('total_interest_eur', 0.0):,.2f}",
        },
        "fixed": {
            "personal": f"{fixed_personal:,.2f}",
            "business": f"{fixed_business:,.2f}",
            "total_with_loan": f"{fixed_total_with_loan:,.2f}",
        },
        "pricing": {
            "policy": {"target_gross_margin_pct": target_gm_pct},
            "fx_buffer_pct": fx_buffer_pct,
            "public_tap_blended_price_per_1k_tokens": f"{blended_1k:,.2f}",
            "model_prices": model_price_map,
            "private_tap_default_markup_over_provider_cost_pct": _deep_find_key(op_prv, "default_markup_over_cost_pct"),
        },
        "private": {
            "management_fee_eur_per_month": _deep_find_key(op_prv, "management_fee_eur_per_month"),
            "default_markup_over_provider_cost_pct": _deep_find_key(op_prv, "default_markup_over_cost_pct"),
        },
        "prepaid": {
            **(prepaid if isinstance(prepaid, dict) else {}),
            "private_tap": {"billing_unit_minutes": _deep_find_key(op_prv, "billing_unit_minutes")},
        },
        "catalog": {
            "allowed_models": allowed_models,
            "allowed_gpus": allowed_gpus,
        },
        "tax": {
            **(tax if isinstance(tax, dict) else {}),
        },
        "targets": state.get("simulation", {}).get("targets", {}),
        "finance": finance,
        "tables": {
            "model_price_per_1m_tokens": model_price_table,
            "private_tap_gpu_economics": private_gpu_econ_table,
            "loan_schedule": "\n".join(
                [
                    f"| {int(r['month']):d} | { _to_float(r['principal_opening']):,.2f} | { _to_float(r['interest']):,.2f} | { _to_float(r['principal_repayment']):,.2f} | { _to_float(r['payment']):,.2f} | { _to_float(r['principal_closing']):,.2f} |"
                    for r in sched_full
                ]
            ),
        },
        "scenarios": {
            "worst": scenarios["worst"],
            "base": scenarios["base"],
            "best": scenarios["best"],
            "yearly": scenarios["yearly"],
            "60m": scenarios["60m"],
        },
        "engine": {
            "version": None,
            "timestamp": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        # Default acquisition block so template conditionals evaluate cleanly
        "acquisition": {"driver": "tokens"},
        "fx": state.get("facts", {}).get("market_env", {}).get("fx", {}),
        "legend": {
            "measured": ["GPU USD/hr", "TPS baselines"],
            "policy": ["Target GM%", "FX buffer", "VAT", "Prepaid"],
            "estimated": ["Throughput (future telemetry)", "Acquisition model"],
        },
    }

    # 4) Render with Jinja2
    if Environment is None or FileSystemLoader is None:
        raise RuntimeError("Jinja2 is not installed. Install with: pacman -S python-jinja (Arch) or pip install jinja2")

    env = Environment(loader=FileSystemLoader(str(Path(__file__).parent / "templates")), autoescape=False)
    tpl = env.get_template(template_filename)
    md = tpl.render(**ctx)

    out_md = analyzed_dir / "FINANCIAL_PLAN.md"
    out_md.write_text(md, encoding="utf-8")
    return str(out_md)
