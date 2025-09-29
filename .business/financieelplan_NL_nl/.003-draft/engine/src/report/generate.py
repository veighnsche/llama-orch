from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import csv
import json
import datetime as _dt

# --------------------
# Helpers
# --------------------

def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    rows: List[Dict[str, str]] = []
    header: List[str] = []
    try:
        with path.open("r", newline="") as f:
            rdr = csv.DictReader(f)
            header = rdr.fieldnames or []
            for r in rdr:
                rows.append(dict(r))
    except FileNotFoundError:
        pass
    return header, rows


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default


def _sum(xs: List[float]) -> float:
    return float(sum(xs))


# --------------------
# Data loading
# --------------------

def _load_data(outputs_dir: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}

    # summaries
    cons_p = outputs_dir / "consolidated_summary.json"
    kpi_p = outputs_dir / "kpi_summary.json"
    data["consolidated"] = json.loads(cons_p.read_text()) if cons_p.exists() else {}
    data["kpis"] = json.loads(kpi_p.read_text()) if kpi_p.exists() else {}

    # tables
    pnl_hdr, pnl_rows = _read_csv(outputs_dir / "pnl_by_month.csv")
    cf_hdr, cf_rows = _read_csv(outputs_dir / "cashflow_by_month.csv")
    loan_hdr, loan_rows = _read_csv(outputs_dir / "loan_schedule.csv")

    # convert numeric fields for convenience
    def _rows_to_arrays(rows: List[Dict[str, str]], numeric_cols: List[str]) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {c: [] for c in numeric_cols}
        for r in rows:
            for c in numeric_cols:
                out[c].append(_to_float(r.get(c)))
        return out

    # Identify columns by names written in aggregator
    pnl_cols = [
        "month",
        "revenue_public",
        "revenue_private",
        "cogs_public",
        "cogs_private",
        "opex_fixed",
        "depreciation",
        "EBITDA",
        "interest",
        "EBT",
        "tax",
        "NetIncome",
    ]
    cf_cols = [
        "month",
        "starting_cash",
        "cash_from_ops",
        "working_capital_delta",
        "vat_cash",
        "capex",
        "debt_service_interest",
        "debt_service_principal",
        "ending_cash",
    ]

    pnl = _rows_to_arrays(pnl_rows, pnl_cols)
    cf = _rows_to_arrays(cf_rows, cf_cols)

    # Derived series for the report
    n_m = max(len(pnl.get("month", [])), len(cf.get("month", [])))
    # totals
    revenue_total = [
        _to_float(pnl.get("revenue_public", [0.0]*n_m)[i]) + _to_float(pnl.get("revenue_private", [0.0]*n_m)[i])
        for i in range(n_m)
    ]
    cogs_total = [
        _to_float(pnl.get("cogs_public", [0.0]*n_m)[i]) + _to_float(pnl.get("cogs_private", [0.0]*n_m)[i])
        for i in range(n_m)
    ]
    ebitda = [ _to_float(pnl.get("EBITDA", [0.0]*n_m)[i]) for i in range(n_m) ]
    interest = [ _to_float(pnl.get("interest", [0.0]*n_m)[i]) for i in range(n_m) ]
    ds_princ = [ _to_float(cf.get("debt_service_principal", [0.0]*n_m)[i]) for i in range(n_m) ]
    dscr = [
        (float('inf') if (interest[i] + ds_princ[i]) <= 0 else (ebitda[i] / max(interest[i] + ds_princ[i], 1e-9)))
        for i in range(n_m)
    ]

    # Tally justifications
    autos = data.get("consolidated", {}).get("autoscaling", {}) if isinstance(data.get("consolidated"), dict) else {}
    perc = data.get("consolidated", {}).get("percentiles", {}) if isinstance(data.get("consolidated"), dict) else {}

    data.update({
        "monthly": {
            "revenue_total": revenue_total,
            "cogs_total": cogs_total,
            "ebitda": ebitda,
            "interest": interest,
            "debt_service_principal": ds_princ,
            "dscr": dscr,
            "ending_cash": [ _to_float(cf.get("ending_cash", [0.0]*n_m)[i]) for i in range(n_m) ],
            "vat_cash_total": _sum([_to_float(x) for x in cf.get("vat_cash", [])]),
            "wc_delta_total": _sum([_to_float(x) for x in cf.get("working_capital_delta", [])]),
        },
        "meta": {
            "horizon_months": n_m,
            "generated_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "selected_percentile": perc.get("selected_percentile"),
            "report_percentiles": perc.get("report_percentiles"),
        },
        "autoscaling": autos,
        "loan_schedule": loan_rows,  # raw strings are fine for table rendering
        "pnl_rows": pnl_rows,
        "cf_rows": cf_rows,
    })

    return data


# --------------------
# Rendering
# --------------------

def generate_analyzed_report(outputs_dir: Path, analyzed_dir: Path) -> List[str]:
    outputs_dir = Path(outputs_dir)
    analyzed_dir = Path(analyzed_dir)
    analyzed_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = _load_data(outputs_dir)

    # Template load
    tpl_path = Path(__file__).parent / "templates" / "index.html"
    tpl = tpl_path.read_text(encoding="utf-8")

    # Title & metadata
    cons = data.get("consolidated", {}) or {}
    total = cons.get("total", {}) if isinstance(cons, dict) else {}
    title = "Qredits – 18-Month Financial Plan (Analysis)"
    gen_at = data.get("meta", {}).get("generated_at", "")

    # Serialize data for client-side rendering
    data_json = json.dumps(data)

    # Simple placeholder substitution (no external deps)
    html = tpl.replace("%%TITLE%%", title).replace("%%GENERATED_AT%%", gen_at).replace("%%DATA_JSON%%", data_json)

    out_html = analyzed_dir / "index.html"
    out_html.write_text(html, encoding="utf-8")

    # Lightweight README to explain provenance
    readme = analyzed_dir / "README.md"
    readme.write_text(
        """
# Analyzed Report

This folder is generated by the engine, using templates and the numeric artifacts in `../outputs/`.
Do not edit `index.html` by hand; re-run the engine to regenerate.

Sources used:
- `pnl_by_month.csv`
- `cashflow_by_month.csv`
- `loan_schedule.csv`
- `kpi_summary.json`
- `consolidated_summary.json`
""".strip()
        + "\n",
        encoding="utf-8",
    )

    return [str(out_html), str(readme)]
