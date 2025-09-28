"""Private artifacts writing (MRPT v0)."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple
import statistics


def _write_csv_header(path: Path, headers: list[str]) -> None:
    if not path.exists():
        path.write_text(",".join(headers) + "\n")


def _append_row(path: Path, row: List[str]) -> None:
    with path.open("a") as f:
        f.write(",".join(row) + "\n")


def _fx_eur_per_usd(facts: Dict[str, Any]) -> float:
    try:
        return float(
            facts.get("market_env", {})
            .get("finance", {})
            .get("eur_usd_fx_rate", {})
            .get("value")
        )
    except Exception:
        return 1.0


def _fx_buffer_pct(prv_op: Dict[str, Any]) -> float:
    try:
        return float(prv_op.get("meta", {}).get("fx_buffer_pct", 0.0))
    except Exception:
        return 0.0


def _eur_hr(usd_hr: float, eur_usd_rate: float, fx_buffer_pct: float) -> float:
    if eur_usd_rate <= 0:
        eur_usd_rate = 1.0
    return (usd_hr / eur_usd_rate) * (1.0 + max(fx_buffer_pct, 0.0) / 100.0)


def write_all(out_dir: Path, state: Dict[str, Any]) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    econ_hdr = ["gpu", "provider_eur_hr_med", "markup_pct", "sell_eur_hr", "margin_eur_hr", "margin_pct"]
    reco_hdr = ["gpu", "provider", "usd_hr", "eur_hr_effective", "score"]
    cust_hdr = ["month", "private_budget_eur", "private_cac_eur", "expected_new_clients", "active_clients", "hours", "sell_eur_hr", "revenue_eur"]

    econ_path = out_dir / "private_tap_economics.csv"
    reco_path = out_dir / "private_vendor_recommendation.csv"
    cust_path = out_dir / "private_tap_customers_by_month.csv"

    for p, h in (
        (econ_path, econ_hdr),
        (reco_path, reco_hdr),
        (cust_path, cust_hdr),
    ):
        _write_csv_header(p, h)
        written.append(p.name)

    rentals: List[Dict[str, Any]] = state.get("curated", {}).get("gpu_rentals", [])
    facts: Dict[str, Any] = state.get("facts", {})
    prv_op: Dict[str, Any] = state.get("operator", {}).get("private_tap", {})

    eur_usd_rate = _fx_eur_per_usd(facts)
    fx_buf = _fx_buffer_pct(prv_op)
    pp = prv_op.get("pricing_policy", {}).get("private_tap", {}) if isinstance(prv_op, dict) else {}
    markup_pct = float(pp.get("default_markup_over_provider_cost_pct", 40.0))
    mgmt_fee = float(pp.get("management_fee_eur_per_month", 0.0))

    # Group rentals by GPU name
    by_gpu: Dict[str, List[Tuple[str, float, float]]] = {}
    for r in rentals:
        gpu = str(r.get("gpu"))
        provider = str(r.get("provider"))
        usd = float(r.get("usd_hr"))
        eur = _eur_hr(usd, eur_usd_rate, fx_buf)
        by_gpu.setdefault(gpu, []).append((provider, usd, eur))

    for gpu, lst in by_gpu.items():
        if not lst:
            continue
        eur_list = [eur for (_, _, eur) in lst]
        eur_med = statistics.median(eur_list)
        sell_eur_hr = eur_med * (1.0 + max(markup_pct, 0.0) / 100.0)
        margin_eur_hr = sell_eur_hr - eur_med
        margin_pct = 0.0 if sell_eur_hr <= 0 else (margin_eur_hr / sell_eur_hr) * 100.0
        _append_row(econ_path, [
            gpu, f"{eur_med}", f"{markup_pct}", f"{sell_eur_hr}", f"{margin_eur_hr}", f"{margin_pct}"
        ])

        # Recommendation: cheapest provider (by EUR/hr)
        best = min(lst, key=lambda t: t[2])
        _append_row(reco_path, [gpu, best[0], f"{best[1]}", f"{best[2]}", "1.0"])

        # Simple month0 customers for private
        acq = prv_op.get("acquisition", {}) if isinstance(prv_op, dict) else {}
        budget0 = float(acq.get("budget_month0_eur", 0.0) or 0.0)
        cac = float(acq.get("cac_base_eur", 0.0) or 0.0)
        hours_per_client = float(acq.get("hours_per_client_month_mean", 0.0) or 0.0)
        churn_pct = float(acq.get("churn_pct_mom", 0.0) or 0.0)
        month = 0
        new = 0.0 if cac <= 0 else (budget0 / cac)
        active = new * (1.0 - max(churn_pct, 0.0) / 100.0)
        hours = active * hours_per_client
        revenue = hours * sell_eur_hr + mgmt_fee
        _append_row(cust_path, [
            str(month), f"{budget0}", f"{cac}", f"{new}", f"{active}", f"{hours}", f"{sell_eur_hr}", f"{revenue}"
        ])

    return written
