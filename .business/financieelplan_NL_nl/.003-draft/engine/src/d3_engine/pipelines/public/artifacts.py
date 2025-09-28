"""Public artifacts writing (scaffold)."""
from __future__ import annotations
from pathlib import Path


def _write_csv(path: Path, headers: list[str]) -> None:
    path.write_text(",".join(headers) + "\n")


def write_all(out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    files_and_headers = [
        ("public_vendor_choice.csv", [
            "model", "gpu", "provider", "usd_hr", "eur_hr_effective", "cost_eur_per_1M"
        ]),
        ("public_tap_prices_per_model.csv", [
            "model", "gpu", "cost_eur_per_1M", "sell_eur_per_1k", "margin_pct"
        ]),
        ("public_tap_scenarios.csv", [
            "scenario", "month", "tokens", "revenue_eur", "cost_eur", "margin_eur"
        ]),
        ("public_tap_customers_by_month.csv", [
            "month", "budget_eur", "cac_eur", "expected_new_customers", "active_customers", "tokens"
        ]),
        ("public_tap_capacity_plan.csv", [
            "model", "gpu", "avg_tokens_per_hour", "peak_tokens_per_hour", "tps", "cap_tokens_per_hour_per_instance", "instances_needed", "target_utilization_pct", "capacity_violation"
        ]),
    ]

    for name, headers in files_and_headers:
        p = out_dir / name
        _write_csv(p, headers)
        written.append(name)

    return written
