"""Private artifacts writing (scaffold)."""
from __future__ import annotations
from pathlib import Path


def _write_csv(path: Path, headers: list[str]) -> None:
    path.write_text(",".join(headers) + "\n")


def write_all(out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    files_and_headers = [
        ("private_tap_economics.csv", [
            "gpu", "provider_eur_hr_med", "markup_pct", "sell_eur_hr", "margin_eur_hr", "margin_pct"
        ]),
        ("private_vendor_recommendation.csv", [
            "gpu", "reason", "score"
        ]),
        ("private_tap_customers_by_month.csv", [
            "month", "private_budget_eur", "private_cac_eur", "expected_new_clients", "active_clients", "hours", "sell_eur_hr", "revenue_eur"
        ]),
    ]

    for name, headers in files_and_headers:
        p = out_dir / name
        _write_csv(p, headers)
        written.append(name)

    return written
