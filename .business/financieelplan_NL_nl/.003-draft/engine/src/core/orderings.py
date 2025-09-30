"""Deterministic table orderings for CSV materialization."""
from __future__ import annotations
from typing import List, Dict


def sorted_rows(name: str, rows: List[Dict]) -> List[Dict]:
    if name == "public_vendor_choice":
        return sorted(rows, key=lambda r: (r.get("model", ""), r.get("gpu", "")))
    if name == "public_tap_prices_per_model":
        return sorted(rows, key=lambda r: (r.get("model", ""), r.get("gpu", "")))
    if name == "public_tap_scenarios":
        return sorted(rows, key=lambda r: (r.get("scenario", ""), int(r.get("month", 0))))
    if name == "public_tap_customers_by_month":
        return sorted(rows, key=lambda r: (int(r.get("month", 0))))
    if name == "public_tap_capacity_plan":
        return sorted(rows, key=lambda r: (r.get("model", ""), r.get("gpu", "")))
    if name == "private_tap_economics":
        return sorted(rows, key=lambda r: (r.get("gpu", "")))
    if name == "private_vendor_recommendation":
        return sorted(rows, key=lambda r: (r.get("gpu", ""), r.get("provider", "")))
    if name == "private_tap_customers_by_month":
        return sorted(rows, key=lambda r: (int(r.get("month", 0))))
    return rows
