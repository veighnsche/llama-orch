from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple
import sys
import csv
import datetime as dt
import json

import pandas as pd


def _load_yaml_or_empty(path: Path) -> Dict[str, Any]:
    import yaml

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, obj: Dict[str, Any]) -> None:
    import yaml

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=True, allow_unicode=True)


def reconcile(inputs_dir: Path) -> Dict[str, Any]:
    """
    Merge non-expired overrides back into base files and drop expired entries.

    - price_overrides → price_sheet.csv (unit_price_eur_per_1k_tokens)
    - capacity_overrides → tps_model_gpu.csv (append/update tps as a new row tagged source=override)

    Returns a report dict summarizing changes.
    """
    inputs_dir = Path(inputs_dir)
    today = dt.date.today()
    report: Dict[str, Any] = {"merged": {}, "removed": {}}

    # Prices
    overrides_path = inputs_dir / "overrides.yaml"
    ov = _load_yaml_or_empty(overrides_path)
    price_ov = ov.get("price_overrides", {}) if isinstance(ov, dict) else {}

    if price_ov:
        ps_path = inputs_dir / "price_sheet.csv"
        df = pd.read_csv(ps_path)
        merged_skus = []
        removed_skus = []
        keep_entries = {}
        for sku, node in price_ov.items():
            exp = node.get("expires_on")
            expired = False
            if isinstance(exp, str) and exp.strip():
                try:
                    expired = dt.date.fromisoformat(exp) < today
                except Exception:
                    # invalid date → treat as expired for safety
                    expired = True
            if not expired:
                # merge into price_sheet
                if "unit_price_eur_per_1k_tokens" in df.columns:
                    mask = df["sku"].astype(str) == sku
                    if mask.any():
                        df.loc[mask, "unit_price_eur_per_1k_tokens"] = float(node["unit_price_eur_per_1k_tokens"])
                        merged_skus.append(sku)
                keep_entries[sku] = node
            else:
                removed_skus.append(sku)
        df.to_csv(ps_path, index=False)
        ov["price_overrides"] = keep_entries
        _write_yaml(overrides_path, ov)
        report["merged"]["price_overrides"] = merged_skus
        report["removed"]["price_overrides"] = removed_skus

    # Capacity
    cap_path = inputs_dir / "capacity_overrides.yaml"
    cap = _load_yaml_or_empty(cap_path)
    cap_ov = cap.get("capacity_overrides", {}) if isinstance(cap, dict) else {}
    if cap_ov:
        tps_path = inputs_dir / "tps_model_gpu.csv"
        df = pd.read_csv(tps_path)
        merged = []
        removed = []
        keep_cap = {}
        for sku, node in cap_ov.items():
            exp = node.get("expires_on")
            expired = False
            if isinstance(exp, str) and exp.strip():
                try:
                    expired = dt.date.fromisoformat(exp) < today
                except Exception:
                    expired = True
            if not expired:
                # Append/update an override row (tag source_tag=OVERRIDE)
                tps = node.get("tps_override_tokens_per_sec")
                gpu = node.get("preferred_gpu")
                if tps is not None:
                    new_row = {
                        "model_name": sku,
                        "model_name_normalized": sku,
                        "engine": "override",
                        "precision": "N/A",
                        "gpu": gpu or "N/A",
                        "gpu_count": 1,
                        "batch": 1,
                        "input_tokens": 512,
                        "output_tokens": 512,
                        "throughput_tokens_per_sec": float(tps),
                        "measurement_type": "override",
                        "scenario_notes": "reconciled override",
                        "source_tag": "OVERRIDE",
                    }
                    # Strategy: append; consumers can choose median across rows
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                keep_cap[sku] = node
                merged.append(sku)
            else:
                removed.append(sku)
        df.to_csv(tps_path, index=False)
        cap["capacity_overrides"] = keep_cap
        _write_yaml(cap_path, cap)
        report["merged"]["capacity_overrides"] = merged
        report["removed"]["capacity_overrides"] = removed

    return report


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m tools.reconcile_overrides <inputs_dir>", file=sys.stderr)
        sys.exit(2)
    path = Path(sys.argv[1])
    if not path.exists() or not path.is_dir():
        print(f"error: {path} is not a directory", file=sys.stderr)
        sys.exit(2)
    rep = reconcile(path)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
