from __future__ import annotations

from pathlib import Path
import shutil
import os

from finance_engine.config import INPUTS
from finance_engine.engine_pkg.validation.preflight import run_preflight


def copy_inputs(tmp: Path) -> Path:
    dst = tmp / "inputs"
    dst.mkdir(parents=True, exist_ok=True)
    for p in INPUTS.iterdir():
        if p.is_file():
            shutil.copy(p, dst / p.name)
    return dst


def test_strict_missing_prices_not_required(tmp_path, monkeypatch):
    d = copy_inputs(tmp_path)
    # unit prices in price_sheet are optional; pricing is computed from policy and costs
    monkeypatch.setenv("ENGINE_STRICT_VALIDATION", "1")
    res = run_preflight(d)
    assert res.ok
    monkeypatch.delenv("ENGINE_STRICT_VALIDATION", raising=False)


def test_strict_mixed_units_error(tmp_path, monkeypatch):
    d = copy_inputs(tmp_path)
    lines = (d / "price_sheet.csv").read_text(encoding="utf-8").splitlines()
    header = lines[0]
    rows = [header]
    for i, line in enumerate(lines[1:], start=1):
        parts = line.split(",")
        if len(parts) >= 3 and parts[1] == "public_tap":
            parts[2] = "1M_tokens"  # set one row to a different unit than default 1k_tokens
            rows.append(",".join(parts))
            rows.extend(lines[i+1:])
            break
    (d / "price_sheet.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("ENGINE_STRICT_VALIDATION", "true")
    res = run_preflight(d)
    assert not res.ok
    monkeypatch.delenv("ENGINE_STRICT_VALIDATION", raising=False)


def test_strict_coverage_error(tmp_path, monkeypatch):
    d = copy_inputs(tmp_path)
    # remove a SKU to trigger coverage error (heuristic vs pseudo_skus)
    lines = (d / "price_sheet.csv").read_text(encoding="utf-8").splitlines()
    header = lines[0]
    filtered = [header] + [ln for ln in lines[1:] if "Mixtral-8x7B" not in ln]
    (d / "price_sheet.csv").write_text("\n".join(filtered) + "\n", encoding="utf-8")
    monkeypatch.setenv("ENGINE_STRICT_VALIDATION", "yes")
    res = run_preflight(d)
    assert not res.ok
    monkeypatch.delenv("ENGINE_STRICT_VALIDATION", raising=False)
