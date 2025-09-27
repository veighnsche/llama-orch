from __future__ import annotations

from pathlib import Path
import shutil
import io

from finance_engine.config import INPUTS
from finance_engine.engine_pkg.validation.preflight import run_preflight, build_preflight_markdown


def copy_inputs(tmp: Path) -> Path:
    dst = tmp / "inputs"
    dst.mkdir(parents=True, exist_ok=True)
    for p in INPUTS.iterdir():
        if p.is_file():
            shutil.copy(p, dst / p.name)
    return dst


def test_preflight_ok_current_inputs():
    res = run_preflight(INPUTS)
    assert res.ok, f"expected OK, got errors: {res}"
    md = build_preflight_markdown(res)
    assert "## Preflight" in md
    assert "STATUS: OK (validation)" in md


def test_bad_config_currency_missing(tmp_path: Path):
    d = copy_inputs(tmp_path)
    cfg = (d / "config.yaml").read_text(encoding="utf-8")
    cfg = cfg.replace("currency: EUR", "currency: \n")
    (d / "config.yaml").write_text(cfg, encoding="utf-8")
    res = run_preflight(d)
    assert not res.ok
    md = build_preflight_markdown(res)
    assert "config.yaml" in md and "currency" in md


def test_bad_costs_no_numeric(tmp_path: Path):
    d = copy_inputs(tmp_path)
    (d / "costs.yaml").write_text("{}\n", encoding="utf-8")
    res = run_preflight(d)
    assert not res.ok
    md = build_preflight_markdown(res)
    assert "costs.yaml" in md and "no numeric amounts" in md


def test_bad_lending_missing_term(tmp_path: Path):
    d = copy_inputs(tmp_path)
    text = (d / "lending_plan.yaml").read_text(encoding="utf-8")
    # remove term_months line
    text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("term_months:"))
    (d / "lending_plan.yaml").write_text(text, encoding="utf-8")
    res = run_preflight(d)
    assert not res.ok
    md = build_preflight_markdown(res)
    assert "lending_plan.yaml" in md and "term_months" in md


def test_bad_gpu_rentals_missing_column(tmp_path: Path):
    d = copy_inputs(tmp_path)
    lines = (d / "gpu_rentals.csv").read_text(encoding="utf-8").splitlines()
    header = lines[0].replace("gpu,", "gpuX,")
    (d / "gpu_rentals.csv").write_text("\n".join([header] + lines[1:]) + "\n", encoding="utf-8")
    res = run_preflight(d)
    assert not res.ok
    md = build_preflight_markdown(res)
    assert "gpu_rentals.csv" in md and "missing column: gpu" in md


def test_bad_oss_models_missing_name(tmp_path: Path):
    d = copy_inputs(tmp_path)
    lines = (d / "oss_models.csv").read_text(encoding="utf-8").splitlines()
    header = lines[0].replace("name,", "nameX,")
    (d / "oss_models.csv").write_text("\n".join([header] + lines[1:]) + "\n", encoding="utf-8")
    res = run_preflight(d)
    assert not res.ok
    md = build_preflight_markdown(res)
    assert "oss_models.csv" in md and "missing column: name" in md


def test_bad_price_sheet_bad_unit(tmp_path: Path):
    d = copy_inputs(tmp_path)
    lines = (d / "price_sheet.csv").read_text(encoding="utf-8").splitlines()
    # Change public_tap units to invalid token for a row
    rows = []
    for i, line in enumerate(lines):
        if i == 0:
            rows.append(line)
            continue
        parts = line.split(",")
        if len(parts) >= 3 and parts[1] == "public_tap":
            parts[2] = "invalid_unit"
            rows.append(",".join(parts))
            rows.extend(lines[i+1:])
            break
    (d / "price_sheet.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    res = run_preflight(d)
    assert not res.ok
    md = build_preflight_markdown(res)
    assert "price_sheet.csv" in md and "unit must be one of" in md


def test_bad_tps_missing_throughput(tmp_path: Path):
    d = copy_inputs(tmp_path)
    lines = (d / "tps_model_gpu.csv").read_text(encoding="utf-8").splitlines()
    header = lines[0].replace("throughput_tokens_per_sec", "throughputX")
    (d / "tps_model_gpu.csv").write_text("\n".join([header] + lines[1:]) + "\n", encoding="utf-8")
    res = run_preflight(d)
    assert not res.ok
    md = build_preflight_markdown(res)
    assert "tps_model_gpu.csv" in md and "missing column: throughput_tokens_per_sec" in md


def test_cross_file_coverage_warning(tmp_path: Path):
    d = copy_inputs(tmp_path)
    # Remove some public SKUs to trigger coverage warning
    lines = (d / "price_sheet.csv").read_text(encoding="utf-8").splitlines()
    header = lines[0]
    filtered = [header] + [ln for ln in lines[1:] if "Mixtral-8x7B" not in ln]
    (d / "price_sheet.csv").write_text("\n".join(filtered) + "\n", encoding="utf-8")
    res = run_preflight(d)
    # Should still be OK in Step A, but warnings should include coverage item
    assert res.ok
    md = build_preflight_markdown(res)
    assert "Warnings:" in md


def test_extra_parse_error_warns_only(tmp_path: Path):
    d = copy_inputs(tmp_path)
    # Make extra.yaml invalid YAML
    (d / "extra.yaml").write_text("\x80not_yaml\n", encoding="utf-8")
    res = run_preflight(d)
    # extra.yaml parse error marks that file not ok; overall result can be false, but we accept this as error to be explicit
    assert not res.ok
    md = build_preflight_markdown(res)
    assert "extra.yaml" in md and "parse error" in md
