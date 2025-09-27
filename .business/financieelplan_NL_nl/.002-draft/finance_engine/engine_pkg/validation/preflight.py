from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import math
import re
import pandas as pd

from ...config import INPUTS
from ...io.loader import load_yaml, read_csv


@dataclass
class FileReport:
    name: str
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    count: int | None = None  # rows for CSVs, keys for YAML
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightResult:
    ok: bool
    files: List[FileReport]
    warnings: List[str] = field(default_factory=list)


# ---- Helpers ----

def _require_keys(obj: Dict[str, Any], path: List[str]) -> Tuple[bool, Any]:
    cur: Any = obj
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return False, None
        cur = cur[p]
    return True, cur


def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _non_empty_string(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


# ---- Validators per file ----

def validate_config(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "config.yaml"
    fr = FileReport(name="config.yaml", ok=True)
    try:
        obj = load_yaml(p)
        if not isinstance(obj, dict):
            fr.ok = False
            fr.errors.append("Top-level structure must be a mapping")
            return fr
        # currency
        if not _non_empty_string(obj.get("currency")):
            fr.ok = False
            fr.errors.append("currency: required non-empty string")
        # finance.marketing_allocation_pct_of_inflow (0-100)
        ok, v = _require_keys(obj, ["finance", "marketing_allocation_pct_of_inflow"])
        if ok and v is not None:
            if not _is_number(v) or not (0 <= float(v) <= 100):
                fr.ok = False
                fr.errors.append("finance.marketing_allocation_pct_of_inflow: must be numeric 0–100 (percent)")
        # tax_billing.vat_standard_rate_pct (0-100)
        ok, v = _require_keys(obj, ["tax_billing", "vat_standard_rate_pct"])
        if ok and v is not None:
            if not _is_number(v) or not (0 <= float(v) <= 100):
                fr.ok = False
                fr.errors.append("tax_billing.vat_standard_rate_pct: must be numeric 0–100 (percent)")
        # legal_policy.refunds.allowed (bool) if present
        ok, v = _require_keys(obj, ["legal_policy", "refunds", "allowed"])
        if ok and v is not None and not isinstance(v, bool):
            fr.ok = False
            fr.errors.append("legal_policy.refunds.allowed: must be boolean if present")
        fr.count = len(obj)
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr


def _walk_numeric_sum(d: Any, errs: List[str], path: str = "") -> float:
    total = 0.0
    if isinstance(d, dict):
        for k, v in d.items():
            if not _non_empty_string(k):
                errs.append(f"empty key at {path}")
            total += _walk_numeric_sum(v, errs, f"{path}.{k}" if path else str(k))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            total += _walk_numeric_sum(v, errs, f"{path}[{i}]")
    else:
        if _is_number(d):
            val = float(d)
            if val < 0:
                errs.append(f"negative value at {path}")
            else:
                total += val
    return total


def validate_costs(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "costs.yaml"
    fr = FileReport(name="costs.yaml", ok=True)
    try:
        obj = load_yaml(p)
        if not isinstance(obj, dict):
            fr.ok = False
            fr.errors.append("Top-level structure must be a mapping")
            return fr
        errs: List[str] = []
        total = _walk_numeric_sum(obj, errs)
        if errs:
            fr.ok = False
            fr.errors.extend(sorted(set(errs)))
        if total == 0.0:
            # If nothing numeric found, treat as error in Step A
            fr.ok = False
            fr.errors.append("no numeric amounts found to compute a sum")
        fr.count = len(obj)
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr


def validate_lending(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "lending_plan.yaml"
    fr = FileReport(name="lending_plan.yaml", ok=True)
    try:
        obj = load_yaml(p)
        if not isinstance(obj, dict):
            fr.ok = False
            fr.errors.append("Top-level structure must be a mapping")
            return fr
        ok_amt, amount = _require_keys(obj, ["loan_request", "amount_eur"])
        ok_term, term = _require_keys(obj, ["repayment_plan", "term_months"])
        ok_rate, rate = _require_keys(obj, ["repayment_plan", "interest_rate_pct"])
        if not ok_amt or not _is_number(amount) or float(amount) <= 0:
            fr.ok = False
            fr.errors.append("loan_request.amount_eur: required > 0")
        if not ok_term or not _is_number(term) or float(term) <= 0:
            fr.ok = False
            fr.errors.append("repayment_plan.term_months: required > 0")
        if not ok_rate or not _is_number(rate) or float(rate) < 0:
            fr.ok = False
            fr.errors.append("repayment_plan.interest_rate_pct: required ≥ 0")
        fr.count = len(obj)
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr


def validate_gpu_rentals(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "gpu_rentals.csv"
    fr = FileReport(name="gpu_rentals.csv", ok=True)
    try:
        df = read_csv(p)
        req = ["gpu", "vram_gb", "hourly_usd_min", "hourly_usd_max"]
        for c in req:
            if c not in df.columns:
                fr.ok = False
                fr.errors.append(f"missing column: {c}")
        if not fr.ok:
            return fr
        if df["gpu"].isna().any() or (df["gpu"].astype(str).str.strip() == "").any():
            fr.ok = False
            fr.errors.append("gpu column has empty values")
        # numeric checks
        for c in ["vram_gb", "hourly_usd_min", "hourly_usd_max"]:
            bad = df[c].apply(lambda x: not _is_number(x) or float(x) < 0)
            if bad.any():
                fr.ok = False
                fr.errors.append(f"{c}: non-numeric or negative values present")
        # duplicates
        dup = df["gpu"].astype(str).duplicated()
        if dup.any():
            fr.ok = False
            fr.errors.append("duplicate gpu identifiers present")
        fr.count = len(df)
        fr.info["gpus"] = sorted(set(df["gpu"].astype(str)))
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr


def validate_oss_models(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "oss_models.csv"
    fr = FileReport(name="oss_models.csv", ok=True)
    try:
        df = pd.read_csv(p)
        if "name" not in df.columns:
            fr.ok = False
            fr.errors.append("missing column: name")
            return fr
        if df["name"].isna().any() or (df["name"].astype(str).str.strip() == "").any():
            fr.ok = False
            fr.errors.append("name column has empty values")
        # Any spec column present?
        if not any(c in df.columns for c in ("variant_size_b", "context_tokens", "license")):
            fr.ok = False
            fr.errors.append("no spec columns present (variant_size_b/context_tokens/license)")
        fr.count = len(df)
        # generate pseudo SKUs for coverage heuristics
        if "variant_size_b" in df.columns:
            def mk_sku(row: pd.Series) -> str:
                try:
                    size = row.get("variant_size_b")
                    if _is_number(size):
                        if float(size).is_integer():
                            size_str = f"{int(float(size))}B"
                        else:
                            size_str = f"{float(size)}B"
                    else:
                        size_str = str(size)
                    return f"{str(row.get('name')).strip().replace(' ', '-')}".replace(".", "-") + f"-{size_str}"
                except Exception:
                    return str(row.get("name")).strip()
            fr.info["pseudo_skus"] = sorted(set(df.apply(mk_sku, axis=1)))
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr


def validate_price_sheet(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "price_sheet.csv"
    fr = FileReport(name="price_sheet.csv", ok=True)
    try:
        df = pd.read_csv(p)
        req = ["sku", "category", "unit"]
        for c in req:
            if c not in df.columns:
                fr.ok = False
                fr.errors.append(f"missing column: {c}")
        if not fr.ok:
            return fr
        # For public tap rows, unit must be explicit
        mask_pub = df["category"].astype(str) == "public_tap"
        units = sorted(set(df.loc[mask_pub, "unit"].dropna().astype(str)))
        if any(u not in ("1k_tokens", "1M_tokens") for u in units):
            fr.ok = False
            fr.errors.append("unit must be one of {1k_tokens, 1M_tokens} for public_tap rows")
        # Warn on missing unit price for Step A only
        if "unit_price_eur_per_1k_tokens" in df.columns:
            missing_prices = int(df.loc[mask_pub, "unit_price_eur_per_1k_tokens"].isna().sum())
            if missing_prices > 0:
                fr.warnings.append(f"{missing_prices} public_tap SKU rows have missing unit_price_eur_per_1k_tokens (Step A: warn)")
        fr.count = len(df)
        fr.info["units_public"] = units
        fr.info["skus"] = sorted(set(df["sku"].astype(str)))
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr


def validate_tps_model_gpu(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "tps_model_gpu.csv"
    fr = FileReport(name="tps_model_gpu.csv", ok=True)
    try:
        df = pd.read_csv(p)
        req = ["model_name", "gpu", "throughput_tokens_per_sec"]
        for c in req:
            if c not in df.columns:
                fr.ok = False
                fr.errors.append(f"missing column: {c}")
        if not fr.ok:
            return fr
        if df["model_name"].isna().any() or (df["model_name"].astype(str).str.strip() == "").any():
            fr.ok = False
            fr.errors.append("model_name column has empty values")
        if df["gpu"].isna().any() or (df["gpu"].astype(str).str.strip() == "").any():
            fr.ok = False
            fr.errors.append("gpu column has empty values")
        bad = df["throughput_tokens_per_sec"].apply(lambda x: not _is_number(x) or float(x) < 0)
        if bad.any():
            fr.ok = False
            fr.errors.append("throughput_tokens_per_sec: non-numeric or negative values present")
        fr.count = len(df)
        fr.info["gpus"] = sorted(set(df["gpu"].astype(str)))
        # Normalized model name heuristic for coverage warnings
        def norm_model(m: str) -> str:
            m = re.sub(r"Instruct", "", str(m))
            m = re.sub(r"\s+", "-", m.strip())
            m = m.replace(".", "-")
            return m
        fr.info["models_norm"] = sorted(set(norm_model(m) for m in df["model_name"].astype(str)))
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr


def validate_extra_readonly(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "extra.yaml"
    fr = FileReport(name="extra.yaml", ok=True)
    if not p.exists():
        fr.warnings.append("extra.yaml not present (Step A: optional)")
        return fr
    try:
        _ = load_yaml(p)
        fr.count = None
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr


# ---- Cross-file checks (warnings in Step A) ----

def cross_file_warnings(reports: Dict[str, FileReport]) -> List[str]:
    warns: List[str] = []
    # GPU coverage
    try:
        gpus_tps = set(reports["tps_model_gpu.csv"].info.get("gpus", []))
        gpus_rent = set(reports["gpu_rentals.csv"].info.get("gpus", []))
        missing = sorted(g for g in gpus_tps if g not in gpus_rent)
        if missing:
            warns.append(f"coverage: GPUs referenced in tps_model_gpu.csv not found in gpu_rentals.csv: {', '.join(missing)} (Step A: warn)")
    except Exception:
        pass
    # Model coverage (heuristic)
    try:
        pseudo_skus = set(reports["oss_models.csv"].info.get("pseudo_skus", []))
        skus = set(reports["price_sheet.csv"].info.get("skus", []))
        missing_models = sorted(m for m in pseudo_skus if m not in skus)
        if missing_models:
            warns.append(f"coverage: {len(missing_models)} pseudo SKUs from oss_models.csv not found in price_sheet.csv (Step A: warn)")
    except Exception:
        pass
    # Units info (informational)
    try:
        units = reports["price_sheet.csv"].info.get("units_public", [])
        if units:
            warns.append(f"units: public_tap unit(s) observed: {', '.join(units)}")
    except Exception:
        pass
    return warns


# ---- Orchestration ----

def run_preflight(inputs_dir: Path | None = None) -> PreflightResult:
    inputs_dir = inputs_dir or INPUTS
    validators = [
        validate_config,
        validate_costs,
        validate_lending,
        validate_gpu_rentals,
        validate_oss_models,
        validate_price_sheet,
        validate_tps_model_gpu,
        validate_extra_readonly,
    ]
    reports: Dict[str, FileReport] = {}
    ok = True
    files: List[FileReport] = []
    for v in validators:
        fr = v(inputs_dir)
        files.append(fr)
        reports[fr.name] = fr
        if not fr.ok:
            ok = False
    warns = cross_file_warnings(reports)
    return PreflightResult(ok=ok, files=files, warnings=warns)


def build_preflight_markdown(res: PreflightResult) -> str:
    lines: List[str] = []
    lines.append("## Preflight")
    status = "OK (validation)" if res.ok else "FAILED (validation)"
    lines.append(f"STATUS: {status}")
    lines.append("")
    lines.append("Files:")
    for fr in res.files:
        if fr.count is not None:
            lines.append(f"- {fr.name}: {'\u2713' if fr.ok else '✗'}{'' if fr.count is None else f' ({fr.count} rows/keys)'}")
        else:
            lines.append(f"- {fr.name}: {'\u2713' if fr.ok else '✗'}")
    # Warnings
    if res.warnings or any(fr.warnings for fr in res.files):
        lines.append("")
        lines.append("Warnings:")
        for fr in res.files:
            for w in fr.warnings:
                lines.append(f"- {fr.name}: {w}")
        for w in res.warnings:
            lines.append(f"- {w}")
    # Errors (only if failed)
    if not res.ok:
        lines.append("")
        lines.append("Errors:")
        for fr in res.files:
            for e in fr.errors:
                lines.append(f"- {fr.name}: {e}")
        lines.append("")
        lines.append("How to fix:")
        # Provide simple hints by file
        hints = {
            "gpu_rentals.csv": "Add missing column(s) gpu, vram_gb, hourly_usd_min, hourly_usd_max and ensure non-negative numbers.",
            "price_sheet.csv": "Ensure columns sku, category, unit exist; for public_tap, unit must be 1k_tokens or 1M_tokens.",
            "tps_model_gpu.csv": "Ensure columns model_name, gpu, throughput_tokens_per_sec exist and throughput is non-negative.",
            "oss_models.csv": "Ensure name column is present and non-empty; include at least one spec column.",
            "config.yaml": "Ensure currency is non-empty; percentages are 0-100; booleans are booleans.",
            "lending_plan.yaml": "Provide loan_request.amount_eur > 0; repayment_plan.term_months > 0; interest_rate_pct ≥ 0.",
            "costs.yaml": "Provide one or more numeric amounts ≥ 0 so a total can be computed.",
        }
        for fname, hint in hints.items():
            lines.append(f"- {fname}: {hint}")
    return "\n".join(lines) + "\n"
