from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import os

from .shared import FileReport, PreflightResult

from .validate_config import validate as validate_config
from .validate_costs import validate as validate_costs
from .validate_lending import validate as validate_lending
from .validate_gpu_rentals import validate as validate_gpu_rentals
from .validate_oss_models import validate as validate_oss_models
from .validate_price_sheet import validate as validate_price_sheet
from .validate_tps_model_gpu import validate as validate_tps_model_gpu
from .validate_scenarios import validate as validate_scenarios
from .validate_gpu_pricing import validate as validate_gpu_pricing
from .validate_overrides import validate as validate_overrides
from .validate_capacity_overrides import validate as validate_capacity_overrides


def _cross_file_warnings(reports: Dict[str, FileReport]) -> List[str]:
    warns: List[str] = []
    # GPU coverage warning
    try:
        gpus_tps = set(reports["tps_model_gpu.csv"].info.get("gpus", []))
        gpus_rent = set(reports["gpu_rentals.csv"].info.get("gpus", []))
        missing = sorted(g for g in gpus_tps if g not in gpus_rent)
        if missing:
            warns.append(
                f"coverage: GPUs referenced in tps_model_gpu.csv not found in gpu_rentals.csv: {', '.join(missing)} (Step A: warn)"
            )
    except Exception:
        pass
    # Model coverage warning
    try:
        # price_sheet SKUs
        skus = set(reports["price_sheet.csv"].info.get("skus", []))
        # heuristic from oss_models (pseudo skus) when present
        pseudo = set(reports.get("oss_models.csv", FileReport("oss_models.csv", ok=True)).info.get("pseudo_skus", []))
        missing_models = sorted(m for m in pseudo if m not in skus)
        if missing_models:
            warns.append(
                f"coverage: {len(missing_models)} pseudo SKUs from oss_models.csv not found in price_sheet.csv (Step A: warn)"
            )
    except Exception:
        pass
    # Units info
    try:
        units = reports["price_sheet.csv"].info.get("units_public", [])
        if units:
            warns.append(f"units: public_tap unit(s) observed: {', '.join(units)}")
    except Exception:
        pass
    return warns


def run_preflight(inputs_dir: Path | None = None) -> PreflightResult:
    validators = [
        validate_config,
        validate_costs,
        validate_lending,
        validate_gpu_rentals,
        validate_oss_models,
        validate_price_sheet,
        validate_tps_model_gpu,
        validate_scenarios,
        validate_gpu_pricing,
        validate_capacity_overrides,
        validate_overrides,
    ]
    reports: Dict[str, FileReport] = {}
    files: List[FileReport] = []
    ok = True
    for v in validators:
        fr = v(inputs_dir or Path.cwd() / "inputs")
        files.append(fr)
        reports[fr.name] = fr
        if not fr.ok:
            ok = False
    warns = _cross_file_warnings(reports)
    # Strict mode flips
    strict = os.getenv("ENGINE_STRICT_VALIDATION", "0").lower() in {"1", "true", "yes"}
    if strict:
        # Missing unit price warnings → errors for public SKUs
        ps = reports.get("price_sheet.csv")
        if ps is not None:
            carry = []
            for w in ps.warnings:
                if "missing unit_price_eur_per_1k_tokens" in w:
                    ps.ok = False
                    ok = False
                    ps.errors.append(w.replace("(Step A: warn)", "(strict: error)"))
                else:
                    carry.append(w)
            ps.warnings = carry
        # Coverage warnings → errors
        carry_warns: List[str] = []
        for w in warns:
            if "coverage:" in w:
                ok = False
                # Attach to appropriate file if possible
                if "tps_model_gpu.csv" in w:
                    reports["tps_model_gpu.csv"].ok = False
                    reports["tps_model_gpu.csv"].errors.append(w.replace("(Step A: warn)", "(strict: error)"))
                else:
                    # price_sheet coverage
                    reports["price_sheet.csv"].ok = False
                    reports["price_sheet.csv"].errors.append(w.replace("(Step A: warn)", "(strict: error)"))
            else:
                carry_warns.append(w)
        warns = carry_warns
        # Mixed units → error
        units = reports.get("price_sheet.csv", FileReport("price_sheet.csv", ok=True)).info.get("units_public", [])
        if isinstance(units, list) and len(units) > 1:
            reports["price_sheet.csv"].ok = False
            ok = False
            reports["price_sheet.csv"].errors.append("units: mixed public_tap units observed; must choose a single mode (strict: error)")

    return PreflightResult(ok=ok, files=files, warnings=warns)
