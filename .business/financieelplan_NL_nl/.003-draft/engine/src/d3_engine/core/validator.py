"""Validator (scaffold).
Schema/type/domain/reference checks per .specs/10_inputs.md and friends.
"""
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import csv
import yaml


class ValidationError(Exception):
    pass


def _require(cond: bool, msg: str, errors: List[str]):
    if not cond:
        errors.append(msg)


def _require_key(d: dict, path: str, errors: List[str], typ=None, pred=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            errors.append(f"missing key: {path}")
            return None
        cur = cur[part]
    if typ is not None and not isinstance(cur, typ):
        errors.append(f"invalid type for {path}: expected {typ.__name__}")
    if pred is not None and isinstance(cur, (int, float)) and not pred(cur):
        errors.append(f"invalid value for {path}: {cur}")
    return cur


def _read_yaml(p: Path) -> dict:
    try:
        return yaml.safe_load(p.read_text()) or {}
    except FileNotFoundError:
        raise ValidationError(f"missing file: {p}")


def _check_simulation_yaml(inputs_dir: Path, errors: List[str]):
    sim_p = inputs_dir / "simulation.yaml"
    data = _read_yaml(sim_p)
    # Required keys (see 14_simulation_parameters.md)
    _require_key(data, "run.pipelines", errors, list)
    rs = _require_key(data, "run.random_seed", errors, int, lambda v: v > 0)
    _require_key(data, "run.output_dir", errors)
    _require_key(data, "run.random_runs_per_simulation", errors, int, lambda v: v >= 1)
    _require_key(data, "stochastic.simulations_per_run", errors, int, lambda v: v >= 1)
    perc = _require_key(data, "stochastic.percentiles", errors, list)
    if isinstance(perc, list):
        for x in perc:
            if not isinstance(x, int) or not (1 <= x <= 99):
                errors.append("stochastic.percentiles must be ints in [1..99]")
                break
    _require_key(data, "targets.horizon_months", errors, int, lambda v: v >= 1)
    _require_key(data, "targets.private_margin_threshold_pct", errors, (int, float), lambda v: v >= 0)
    _require_key(data, "targets.require_monotonic_growth_public_active_customers", errors, bool)
    _require_key(data, "targets.require_monotonic_growth_private_active_customers", errors, bool)
    _require_key(data, "targets.autoscaling_util_tolerance_pct", errors, (int, float), lambda v: v >= 0)


def _check_public_operator(inputs_dir: Path, errors: List[str]):
    op_p = inputs_dir / "operator" / "public_tap.yaml"
    data = _read_yaml(op_p)
    _require_key(data, "pricing_policy.target_margin_pct", errors, (int, float), lambda v: 0 <= v <= 95)
    _require_key(data, "autoscaling.target_utilization_pct", errors, (int, float), lambda v: 1 <= v <= 100)
    _require_key(data, "autoscaling.peak_factor", errors, (int, float), lambda v: v >= 1.0)
    min_i = _require_key(data, "autoscaling.min_instances_per_model", errors, int, lambda v: v >= 0)
    max_i = _require_key(data, "autoscaling.max_instances_per_model", errors, int, lambda v: v >= 1)
    if isinstance(min_i, int) and isinstance(max_i, int) and min_i > max_i:
        errors.append("autoscaling.min_instances_per_model must be <= max_instances_per_model")
    # Simulator policy (all required)
    _require_key(data, "autoscaling.evaluation_interval_s", errors, int, lambda v: v > 0)
    up = _require_key(data, "autoscaling.scale_up_threshold_pct", errors, (int, float))
    down = _require_key(data, "autoscaling.scale_down_threshold_pct", errors, (int, float))
    if isinstance(up, (int, float)) and isinstance(down, (int, float)):
        if not (0 < down < up <= 100):
            errors.append("autoscaling thresholds must satisfy 0 < scale_down < scale_up <= 100")
    _require_key(data, "autoscaling.scale_up_step_replicas", errors, int, lambda v: v >= 1)
    _require_key(data, "autoscaling.scale_down_step_replicas", errors, int, lambda v: v >= 1)
    _require_key(data, "autoscaling.stabilization_window_s", errors, int, lambda v: v >= 0)
    _require_key(data, "autoscaling.warmup_s", errors, int, lambda v: v >= 0)
    _require_key(data, "autoscaling.cooldown_s", errors, int, lambda v: v >= 0)


def _check_private_operator(inputs_dir: Path, errors: List[str]):
    op_p = inputs_dir / "operator" / "private_tap.yaml"
    data = _read_yaml(op_p)
    _require_key(data, "pricing_policy.default_markup_over_provider_cost_pct", errors, (int, float), lambda v: v >= 0)


def _check_curated(inputs_dir: Path, errors: List[str]):
    gpu_p = inputs_dir / "operator" / "curated_gpu.csv"
    try:
        with gpu_p.open() as f:
            rdr = csv.DictReader(f)
            hdr = [h.strip() for h in rdr.fieldnames or []]
            # Accept user's richer schema; require subset columns
            required = {"provider", "gpu_vram_gb", "price_per_gpu_hr"}
            if not required.issubset({h for h in (h.strip() for h in hdr)}):
                errors.append(
                    "curated_gpu.csv must contain columns: provider,gpu_vram_gb,price_per_gpu_hr"
                )
            for row in rdr:
                try:
                    usd = float((row.get("price_per_gpu_hr") or "").strip())
                    vram = float((row.get("gpu_vram_gb") or "").strip())
                except Exception:
                    errors.append("curated_gpu.csv invalid numeric values in price_per_gpu_hr/gpu_vram_gb")
                    break
                if usd <= 0:
                    errors.append("curated_gpu.csv price_per_gpu_hr must be > 0")
                    break
                if vram <= 0:
                    errors.append("curated_gpu.csv gpu_vram_gb must be > 0")
                    break
    except FileNotFoundError:
        errors.append(f"missing file: {gpu_p}")

    models_p = inputs_dir / "operator" / "curated_public_tap_models.csv"
    try:
        with models_p.open() as f:
            rdr = csv.reader(f)
            hdr = next(rdr)
            hdr_lc = [h.strip().lower() for h in hdr]
            if "model" not in hdr_lc:
                errors.append("curated_public_tap_models.csv must contain a 'model' column (case-insensitive)")
            else:
                model_idx = hdr_lc.index("model")
                rows = 0
                for row in rdr:
                    rows += 1
                    if model_idx >= len(row) or not str(row[model_idx]).strip():
                        errors.append("curated_public_tap_models.csv contains empty 'model' value")
                        break
                if rows < 1:
                    errors.append("curated_public_tap_models.csv must have at least one row")
    except FileNotFoundError:
        errors.append(f"missing file: {models_p}")


def _check_variables(inputs_dir: Path, errors: List[str]):
    vars_dir = inputs_dir / "variables"
    if not vars_dir.exists():
        errors.append(f"missing directory: {vars_dir}")
        return
    required_headers = [
        "variable_id","scope","path","type","unit","min","max","step","default","treatment","notes"
    ]
    allowed_scopes = {"general", "public_tap", "private_tap"}
    for csv_name in ("general.csv", "public_tap.csv", "private_tap.csv"):
        p = vars_dir / csv_name
        if not p.exists():
            # only warn if missing? For now, require existence for the three known files
            errors.append(f"missing file: {p}")
            continue
        with p.open() as f:
            rdr = csv.DictReader(f)
            hdr = rdr.fieldnames or []
            if [h.strip() for h in hdr] != required_headers:
                errors.append(f"{csv_name} headers must exactly match {required_headers}")
                continue
            for row in rdr:
                scope = (row.get("scope") or "").strip()
                if scope not in allowed_scopes:
                    errors.append(f"{csv_name} invalid scope: {scope}")
                    break
                typ = (row.get("type") or "").strip()
                if typ not in ("numeric", "discrete"):
                    errors.append(f"{csv_name} invalid type: {typ}")
                    break
                if typ == "numeric":
                    try:
                        mn = float(row.get("min") or "")
                        mx = float(row.get("max") or "")
                        st = float(row.get("step") or "")
                        df = float(row.get("default") or "")
                    except Exception:
                        errors.append(f"{csv_name} numeric row has non-numeric min/max/step/default")
                        break
                    if st <= 0:
                        errors.append(f"{csv_name} step must be > 0")
                        break
                    if mn > mx:
                        errors.append(f"{csv_name} min must be <= max")
                        break
                    if not (mn <= df <= mx):
                        errors.append(f"{csv_name} default must be within [min,max]")
                        break


def _check_facts(inputs_dir: Path, errors: List[str]):
    facts_p = inputs_dir / "facts" / "market_env.yaml"
    data = _read_yaml(facts_p)
    v = (
        data.get("finance", {})
        .get("eur_usd_fx_rate", {})
        .get("value")
    )
    if v is None or not isinstance(v, (int, float)) or v <= 0:
        errors.append("facts.market_env.finance.eur_usd_fx_rate.value must be > 0")


def validate(state: Dict[str, Any]) -> None:
    """Fast-fail validator per 14_simulation_parameters.md.

    This reads required files directly under inputs_dir and checks presence and constraints.
    """
    inputs_dir_s = state.get("inputs_dir")
    if not inputs_dir_s:
        raise ValidationError("missing inputs_dir in state")
    inputs_dir = Path(inputs_dir_s)
    errors: List[str] = []

    _check_simulation_yaml(inputs_dir, errors)
    _check_public_operator(inputs_dir, errors)
    _check_private_operator(inputs_dir, errors)
    _check_curated(inputs_dir, errors)
    _check_variables(inputs_dir, errors)
    _check_facts(inputs_dir, errors)

    if errors:
        raise ValidationError("; ".join(errors))
