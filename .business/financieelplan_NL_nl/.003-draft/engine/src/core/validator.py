"""Validator (scaffold).
Schema/type/domain/reference checks per .specs/10_inputs.md and friends.
"""
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import csv
import yaml
from . import logging as elog
import math


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
    # run.random_seed is optional; if present must be int > 0
    cur = data.get("run", {}) if isinstance(data, dict) else {}
    if isinstance(cur, dict) and "random_seed" in cur:
        v = cur.get("random_seed")
        if not isinstance(v, int) or v <= 0:
            errors.append("invalid value for run.random_seed: must be int > 0 when set")
    _require_key(data, "run.output_dir", errors)
    _require_key(data, "run.random_runs_per_simulation", errors, int, lambda v: v >= 1)
    _require_key(data, "stochastic.simulations_per_run", errors, int, lambda v: v >= 1)
    perc = _require_key(data, "stochastic.percentiles", errors, list)
    if isinstance(perc, list):
        ok_types = all(isinstance(x, int) for x in perc)
        ok_range = all(0 <= int(x) <= 100 for x in perc) if ok_types else False
        ok_asc = all(perc[i] < perc[i+1] for i in range(len(perc)-1)) if len(perc) > 1 and ok_types else True
        if not (ok_types and ok_range and ok_asc):
            errors.append("stochastic.percentiles must be strictly ascending ints in [0..100]")
    _require_key(data, "targets.horizon_months", errors, int, lambda v: v >= 1)
    _require_key(data, "targets.private_margin_threshold_pct", errors, (int, float), lambda v: v >= 0)
    _require_key(data, "targets.require_monotonic_growth_public_active_customers", errors, bool)
    _require_key(data, "targets.require_monotonic_growth_private_active_customers", errors, bool)
    _require_key(data, "targets.autoscaling_util_tolerance_pct", errors, (int, float), lambda v: 0 <= v <= 100)


def _check_public_operator(inputs_dir: Path, errors: List[str]):
    op_p = inputs_dir / "operator" / "public_tap.yaml"
    data = _read_yaml(op_p)
    _require_key(data, "pricing_policy.public_tap.target_margin_pct", errors, (int, float), lambda v: 0 <= v <= 95)
    _require_key(data, "autoscaling.target_utilization_pct", errors, (int, float), lambda v: 1 <= v <= 100)
    _require_key(data, "autoscaling.peak_factor", errors, (int, float), lambda v: v >= 1.0)
    min_i = _require_key(data, "autoscaling.min_instances_per_model", errors, int, lambda v: v >= 0)
    max_i = _require_key(data, "autoscaling.max_instances_per_model", errors, int, lambda v: v >= 1)
    if isinstance(min_i, int) and isinstance(max_i, int) and min_i > max_i:
        errors.append("autoscaling.min_instances_per_model must be <= max_instances_per_model")
    # Simulator policy is SHOULD: do not error if missing; validate only when present
    def _opt_num(path: str, typ, pred=None):
        cur = data
        for part in path.split('.'):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        if not isinstance(cur, typ):
            errors.append(f"invalid type for {path}: expected {typ.__name__}")
            return None
        if pred is not None and isinstance(cur, (int, float)) and not pred(cur):
            errors.append(f"invalid value for {path}: {cur}")
        return cur
    _opt_num("autoscaling.evaluation_interval_s", int, lambda v: v > 0)
    up = _opt_num("autoscaling.scale_up_threshold_pct", (int, float))
    down = _opt_num("autoscaling.scale_down_threshold_pct", (int, float))
    if isinstance(up, (int, float)) and isinstance(down, (int, float)):
        if not (0 < down < up <= 100):
            errors.append("autoscaling thresholds must satisfy 0 < scale_down < scale_up <= 100")
    _opt_num("autoscaling.scale_up_step_replicas", int, lambda v: v >= 1)
    _opt_num("autoscaling.scale_down_step_replicas", int, lambda v: v >= 1)
    _opt_num("autoscaling.stabilization_window_s", int, lambda v: v >= 0)
    _opt_num("autoscaling.warmup_s", int, lambda v: v >= 0)
    _opt_num("autoscaling.cooldown_s", int, lambda v: v >= 0)


def _check_private_operator(inputs_dir: Path, errors: List[str]):
    op_p = inputs_dir / "operator" / "private_tap.yaml"
    data = _read_yaml(op_p)
    # Align with nested policy under pricing_policy.private_tap.* (mirrors public_tap structure)
    _require_key(
        data,
        "pricing_policy.private_tap.default_markup_over_provider_cost_pct",
        errors,
        (int, float),
        lambda v: v >= 0,
    )


def _check_curated(inputs_dir: Path, errors: List[str], warn_to_error: bool = False):
    # Enforce single strict schema for curated GPU rentals: [gpu, vram_gb, provider, usd_hr]
    gpu_p = inputs_dir / "operator" / "curated_gpu.csv"
    try:
        with gpu_p.open() as f:
            rdr = csv.DictReader(f)
            hdr = [h.strip() for h in rdr.fieldnames or []]
            have = {h for h in (h.strip() for h in hdr)}
            has_provider = "provider" in have
            has_gpu = ("gpu" in have) or ("gpu_model" in have)
            has_vram = ("vram_gb" in have) or ("gpu_vram_gb" in have)
            has_price = ("usd_hr" in have) or ("price_per_gpu_hr" in have) or (("price_usd_hr" in have) and ("num_gpus" in have))
            if not (has_provider and has_gpu and has_vram and has_price):
                errors.append("curated_gpu.csv must include provider and GPU model/VRAM columns and a per-GPU USD/hr price (accepts headers: gpu|gpu_model, vram_gb|gpu_vram_gb, usd_hr|price_per_gpu_hr|price_usd_hr+num_gpus)")
            for row in rdr:
                # Derive canonical numeric fields like the loader
                try:
                    vram_s = (row.get("vram_gb") or row.get("gpu_vram_gb") or "").strip()
                    vram = float(vram_s)
                except Exception:
                    errors.append("curated_gpu.csv invalid numeric VRAM (vram_gb/gpu_vram_gb)")
                    break
                # Price per GPU
                price_s = (row.get("usd_hr") or row.get("price_per_gpu_hr") or "").strip()
                if not price_s:
                    try:
                        total = float((row.get("price_usd_hr") or "").strip())
                        num = float((row.get("num_gpus") or "").strip())
                        usd = total / num if num > 0 else float("nan")
                    except Exception:
                        usd = float("nan")
                else:
                    try:
                        usd = float(price_s)
                    except Exception:
                        usd = float("nan")
                if not (math.isfinite(vram) and vram > 0):
                    errors.append("curated_gpu.csv vram_gb must be a finite number > 0")
                    break
                if not (math.isfinite(usd) and usd > 0):
                    errors.append("curated_gpu.csv usd_hr must be a finite number > 0")
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
            # Require a VRAM estimate column. Prefer numeric 'weights_vram_4bit_gb_est' if present,
            # otherwise fall back to any header containing 'typical_vram'.
            vram_idx: int | None = None
            if "weights_vram_4bit_gb_est" in hdr_lc:
                vram_idx = hdr_lc.index("weights_vram_4bit_gb_est")
            else:
                for i, h in enumerate(hdr_lc):
                    if "typical_vram" in h:
                        vram_idx = i
                        break
            if vram_idx is None:
                errors.append("curated_public_tap_models.csv must include a VRAM estimate column (e.g., 'Typical_VRAM_*' or 'weights_vram_4bit_gb_est')")
            rows = 0
            for row in rdr:
                rows += 1
                if "model" in hdr_lc:
                    model_idx = hdr_lc.index("model")
                    if model_idx >= len(row) or not str(row[model_idx]).strip():
                        errors.append("curated_public_tap_models.csv contains empty 'model' value")
                        break
                if vram_idx is not None:
                    try:
                        vram_val = float(str(row[vram_idx]).strip())
                        if vram_val <= 0:
                            errors.append("curated_public_tap_models.csv VRAM estimate must be > 0 for all rows")
                            break
                    except Exception:
                        errors.append("curated_public_tap_models.csv VRAM estimate must be numeric for all rows")
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
    # Build allowed path roots dynamically from operator YAMLs (up to two levels)
    def _prefixes_from_yaml(d: dict) -> List[str]:
        prefs: List[str] = []
        if not isinstance(d, dict):
            return prefs
        for k, v in d.items():
            if isinstance(v, dict):
                prefs.append(f"{k}.")
                for k2, v2 in v.items():
                    if isinstance(v2, dict):
                        prefs.append(f"{k}.{k2}.")
        return prefs

    gen_data = _read_yaml(inputs_dir / "operator" / "general.yaml")
    pub_data = _read_yaml(inputs_dir / "operator" / "public_tap.yaml")
    prv_data = _read_yaml(inputs_dir / "operator" / "private_tap.yaml")
    dynamic_roots = {
        "general": _prefixes_from_yaml(gen_data),
        "public_tap": _prefixes_from_yaml(pub_data),
        "private_tap": _prefixes_from_yaml(prv_data),
    }

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
                # Skip completely empty rows (robustness against trailing newlines)
                if not any((v or "").strip() for v in row.values()):
                    continue
                # Skip comment rows: when the first non-empty field starts with '#'
                vals = [(v or "").strip() for v in row.values()]
                first_nonempty = next((v for v in vals if v), "")
                if first_nonempty.startswith("#"):
                    continue
                scope = (row.get("scope") or "").strip()
                if scope not in allowed_scopes:
                    errors.append(f"{csv_name} invalid scope: {scope}")
                    break
                typ = (row.get("type") or "").strip()
                if typ not in ("numeric", "discrete"):
                    errors.append(f"{csv_name} invalid type: {typ}")
                    break
                # Enforce treatment semantics (support extended distributions)
                treatment = (row.get("treatment") or "").strip()
                allowed_treatments = {
                    "fixed",
                    "low_to_high",
                    "random",
                    "random_uniform",
                    "random_normal",
                    "random_lognormal",
                    "random_beta",
                }
                if treatment not in allowed_treatments:
                    errors.append(f"{csv_name} invalid treatment: {treatment}")
                    break
                # Allowed path roots per scope: derived from operator YAMLs (two-level prefixes)
                path = (row.get("path") or "").strip()
                roots = dynamic_roots.get(scope, [])
                if path and not any(path.startswith(r) for r in roots):
                    errors.append(f"{csv_name} path not allowed for scope {scope}: {path}")
                    break
                # Policy note: treatment semantics (controllable vs uncontrollable) are
                # explicitly declared in the CSV by authors. We do not hard-code lists
                # in validator to avoid maintenance burden. Optional linting can be added
                # via sidecar policy files (see _check_sidecar_schemas) without code edits.
                if typ == "numeric":
                    try:
                        mn = float(row.get("min") or "")
                        mx = float(row.get("max") or "")
                        st = float(row.get("step") or "")
                        df = float(row.get("default") or "")
                        if not all(math.isfinite(x) for x in (mn, mx, st, df)):
                            raise ValueError("non-finite numeric fields")
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

    # GPU baseline TPS mapping used to synthesize missing TPS facts
    gpu_base_p = inputs_dir / "facts" / "gpu_baselines.yaml"
    try:
        base = _read_yaml(gpu_base_p)
    except ValidationError as e:
        errors.append(str(e))
        return
    if not isinstance(base, dict) or not base:
        errors.append(f"{gpu_base_p} must be a non-empty mapping of GPU names/patterns to tokens/sec")
        return
    # Validate values are positive numbers
    for k, val in base.items():
        try:
            x = float(val)
            if not (math.isfinite(x) and x > 0):
                errors.append(f"gpu_baselines.yaml value for '{k}' must be a finite number > 0")
                break
        except Exception:
            errors.append(f"gpu_baselines.yaml value for '{k}' must be numeric")
            break


def validate(state: Dict[str, Any]) -> None:
    """Fast-fail validator per 14_simulation_parameters.md.

    This reads required files directly under inputs_dir and checks presence and constraints.
    """
    inputs_dir_s = state.get("inputs_dir")
    if not inputs_dir_s:
        raise ValidationError("missing inputs_dir in state")
    inputs_dir = Path(inputs_dir_s)
    errors: List[str] = []

    # Escalation policy for warnings
    warn_to_error = False
    try:
        warn_to_error = bool(state.get("simulation", {}).get("run", {}).get("fail_on_warning", False))
    except Exception:
        warn_to_error = False

    _check_simulation_yaml(inputs_dir, errors)
    _check_public_operator(inputs_dir, errors)
    _check_private_operator(inputs_dir, errors)
    _check_curated(inputs_dir, errors, warn_to_error=warn_to_error)
    _check_variables(inputs_dir, errors)
    _check_facts(inputs_dir, errors)

    if errors:
        raise ValidationError("; ".join(errors))
