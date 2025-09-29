"""Input loader & overlay (v0).
Reads YAML/CSV from inputs/, prepares normalized state for pipelines.
See `.specs/10_inputs.md` and `.specs/16_simulation_variables.md`.
"""

from pathlib import Path
from typing import Iterator, Dict, Any, List
import csv
import yaml
from . import logging as elog
from .validator import ValidationError


def _read_yaml(p: Path) -> dict:
    try:
        return yaml.safe_load(p.read_text()) or {}
    except FileNotFoundError:
        return {}


def _read_curated_models(p: Path) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    try:
        with p.open() as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                models.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    except FileNotFoundError:
        pass
    return models


def _read_curated_gpu(p: Path) -> List[Dict[str, Any]]:
    """Read curated GPU rentals and normalize to schema: gpu,vram_gb,provider,usd_hr.

    Accepts richer real-world CSVs with headers like:
    - provider
    - gpu_model or gpu
    - gpu_vram_gb or vram_gb
    - price_per_gpu_hr or usd_hr or (price_usd_hr / num_gpus when num_gpus is numeric and >0)
    """
    rentals: List[Dict[str, Any]] = []
    try:
        with p.open() as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                provider = (row.get("provider") or "").strip()
                gpu_model = (row.get("gpu") or row.get("gpu_model") or "").strip()
                vram_s = (row.get("vram_gb") or row.get("gpu_vram_gb") or "").strip()
                # Price derivation priority: price_per_gpu_hr > usd_hr > price_usd_hr/num_gpus
                price_gpu_s = (row.get("price_per_gpu_hr") or row.get("usd_hr") or "").strip()
                if not price_gpu_s:
                    total_s = (row.get("price_usd_hr") or "").strip()
                    num_s = (row.get("num_gpus") or "").strip()
                    try:
                        total = float(total_s)
                        num = float(num_s)
                        price_gpu = total / num if num > 0 else float("nan")
                    except Exception:
                        price_gpu = float("nan")
                else:
                    try:
                        price_gpu = float(price_gpu_s)
                    except Exception:
                        price_gpu = float("nan")
                try:
                    vram = float(vram_s)
                except Exception:
                    vram = float("nan")

                if not provider or not gpu_model:
                    continue
                if not (isinstance(vram, float) and vram > 0):
                    continue
                if not (isinstance(price_gpu, float) and price_gpu > 0):
                    continue
                rentals.append({
                    "gpu": gpu_model,
                    "vram_gb": vram,
                    "provider": provider,
                    "usd_hr": price_gpu,
                })
    except FileNotFoundError:
        pass
    return rentals


def _estimate_model_vram_gb(model_row: Dict[str, Any]) -> float:
    # Prefer explicit weights estimate, else any Typical_VRAM column
    try:
        v = model_row.get("weights_vram_4bit_gb_est")
        if v is not None:
            return float(str(v).strip())
    except Exception:
        pass
    for k, v in model_row.items():
        if str(k).strip().lower().startswith("typical_vram"):
            try:
                return float(str(v).strip())
            except Exception:
                continue
    return 0.0


def _choose_gpu_for_vram(rentals: List[Dict[str, Any]], need_vram: float) -> Dict[str, Any] | None:
    candidates = [r for r in rentals if float(r.get("vram_gb", 0.0)) >= float(need_vram)]
    if not candidates:
        return None
    return min(candidates, key=lambda r: float(r.get("usd_hr", 1e9)))


def _gpu_baseline_tps_from_map(baselines: Dict[str, Any], gpu_name: str) -> float:
    g = (gpu_name or "").lower()
    # baselines is a mapping {pattern_or_gpu_name: tokens_per_sec}
    for k, v in (baselines or {}).items():
        try:
            if str(k).strip().lower() in g:
                val = float(v)
                if val > 0:
                    return val
        except Exception:
            continue
    raise ValidationError(f"Missing GPU TPS baseline for gpu='{gpu_name}' in facts/gpu_baselines.yaml")


def _synthesize_tps_model_gpu(curated_models: List[Dict[str, Any]], rentals: List[Dict[str, Any]], baselines: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for m in curated_models:
        model_name = (m.get("Model") or m.get("model") or "").strip()
        if not model_name:
            continue
        need_vram = _estimate_model_vram_gb(m)
        choice = _choose_gpu_for_vram(rentals, need_vram)
        if not choice:
            continue
        gpu = str(choice.get("gpu"))
        tps = _gpu_baseline_tps_from_map(baselines, gpu)
        rows.append({
            "model": model_name,
            "gpu": gpu,
            "throughput_tokens_per_sec": f"{tps}",
            "measurement_type": "aggregate",
            "gpu_count": "1",
            "batch": "32",
        })
    return rows


def load_all(inputs_dir: Path) -> Dict[str, Any]:
    inputs_dir = Path(inputs_dir)
    sim = _read_yaml(inputs_dir / "simulation.yaml")
    gen = _read_yaml(inputs_dir / "operator" / "general.yaml")
    pub = _read_yaml(inputs_dir / "operator" / "public_tap.yaml")
    prv = _read_yaml(inputs_dir / "operator" / "private_tap.yaml")
    facts_market = _read_yaml(inputs_dir / "facts" / "market_env.yaml")
    facts_gpu_baselines = _read_yaml(inputs_dir / "facts" / "gpu_baselines.yaml")
    curated_models_csv = inputs_dir / "operator" / "curated_public_tap_models.csv"
    curated_models_yaml = inputs_dir / "operator" / "curated_public_tap_models.yaml"
    curated_models = _read_curated_models(curated_models_csv)
    if curated_models_yaml.exists():
        print(elog.jsonl("shadowing_warning", dataset="curated_public_tap_models", chosen="csv", yaml=str(curated_models_yaml), csv=str(curated_models_csv)))
        if sim.get("run", {}).get("fail_on_warning"):
            raise ValidationError("CSV>YAML shadowing escalated to ERROR: curated_public_tap_models")

    curated_gpu_csv = inputs_dir / "operator" / "curated_gpu.csv"
    curated_gpu_yaml = inputs_dir / "operator" / "curated_gpu.yaml"
    curated_gpu = _read_curated_gpu(curated_gpu_csv)
    if curated_gpu_yaml.exists():
        print(elog.jsonl("shadowing_warning", dataset="curated_gpu", chosen="csv", yaml=str(curated_gpu_yaml), csv=str(curated_gpu_csv)))
        if sim.get("run", {}).get("fail_on_warning"):
            raise ValidationError("CSV>YAML shadowing escalated to ERROR: curated_gpu")
    # Synthesize TPS dataset per (model,gpu) from curated models and rentals using input baselines
    if not isinstance(facts_gpu_baselines, dict) or not facts_gpu_baselines:
        raise ValidationError("missing or invalid file: inputs/facts/gpu_baselines.yaml (must be a mapping of GPU names/patterns to tokens/sec)")
    tps_rows: List[Dict[str, Any]] = _synthesize_tps_model_gpu(curated_models, curated_gpu, facts_gpu_baselines)

    # Variables CSVs (optional in v0; read if present)
    variables: Dict[str, List[Dict[str, Any]]] = {}
    for name in ("general", "public_tap", "private_tap"):
        p = inputs_dir / "variables" / f"{name}.csv"
        rows: List[Dict[str, Any]] = []
        try:
            with p.open() as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
        except FileNotFoundError:
            pass
        variables[name] = rows

    return {
        "inputs_dir": str(inputs_dir),
        "simulation": sim,
        "operator": {
            "general": gen,
            "public_tap": pub,
            "private_tap": prv,
        },
        "facts": {
            "market_env": facts_market,
            "gpu_baselines": facts_gpu_baselines,
        },
        "curated": {
            "public_models": curated_models,
            "gpu_rentals": curated_gpu,
            "tps_model_gpu": tps_rows,
        },
        "variables": variables,
    }


def variable_grid(state: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    # v0: single configuration passthrough; expand later per 16_simulation_variables.md
    yield state
