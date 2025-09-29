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
    """Read strict curated GPU rentals with schema: gpu,vram_gb,provider,usd_hr"""
    rentals: List[Dict[str, Any]] = []
    try:
        with p.open() as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                provider = (row.get("provider") or "").strip()
                gpu_model = (row.get("gpu") or "").strip()
                vram_s = (row.get("vram_gb") or "").strip()
                usd_s = (row.get("usd_hr") or "").strip()
                try:
                    vram = float(vram_s)
                    usd = float(usd_s)
                except Exception:
                    continue
                if not provider or not gpu_model or not (vram > 0 and usd > 0):
                    continue
                rentals.append({
                    "gpu": gpu_model,
                    "vram_gb": vram,
                    "provider": provider,
                    "usd_hr": usd,
                })
    except FileNotFoundError:
        pass
    return rentals


def _read_tps_model_gpu(p: Path) -> List[Dict[str, Any]]:
    """Read TPS dataset and normalize to canonical keys.

    Canonical keys ensured on each row:
      - model
      - gpu
      - throughput_tokens_per_sec
      - measurement_type (default 'aggregate' if missing)
      - gpu_count (default '1')
      - batch (may be empty)
    """
    rows: List[Dict[str, Any]] = []
    try:
        with p.open() as f:
            rdr = csv.DictReader(f)
            for raw in rdr:
                model = (raw.get("model") or raw.get("Model") or raw.get("model_name") or raw.get("model_id") or "").strip()
                gpu = (raw.get("gpu") or raw.get("gpu_model") or "").strip()
                tps = (raw.get("throughput_tokens_per_sec") or raw.get("tps") or "").strip()
                mt = (raw.get("measurement_type") or "aggregate").strip()
                gc = (raw.get("gpu_count") or "1").strip()
                batch = (raw.get("batch") or "").strip()
                row = dict(raw)
                row.update({
                    "model": model,
                    "gpu": gpu,
                    "throughput_tokens_per_sec": tps,
                    "measurement_type": mt,
                    "gpu_count": gc,
                    "batch": batch,
                })
                rows.append(row)
    except FileNotFoundError:
        pass
    return rows


def load_all(inputs_dir: Path) -> Dict[str, Any]:
    inputs_dir = Path(inputs_dir)
    sim = _read_yaml(inputs_dir / "simulation.yaml")
    gen = _read_yaml(inputs_dir / "operator" / "general.yaml")
    pub = _read_yaml(inputs_dir / "operator" / "public_tap.yaml")
    prv = _read_yaml(inputs_dir / "operator" / "private_tap.yaml")
    facts_market = _read_yaml(inputs_dir / "facts" / "market_env.yaml")
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
    # Optional TPS dataset per (model,gpu)
    tps_csv = inputs_dir / "facts" / "tps_model_gpu.csv"
    tps_yaml = inputs_dir / "facts" / "tps_model_gpu.yaml"
    tps_rows: List[Dict[str, Any]] = _read_tps_model_gpu(tps_csv)
    if tps_rows and tps_yaml.exists():
        print(elog.jsonl("shadowing_warning", dataset="tps_model_gpu", chosen="csv", yaml=str(tps_yaml), csv=str(tps_csv)))
        if sim.get("run", {}).get("fail_on_warning"):
            raise ValidationError("CSV>YAML shadowing escalated to ERROR: tps_model_gpu")

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
