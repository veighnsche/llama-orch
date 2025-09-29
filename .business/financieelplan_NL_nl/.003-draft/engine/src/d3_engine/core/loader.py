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
    rentals: List[Dict[str, Any]] = []
    try:
        with p.open() as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                provider = (row.get("provider") or "").strip()
                gpu_model = (row.get("gpu_model") or row.get("gpu") or "").strip()
                vram_s = (row.get("gpu_vram_gb") or "").strip()
                usd_hr = None
                ppg = row.get("price_per_gpu_hr")
                if ppg is not None and str(ppg).strip() != "":
                    try:
                        usd_hr = float(str(ppg).strip())
                    except Exception:
                        usd_hr = None
                if usd_hr is None:
                    try:
                        total = float((row.get("price_usd_hr") or "").strip())
                        num = float((row.get("num_gpus") or "").strip())
                        usd_hr = total / num if num > 0 else None
                    except Exception:
                        usd_hr = None
                try:
                    vram = float(vram_s) if vram_s != "" else None
                except Exception:
                    vram = None
                if not provider or not gpu_model or usd_hr is None or (vram is None):
                    continue
                rentals.append({
                    "gpu": gpu_model,
                    "vram_gb": vram,
                    "provider": provider,
                    "usd_hr": usd_hr,
                })
    except FileNotFoundError:
        pass
    return rentals


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
    tps_rows: List[Dict[str, Any]] = []
    tps_csv = inputs_dir / "facts" / "tps_model_gpu.csv"
    tps_yaml = inputs_dir / "facts" / "tps_model_gpu.yaml"
    try:
        with tps_csv.open() as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                tps_rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    except FileNotFoundError:
        pass
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
