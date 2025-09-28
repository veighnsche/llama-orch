"""Input loader & overlay (v0).
Reads YAML/CSV from inputs/, prepares normalized state for pipelines.
See `.specs/10_inputs.md` and `.specs/16_simulation_variables.md`.
"""

from pathlib import Path
from typing import Iterator, Dict, Any, List
import csv
import yaml


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
    pub = _read_yaml(inputs_dir / "operator" / "public_tap.yaml")
    prv = _read_yaml(inputs_dir / "operator" / "private_tap.yaml")
    facts_market = _read_yaml(inputs_dir / "facts" / "market_env.yaml")
    curated_models = _read_curated_models(inputs_dir / "operator" / "curated_public_tap_models.csv")
    curated_gpu = _read_curated_gpu(inputs_dir / "operator" / "curated_gpu.csv")

    return {
        "inputs_dir": str(inputs_dir),
        "simulation": sim,
        "operator": {
            "public_tap": pub,
            "private_tap": prv,
        },
        "facts": {
            "market_env": facts_market,
        },
        "curated": {
            "public_models": curated_models,
            "gpu_rentals": curated_gpu,
        },
    }


def variable_grid(state: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    # v0: single configuration passthrough; expand later per 16_simulation_variables.md
    yield state
