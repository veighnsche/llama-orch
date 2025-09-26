from __future__ import annotations

import copy
from typing import Dict, Any, List

from finplan.common.money import D
from finplan.compute.engine import compute_model


def _scale_streams(cfg: Dict[str, Any], factor: D) -> None:
    for s in (cfg.get("omzetstromen", []) or []):
        vols = s.get("volume_pm") or []
        s["volume_pm"] = [float(D(str(v)) * factor) for v in vols]


def _scale_opex(cfg: Dict[str, Any], factor: D) -> None:
    opex = cfg.get("opex_vast_pm", {}) or {}
    # personeel list entries
    pers = opex.get("personeel") or []
    for p in pers:
        p["bruto_pm"] = float(D(str(p.get("bruto_pm", 0))) * factor)
    opex["personeel"] = pers
    for cat in ["marketing", "software", "huisvesting", "overig"]:
        if cat in opex:
            opex[cat] = float(D(str(opex.get(cat, 0))) * factor)
    cfg["opex_vast_pm"] = opex


def _shift_dso(cfg: Dict[str, Any], plus_days: int) -> None:
    wc = cfg.setdefault("werkkapitaal", {})
    wc["dso_dagen"] = int(wc.get("dso_dagen", 30)) + int(plus_days)


def compute_stress_variants(cfg: Dict[str, Any], months: List[str], vat_period: str = "monthly") -> Dict[str, Dict[str, Any]]:
    """Return models for stress variants: omzet(-30%), dso(+30d), opex(+20%), and combo.
    Keys: 'omzet', 'dso', 'opex', 'combo'
    """
    horizon = len(months)
    # Omzet −30%
    c1 = copy.deepcopy(cfg)
    _scale_streams(c1, D("0.70"))
    m1 = compute_model(c1, horizon, scenario=c1.get("scenario", "base"), vat_period_flag=vat_period)

    # DSO +30 dagen
    c2 = copy.deepcopy(cfg)
    _shift_dso(c2, 30)
    m2 = compute_model(c2, horizon, scenario=c2.get("scenario", "base"), vat_period_flag=vat_period)

    # OPEX +20%
    c3 = copy.deepcopy(cfg)
    _scale_opex(c3, D("1.20"))
    m3 = compute_model(c3, horizon, scenario=c3.get("scenario", "base"), vat_period_flag=vat_period)

    # Combo
    c4 = copy.deepcopy(cfg)
    _scale_streams(c4, D("0.70"))
    _shift_dso(c4, 30)
    _scale_opex(c4, D("1.20"))
    m4 = compute_model(c4, horizon, scenario=c4.get("scenario", "base"), vat_period_flag=vat_period)

    return {"omzet": m1, "dso": m2, "opex": m3, "combo": m4}
