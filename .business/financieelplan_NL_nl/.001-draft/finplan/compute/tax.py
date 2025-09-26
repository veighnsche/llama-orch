from __future__ import annotations

from typing import Dict, Any, Tuple

from finplan.common.money import D, CENT

# Indicative flat rates (no advice). Simple and transparent.
IB_RATE = D("0.28")     # 28% indicative
VPB_RATE = D("0.258")   # 25.8% indicative


def _sum_result_vb(model: Dict[str, Any]) -> D:
    months = model["months"]
    revenue = model["revenue"]
    cogs = model["cogs"]
    opex_total = model["opex_total"]
    depreciation = model["depreciation"]
    interest = model["interest"]
    total = D("0")
    for m in months:
        total += (revenue[m] - cogs[m] - opex_total[m] - depreciation[m] - interest[m])
    return total.quantize(CENT)


def compute_indicative_tax(model: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[D, D]:
    """Return (indicatieve_heffing_jaar, belastingdruk_pct_as_ratio).
    belastingdruk_pct_as_ratio is a ratio (e.g., 0.12 for 12%).
    """
    resultaat_vb_totaal = _sum_result_vb(model)
    if resultaat_vb_totaal <= 0:
        return D("0"), D("0")

    regime = (cfg.get("bedrijf", {}) or {}).get("rechtsvorm", "IB")
    btw_cfg = (cfg.get("btw", {}) or {})
    mkb_pct = D(str(btw_cfg.get("mkb_vrijstelling_pct", 0))) / D("100")

    taxable = (resultaat_vb_totaal * (D("1") - mkb_pct)).quantize(CENT)

    rate = IB_RATE if regime == "IB" else VPB_RATE
    heffing = (taxable * rate).quantize(CENT)
    druk = (heffing / resultaat_vb_totaal) if resultaat_vb_totaal != 0 else D("0")
    return heffing, druk
