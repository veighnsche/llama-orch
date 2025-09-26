#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

# Optional YAML support
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore


def load_any(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yml", ".yaml"):
        if yaml is None:
            raise SystemExit("PyYAML is required to read YAML; install pyyaml or provide JSON input")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise SystemExit("Top-level input must be a mapping")
    return data


def dump_any(path: Path, data: Dict[str, Any]) -> None:
    suf = path.suffix.lower()
    if suf in (".yml", ".yaml") and yaml is not None:
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def migrate_v1_to_v2(v1: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "schema_version": 1,
        "bedrijf": {},
        "horizon_maanden": v1.get("horizon_maanden", 12),
        "scenario": v1.get("scenario", "base"),
        "vat_period": v1.get("belastingen", {}).get("vat_period", v1.get("vat_period", "monthly")),
        "btw": {},
        "omzetstromen": [],
        "opex_vast_pm": {},
        "investeringen": [],
        "financiering": {"eigen_inbreng": 0, "leningen": []},
        "werkkapitaal": {"dso_dagen": 30, "dpo_dagen": 14, "dio_dagen": 0, "vooruitbetaald_pm": 0, "deferred_revenue_pm": 0},
        "assumpties": {"discount_rate_pct": 10.0, "kas_buffer_norm": 2500},
    }

    # Bedrijf
    bedrijf = v1.get("bedrijf", {}) or {}
    out["bedrijf"] = {
        "naam": bedrijf.get("naam", bedrijf.get("handelsnaam", "Onbekend")),
        "rechtsvorm": bedrijf.get("rechtsvorm", "IB"),
        "start_maand": bedrijf.get("start_maand", "2025-01"),
        "valuta": bedrijf.get("valuta", "EUR"),
    }

    # BTW / belastingen
    bel = v1.get("belastingen", {}) or {}
    out["btw"] = {
        "btw_pct": bel.get("btw_pct", v1.get("btw_pct", 21)),
        "model": bel.get("btw_model", bel.get("btw_model", "omzet_enkel")),
        "kor": bel.get("kor", False),
        "btw_vrij": bel.get("btw_vrij", False),
        "mkb_vrijstelling_pct": bel.get("mkb_vrijstelling_pct", 0),
    }

    # Omzetstromen from v1.omzetmodel
    om = v1.get("omzetmodel", {}) or {}
    omzet_pm = om.get("omzet_pm", 0)
    cogs_pct = om.get("cogs_pct", 0)
    price = 1.0
    var_per_unit = float(cogs_pct) / 100.0  # ensures COGS = omzet * cogs_pct
    horizon = int(out.get("horizon_maanden", 12))
    if isinstance(omzet_pm, (int, float)):
        vol_arr = [float(omzet_pm)] * horizon
    elif isinstance(omzet_pm, list):
        vol_arr = ([float(x) for x in omzet_pm] + [float(omzet_pm[-1])] * horizon)[:horizon]
    else:
        vol_arr = [0.0] * horizon
    out["omzetstromen"].append({
        "naam": "Omzet",
        "prijs": price,
        "volume_pm": vol_arr,
        "var_kosten_per_eenheid": var_per_unit,
        "btw_pct": out["btw"].get("btw_pct", 21),
        "dso_dagen": v1.get("werkkapitaal", {}).get("dso_dagen", 30),
    })

    # OPEX mapping
    opex_v1 = om.get("opex_pm", {}) or {}
    personeel = []
    salarissen = opex_v1.get("salarissen", 0)
    if salarissen:
        personeel.append({"rol": "Salarissen", "bruto_pm": float(salarissen)})
    out["opex_vast_pm"] = {
        "personeel": personeel,
        "marketing": float(opex_v1.get("marketing", 0)),
        "software": float(opex_v1.get("ict", 0)),
        "huisvesting": float(opex_v1.get("huur", 0)),
        "overig": float(opex_v1.get("overig", 0)) + float(opex_v1.get("verzekeringen", 0)),
    }

    # Investeringen
    for it in (v1.get("investeringen", []) or []):
        out["investeringen"].append({
            "omschrijving": it.get("omschrijving", "Investering"),
            "bedrag": float(it.get("bedrag", 0)),
            "levensduur_mnd": int(it.get("levensduur_mnd", 36)),
            "start_maand": it.get("start_maand", out["bedrijf"]["start_maand"]),
        })

    # Financiering
    fin = v1.get("financiering", {}) or {}
    out["financiering"]["eigen_inbreng"] = float(fin.get("eigen_inbreng", 0))
    for ln in fin.get("leningen", []) or []:
        out["financiering"]["leningen"].append({
            "verstrekker": ln.get("verstrekker", "Onbekend"),
            "hoofdsom": float(ln.get("hoofdsom", 0)),
            "rente_nominaal_jr": float(ln.get("rente_nominaal_jr", ln.get("rente_nominaal_jr_pct", 0))),
            "looptijd_mnd": int(ln.get("looptijd_mnd", 0)),
            "grace_mnd": int(ln.get("grace_mnd", 0)),
            "alleen_rente_in_grace": bool(ln.get("alleen_rente_in_grace", True)),
        })

    # Werkkapitaal defaults, reuse if present
    wc = v1.get("werkkapitaal", {}) or {}
    out["werkkapitaal"].update({
        "dso_dagen": int(wc.get("dso_dagen", out["werkkapitaal"]["dso_dagen"])),
        "dpo_dagen": int(wc.get("dpo_dagen", out["werkkapitaal"]["dpo_dagen"])),
        "dio_dagen": int(wc.get("dio_dagen", out["werkkapitaal"]["dio_dagen"])),
    })

    return out


def main(argv: list[str]) -> None:
    if len(argv) < 3:
        print("Usage: migrate_v1_to_v2.py <input.yml|json> <output.json>")
        sys.exit(2)
    src = Path(argv[1]).resolve()
    dst = Path(argv[2]).resolve()
    v1 = load_any(src)
    v2 = migrate_v1_to_v2(v1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dump_any(dst, v2)
    print(f"Migrated {src} -> {dst}")


if __name__ == "__main__":
    main(sys.argv)
