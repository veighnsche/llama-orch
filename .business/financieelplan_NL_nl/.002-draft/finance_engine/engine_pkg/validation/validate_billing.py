from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .shared import FileReport, is_number, non_empty_string
from ...io.loader import load_yaml


def _pct_ok(x: Any) -> bool:
    try:
        v = float(x)
        return 0.0 <= v <= 100.0
    except Exception:
        return False


def validate(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "billing.yaml"
    fr = FileReport(name="billing.yaml", ok=True)
    if not p.exists():
        fr.ok = False
        fr.errors.append("billing.yaml missing (required)")
        return fr
    try:
        obj: Dict[str, Any] = load_yaml(p)
        if not isinstance(obj, dict):
            fr.ok = False
            fr.errors.append("Top-level structure must be a mapping")
            return fr
        prepaid = obj.get("prepaid")
        if not isinstance(prepaid, dict):
            fr.ok = False
            fr.errors.append("prepaid: required mapping")
            return fr
        packs = prepaid.get("credit_packs")
        if not isinstance(packs, list) or len(packs) == 0:
            fr.ok = False
            fr.errors.append("prepaid.credit_packs: required non-empty list")
        else:
            names: List[str] = []
            for i, pk in enumerate(packs):
                if not isinstance(pk, dict):
                    fr.ok = False
                    fr.errors.append(f"prepaid.credit_packs[{i}]: must be mapping { '{name, eur}' }")
                    break
                nm = pk.get("name")
                eur = pk.get("eur")
                if not non_empty_string(nm):
                    fr.ok = False
                    fr.errors.append(f"prepaid.credit_packs[{i}].name: required non-empty string")
                if not is_number(eur) or float(eur) <= 0:
                    fr.ok = False
                    fr.errors.append(f"prepaid.credit_packs[{i}].eur: required numeric > 0")
                names.append(str(nm))
            mix = prepaid.get("pack_mix_pct")
            if not isinstance(mix, dict) or not mix:
                fr.ok = False
                fr.errors.append("prepaid.pack_mix_pct: required mapping of pack name -> percent")
            else:
                total = 0.0
                for k, v in mix.items():
                    if k not in names:
                        fr.ok = False
                        fr.errors.append(f"prepaid.pack_mix_pct: unknown pack name '{k}'")
                    if not _pct_ok(v):
                        fr.ok = False
                        fr.errors.append(f"prepaid.pack_mix_pct[{k}]: must be 0–100")
                    else:
                        total += float(v)
                if abs(total - 100.0) > 1e-6:
                    fr.ok = False
                    fr.errors.append("prepaid.pack_mix_pct: must sum to 100")
        # Other prepaid keys
        if not _pct_ok(prepaid.get("processor_fee_pct_of_revenue")):
            fr.ok = False
            fr.errors.append("prepaid.processor_fee_pct_of_revenue: required 0–100")
        if not is_number(prepaid.get("processor_fixed_fee_eur_per_tx")) or float(prepaid.get("processor_fixed_fee_eur_per_tx", 0)) < 0:
            fr.ok = False
            fr.errors.append("prepaid.processor_fixed_fee_eur_per_tx: required numeric ≥ 0")
        if int(float(prepaid.get("expiry_months", 0))) <= 0:
            fr.ok = False
            fr.errors.append("prepaid.expiry_months: required positive integer")
        if not is_number(prepaid.get("wallet_buffer_months")) or float(prepaid.get("wallet_buffer_months", 0)) < 0:
            fr.ok = False
            fr.errors.append("prepaid.wallet_buffer_months: required numeric ≥ 0")
        if str(prepaid.get("refund_policy")).strip().lower() != "none":
            fr.ok = False
            fr.errors.append("prepaid.refund_policy: must be 'none'")

        b2b = obj.get("b2b")
        if not isinstance(b2b, dict):
            fr.ok = False
            fr.errors.append("b2b: required mapping")
        else:
            if not _pct_ok(b2b.get("share_pct_of_revenue")):
                fr.ok = False
                fr.errors.append("b2b.share_pct_of_revenue: required 0–100")
            if not is_number(b2b.get("dso_days")) or float(b2b.get("dso_days", 0)) < 0:
                fr.ok = False
                fr.errors.append("b2b.dso_days: required numeric ≥ 0")
            if not _pct_ok(b2b.get("invoice_discount_pct")):
                fr.ok = False
                fr.errors.append("b2b.invoice_discount_pct: required 0–100")

        fr.count = 1
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr
