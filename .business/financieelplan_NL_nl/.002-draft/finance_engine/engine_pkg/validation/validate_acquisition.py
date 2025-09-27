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


essential_global_keys = [
    "avg_tokens_paid_per_user_per_month",
    "avg_tokens_free_per_user_per_month",
    "churn_pct_per_month",
]


def _validate_segment(fr: FileReport, seg_name: str, node: Dict[str, Any]) -> int:
    if not isinstance(node, dict):
        fr.ok = False
        fr.errors.append(f"segments.{seg_name}: must be a mapping")
        return 0
    g = node.get("global") or {}
    if not isinstance(g, dict):
        fr.ok = False
        fr.errors.append(f"segments.{seg_name}.global: required mapping")
    else:
        for k in essential_global_keys:
            if k not in g or not is_number(g.get(k)) or float(g.get(k)) < 0:
                fr.ok = False
                fr.errors.append(f"segments.{seg_name}.global.{k}: required numeric ≥ 0")
        # Optional percentages on global
        for key in ("payment_fee_pct_of_revenue",):
            if key in g and not _pct_ok(g.get(key)):
                fr.ok = False
                fr.errors.append(f"segments.{seg_name}.global.{key}: must be 0–100 if present")

    channels = node.get("channels")
    if channels is None or not isinstance(channels, list):
        fr.ok = False
        fr.errors.append(f"segments.{seg_name}.channels: required list (can be empty)")
        return 0
    count = 0
    for i, ch in enumerate(channels):
        if not isinstance(ch, dict):
            fr.ok = False
            fr.errors.append(f"segments.{seg_name}.channels[{i}]: must be a mapping")
            continue
        name = ch.get("name")
        if not non_empty_string(name):
            fr.ok = False
            fr.errors.append(f"segments.{seg_name}.channels[{i}].name: required non-empty string")
        budget = ch.get("budget_eur")
        if not is_number(budget) or float(budget) < 0:
            fr.ok = False
            fr.errors.append(f"segments.{seg_name}.channels[{i}].budget_eur: required numeric ≥ 0")
        # At least one of CPC pathway or CAC pathway should be provided
        cpc = ch.get("cpc_eur")
        cac = ch.get("cac_eur")
        v2s = ch.get("visit_to_signup_cvr_pct")
        s2p = ch.get("signup_to_paid_cvr_pct")
        if is_number(cpc) and float(cpc) > 0:
            if not _pct_ok(v2s):
                fr.ok = False
                fr.errors.append(f"segments.{seg_name}.channels[{i}].visit_to_signup_cvr_pct: 0–100 required when cpc_eur provided")
            if not _pct_ok(s2p):
                fr.ok = False
                fr.errors.append(f"segments.{seg_name}.channels[{i}].signup_to_paid_cvr_pct: 0–100 required when cpc_eur provided")
            if "oss_share_pct" in ch and not _pct_ok(ch.get("oss_share_pct")):
                fr.ok = False
                fr.errors.append(f"segments.{seg_name}.channels[{i}].oss_share_pct: must be 0–100 if present")
        elif is_number(cac) and float(cac) > 0:
            pass
        else:
            fr.ok = False
            fr.errors.append(
                f"segments.{seg_name}.channels[{i}]: provide either CPC (cpc_eur>0 with conversion rates) or CAC (cac_eur>0)"
            )
        count += 1
    return count


def validate(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "acquisition.yaml"
    fr = FileReport(name="acquisition.yaml", ok=True)
    if not p.exists():
        fr.ok = False
        fr.errors.append("acquisition.yaml missing (required)")
        return fr
    try:
        obj = load_yaml(p)
        if not isinstance(obj, dict):
            fr.ok = False
            fr.errors.append("Top-level structure must be a mapping")
            return fr
        segs = obj.get("segments")
        if not isinstance(segs, dict):
            fr.ok = False
            fr.errors.append("segments: required mapping with keys oss, agencies, it_teams, compliance")
            return fr
        required_segs = ["oss", "agencies", "it_teams", "compliance"]
        total_channels = 0
        for s in required_segs:
            if s not in segs:
                fr.ok = False
                fr.errors.append(f"segments.{s}: required segment missing")
                continue
            total_channels += _validate_segment(fr, s, segs.get(s) or {})
        # At least one channel across all segments
        if total_channels == 0:
            fr.ok = False
            fr.errors.append("segments: must define at least one channel across segments")
        fr.count = total_channels
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr
