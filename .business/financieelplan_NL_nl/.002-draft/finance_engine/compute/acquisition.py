from __future__ import annotations

from typing import Dict, Any, Tuple, Iterable, Optional

import math


def _clampf(x: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo


def _num(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _case_multipliers(funnel_overrides: Dict[str, Any], case: str) -> Tuple[float, float, float]:
    """Return (budget_multiplier, cvr_multiplier, cac_multiplier) for a case name."""
    node = (funnel_overrides or {}).get(case, {})
    bm = _num(node.get("budget_multiplier"), 1.0)
    cvr_m = _num(node.get("cvr_multiplier"), 1.0)
    cac_m = _num(node.get("cac_multiplier"), 1.0)
    return max(0.0, bm), max(0.0, cvr_m), max(1e-9, cac_m)


def _segments(acq: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if isinstance(acq.get("segments"), dict):
        for name in ["oss", "agencies", "it_teams", "compliance"]:
            node = acq["segments"].get(name)
            if isinstance(node, dict):
                yield name, node

def _simulate_one(channels: Any, global_node: Dict[str, Any], *, bm: float, cvr_m: float, cac_m: float) -> Tuple[float, float]:
    total_paid_new = 0.0
    total_free_new = 0.0
    total_marketing_eur = 0.0
    avg_paid = _num(global_node.get("avg_tokens_paid_per_user_per_month"), 0.0)
    avg_free = _num(global_node.get("avg_tokens_free_per_user_per_month"), 0.0)
    if not isinstance(channels, list):
        channels = []
    for ch in channels:
        if not isinstance(ch, dict):
            continue
        budget = _num(ch.get("budget_eur"), 0.0) * bm
        if budget <= 0:
            continue
        total_marketing_eur += budget
        cpc = _num(ch.get("cpc_eur"), 0.0)
        v2s = _num(ch.get("visit_to_signup_cvr_pct"), 0.0) / 100.0
        s2p = _num(ch.get("signup_to_paid_cvr_pct"), 0.0) / 100.0
        oss_share = _clampf(_num(ch.get("oss_share_pct"), 0.0) / 100.0, 0.0, 1.0)
        cac = _num(ch.get("cac_eur"), 0.0)
        if cpc > 0 and v2s > 0 and s2p > 0:
            eff_v2s = _clampf(v2s * cvr_m, 0.0, 1.0)
            eff_s2p = _clampf(s2p * cvr_m, 0.0, 1.0)
            visits = budget / cpc
            signups = visits * eff_v2s
            paid_new = signups * eff_s2p
            non_paid = max(signups - paid_new, 0.0)
            free_new = non_paid * oss_share
        elif cac > 0:
            eff_cac = cac * cac_m
            paid_new = budget / eff_cac if eff_cac > 0 else 0.0
            free_new = 0.0
        else:
            continue
        total_paid_new += max(0.0, paid_new)
        total_free_new += max(0.0, free_new)
    m_tokens = ((total_paid_new * avg_paid) + (total_free_new * avg_free)) / 1_000_000.0

    return float(m_tokens), float(total_marketing_eur)

def simulate_m_tokens_from_funnel(
    *, acquisition: Dict[str, Any], funnel_overrides: Dict[str, Any] | None, case: str, segment: Optional[str] = None
) -> Tuple[float, float]:
    """
    Compute monthly public usage in million tokens and total marketing spend (EUR) for a scenario case.

    Returns (m_tokens_public, marketing_spend_eur).
    Rules:
    - Prefer CPC + CVRs if provided to compute paid/free volumes; if only CAC is present, compute paid_new = budget / (cac * cac_multiplier) and assume free_new = 0 from that channel.
    - Multipliers applied:
      - budget_multiplier scales channel budgets
      - cvr_multiplier scales both conversion rates
      - cac_multiplier scales CAC (effective_cac = cac * cac_multiplier)
    - Free (OSS) users estimated as a share of non-paid signups when CPC + CVRs are provided.
    - If segmented acquisition is present, aggregate over required segments.
    """
    acq = acquisition or {}
    bm, cvr_m, cac_m = _case_multipliers(funnel_overrides or {}, case)

    # Specific Segment
    if segment is not None and isinstance(acq.get("segments"), dict):
        node = acq["segments"].get(segment) or {}
        m, mk = _simulate_one(node.get("channels"), node.get("global") or {}, bm=bm, cvr_m=cvr_m, cac_m=cac_m)
        return float(m), float(mk)

    # Aggregate across all segments if present, else treat as flat
    if isinstance(acq.get("segments"), dict):
        m_total = 0.0
        mk_total = 0.0
        for _, node in _segments(acq):
            m, mk = _simulate_one(node.get("channels"), node.get("global") or {}, bm=bm, cvr_m=cvr_m, cac_m=cac_m)
            m_total += m
            mk_total += mk
        return float(m_total), float(mk_total)

    # Legacy flat shape fallback
    m, mk = _simulate_one(acq.get("channels"), acq.get("global") or acq.get("global_") or {}, bm=bm, cvr_m=cvr_m, cac_m=cac_m)
    return float(m), float(mk)


def simulate_funnel_details(
    *, acquisition: Dict[str, Any], funnel_overrides: Dict[str, Any] | None, case: str, segment: Optional[str] = None
) -> Dict[str, float]:
    """
    Detailed simulation returning counts and tokens decomposition for a given case.

    Returns dict with keys:
      - paid_new
      - free_new
      - visits (if CPC available; else 0)
      - signups (if CPC available; else 0)
      - marketing_eur
      - paid_tokens_m
      - free_tokens_m
      - total_tokens_m
    """
    acq = acquisition or {}
    if segment is not None and isinstance(acq.get("segments"), dict):
        seg = acq["segments"].get(segment) or {}
        channels = seg.get("channels") or []
        global_node = seg.get("global") or {}
    else:
        # Combine all segments
        channels = []
        global_node = acq.get("global") or acq.get("global_") or {}
        # Use a weighted average approach for tokens per user if multiple segments present is non-trivial; keep global fallback
        if isinstance(acq.get("segments"), dict):
            for _, node in _segments(acq):
                if isinstance(node.get("channels"), list):
                    channels.extend(node.get("channels"))
                # Prefer the segment globals only if no top-level globals provided
                if not global_node and isinstance(node.get("global"), dict):
                    global_node = node.get("global")

    avg_paid = _num(global_node.get("avg_tokens_paid_per_user_per_month"), 0.0)
    avg_free = _num(global_node.get("avg_tokens_free_per_user_per_month"), 0.0)

    bm, cvr_m, cac_m = _case_multipliers(funnel_overrides or {}, case)

    paid_new = 0.0
    free_new = 0.0
    visits_total = 0.0
    signups_total = 0.0
    marketing_eur = 0.0

    for ch in channels:
        budget = _num(ch.get("budget_eur"), 0.0) * bm
        if budget <= 0:
            continue
        marketing_eur += budget
        cpc = _num(ch.get("cpc_eur"), 0.0)
        v2s = _num(ch.get("visit_to_signup_cvr_pct"), 0.0) / 100.0
        s2p = _num(ch.get("signup_to_paid_cvr_pct"), 0.0) / 100.0
        oss_share = _clampf(_num(ch.get("oss_share_pct"), 0.0) / 100.0, 0.0, 1.0)
        cac = _num(ch.get("cac_eur"), 0.0)

        if cpc > 0 and v2s > 0 and s2p > 0:
            eff_v2s = _clampf(v2s * cvr_m, 0.0, 1.0)
            eff_s2p = _clampf(s2p * cvr_m, 0.0, 1.0)
            visits = budget / cpc
            signups = visits * eff_v2s
            paid = signups * eff_s2p
            non_paid = max(signups - paid, 0.0)
            free = non_paid * oss_share
            paid_new += max(0.0, paid)
            free_new += max(0.0, free)
            visits_total += max(0.0, visits)
            signups_total += max(0.0, signups)
        elif cac > 0:
            eff_cac = cac * cac_m
            paid = budget / eff_cac if eff_cac > 0 else 0.0
            paid_new += max(0.0, paid)
        else:
            continue

    paid_tokens_m = (paid_new * avg_paid) / 1_000_000.0
    free_tokens_m = (free_new * avg_free) / 1_000_000.0
    total_tokens_m = paid_tokens_m + free_tokens_m
    return {
        "paid_new": float(paid_new),
        "free_new": float(free_new),
        "visits": float(visits_total),
        "signups": float(signups_total),
        "marketing_eur": float(marketing_eur),
        "paid_tokens_m": float(paid_tokens_m),
        "free_tokens_m": float(free_tokens_m),
        "total_tokens_m": float(total_tokens_m),
    }
