"""Scenario application (base/best/worst deltas) for v2 schema.
Supported delta keys (percent ints/floats):
- omzet_pm_pct: scales stream volumes (proxy for revenue)
- cogs_pct: scales var_kosten_per_eenheid
- opex_pm_pct: scales all fixed opex categories and personeel bruto_pm
"""
from __future__ import annotations

import json
from typing import Any, Dict
from .money import D, HUNDRED


def apply_scenario(base: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    d = json.loads(json.dumps(base))  # deep copy via json
    if scenario == 'base':
        return d
    delta = (base.get('scenario_delta', {}) or {}).get(scenario, {}) or {}
    if not delta:
        return d

    # Streams volume scale
    if 'omzet_pm_pct' in delta:
        f = (D('1') + D(str(delta['omzet_pm_pct'])) / HUNDRED)
        streams = d.get('omzetstromen', []) or []
        for s in streams:
            vols = s.get('volume_pm') or []
            s['volume_pm'] = [float(D(str(v)) * f) for v in vols]
        d['omzetstromen'] = streams

    # Variable cost per unit scale
    if 'cogs_pct' in delta:
        f = (D('1') + D(str(delta['cogs_pct'])) / HUNDRED)
        streams = d.get('omzetstromen', []) or []
        for s in streams:
            s['var_kosten_per_eenheid'] = float(D(str(s.get('var_kosten_per_eenheid', 0))) * f)
        d['omzetstromen'] = streams

    # OPEX scale
    if 'opex_pm_pct' in delta:
        f = (D('1') + D(str(delta['opex_pm_pct'])) / HUNDRED)
        opex = d.get('opex_vast_pm', {}) or {}
        # personeel list entries
        pers = opex.get('personeel') or []
        new_pers = []
        for p in pers:
            p = dict(p)
            p['bruto_pm'] = float(D(str(p.get('bruto_pm', 0))) * f)
            new_pers.append(p)
        opex['personeel'] = new_pers
        # scalar categories
        for cat in ['marketing', 'software', 'huisvesting', 'overig']:
            if cat in opex:
                opex[cat] = float(D(str(opex.get(cat, 0))) * f)
        d['opex_vast_pm'] = opex

    return d
