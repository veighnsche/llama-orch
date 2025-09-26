"""Scenario application (base/best/worst deltas)"""
from __future__ import annotations

import json
from typing import Any, Dict
from .money import D, HUNDRED


def apply_scenario(base: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    d = json.loads(json.dumps(base))  # deep copy via json
    if scenario == 'base':
        return d
    delta = base.get('scenario_delta', {}).get(scenario, {})
    om = d.get('omzetmodel', {})
    if 'omzet_pm_pct' in delta:
        om['omzet_pm'] = float(D(str(om.get('omzet_pm', 0))) * (D('1') + D(str(delta['omzet_pm_pct'])) / HUNDRED))
    if 'cogs_pct' in delta:
        om['cogs_pct'] = float(D(str(om.get('cogs_pct', 0))) + D(str(delta['cogs_pct'])))
    if 'opex_pm_pct' in delta:
        opex = om.get('opex_pm', {}) or {}
        new_opex = {}
        for k, v in opex.items():
            new_opex[k] = float(D(str(v)) * (D('1') + D(str(delta['opex_pm_pct'])) / HUNDRED))
        om['opex_pm'] = new_opex
    d['omzetmodel'] = om
    return d
