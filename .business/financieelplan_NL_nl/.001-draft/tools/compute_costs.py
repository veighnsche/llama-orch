#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, getcontext
from pathlib import Path
from typing import Dict, List, Optional
import yaml

getcontext().prec = 28
Q2 = Decimal('0.01')

BASE_DIR = Path(__file__).resolve().parent.parent
RESEARCH_DIR = BASE_DIR / 'research'
OUT_DIR = BASE_DIR / 'out'
HOURS_PER_MONTH = Decimal('720')

@dataclass
class OnPremGPU:
    id: str
    name: str
    vram_gb: int
    purchase_price_eur: Decimal
    depreciation_months: int
    power_watts_full: Decimal
    electricity_eur_per_kwh: Decimal
    cooling_overhead_pct: Decimal
    maintenance_eur_pm: Decimal
    target_utilization_pct: Decimal
    def effective_hourly_eur(self) -> Decimal:
        dep_pm = (self.purchase_price_eur / Decimal(self.depreciation_months)) if self.depreciation_months else Decimal('0')
        kwh_pm = (self.power_watts_full / Decimal('1000')) * HOURS_PER_MONTH * (Decimal('1') + self.cooling_overhead_pct / Decimal('100'))
        elec_pm = kwh_pm * self.electricity_eur_per_kwh
        tot_pm = dep_pm + self.maintenance_eur_pm + elec_pm
        util = max(Decimal('0.01'), self.target_utilization_pct / Decimal('100'))
        eff_hours = HOURS_PER_MONTH * util
        return (tot_pm / eff_hours).quantize(Q2, rounding=ROUND_HALF_UP)

@dataclass
class ProviderSKU:
    provider_id: str
    sku_id: str
    mode: str
    hourly_eur: Optional[Decimal]
    min_hours: int
    notes: str

@dataclass
class ModelPerf:
    model_id: str
    model_name: str
    gpu_sku_id: str
    tokens_per_second: Decimal


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def load_onprem_gpus(data: Dict) -> Dict[str, OnPremGPU]:
    out: Dict[str, OnPremGPU] = {}
    for item in data.get('on_prem', []) or []:
        out[item['id']] = OnPremGPU(
            id=item['id'],
            name=item['name'],
            vram_gb=int(item.get('vram_gb') or 0),
            purchase_price_eur=Decimal(str(item.get('purchase_price_eur') or '0')),
            depreciation_months=int(item.get('depreciation_months') or 36),
            power_watts_full=Decimal(str(item.get('power_watts_full') or '0')),
            electricity_eur_per_kwh=Decimal(str(item.get('electricity_eur_per_kwh') or '0.30')),
            cooling_overhead_pct=Decimal(str(item.get('cooling_overhead_pct') or '10')),
            maintenance_eur_pm=Decimal(str(item.get('maintenance_eur_pm') or '0')),
            target_utilization_pct=Decimal(str(item.get('target_utilization_pct') or '60')),
        )
    return out


def load_provider_skus(data: Dict) -> Dict[str, List[ProviderSKU]]:
    out: Dict[str, List[ProviderSKU]] = {}
    for p in data.get('providers', []) or []:
        pid = p['id']
        lst: List[ProviderSKU] = []
        for sku in p.get('skus', []) or []:
            hourly = sku.get('hourly_eur')
            lst.append(ProviderSKU(
                provider_id=pid,
                sku_id=sku['sku_id'],
                mode=sku.get('mode', 'on_demand'),
                hourly_eur=Decimal(str(hourly)) if hourly is not None else None,
                min_hours=int(sku.get('min_hours') or 0),
                notes=sku.get('notes') or '',
            ))
        out[pid] = lst
    return out


def load_model_perf(data: Dict) -> List[ModelPerf]:
    out: List[ModelPerf] = []
    for m in data.get('models', []) or []:
        mid = m['id']
        name = m.get('name', mid)
        perf = m.get('tokens_per_second_per_gpu') or {}
        for sku_id, tps in perf.items():
            if tps is None:
                continue
            out.append(ModelPerf(
                model_id=mid,
                model_name=name,
                gpu_sku_id=sku_id,
                tokens_per_second=Decimal(str(tps)),
            ))
    return out


def cost_per_1m_tokens(hourly_eur: Decimal, tps: Decimal) -> Decimal:
    if tps <= 0:
        return Decimal('0')
    seconds_for_1m = Decimal('1000000') / tps
    hours_for_1m = seconds_for_1m / Decimal('3600')
    return (hourly_eur * hours_for_1m).quantize(Q2, rounding=ROUND_HALF_UP)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gpus_data = load_yaml(RESEARCH_DIR / 'gpus.yaml') if (RESEARCH_DIR / 'gpus.yaml').exists() else {}
    providers_data = load_yaml(RESEARCH_DIR / 'providers.yaml') if (RESEARCH_DIR / 'providers.yaml').exists() else {}
    models_data = load_yaml(RESEARCH_DIR / 'models.yaml') if (RESEARCH_DIR / 'models.yaml').exists() else {}

    onprem = load_onprem_gpus(gpus_data)
    providers = load_provider_skus(providers_data)
    perf = load_model_perf(models_data)

    sku_hourlies: Dict[str, List[ProviderSKU]] = {}
    for pid, lst in providers.items():
        for s in lst:
            sku_hourlies.setdefault(s.sku_id, []).append(s)

    rows = []
    for mp in perf:
        if mp.gpu_sku_id in onprem:
            hourly = onprem[mp.gpu_sku_id].effective_hourly_eur()
            rows.append({
                'model_id': mp.model_id,
                'model_name': mp.model_name,
                'where': 'on_prem',
                'provider': onprem[mp.gpu_sku_id].name,
                'gpu_sku': mp.gpu_sku_id,
                'mode': 'owned',
                'hourly_eur': str(hourly),
                'tps': str(mp.tokens_per_second),
                'eur_per_1m_tokens': str(cost_per_1m_tokens(hourly, mp.tokens_per_second)),
            })
        for s in sku_hourlies.get(mp.gpu_sku_id, []) or []:
            if s.hourly_eur is None:
                continue
            rows.append({
                'model_id': mp.model_id,
                'model_name': mp.model_name,
                'where': 'cloud',
                'provider': s.provider_id,
                'gpu_sku': s.sku_id,
                'mode': s.mode,
                'hourly_eur': str(s.hourly_eur.quantize(Q2, rounding=ROUND_HALF_UP)),
                'tps': str(mp.tokens_per_second),
                'eur_per_1m_tokens': str(cost_per_1m_tokens(s.hourly_eur, mp.tokens_per_second)),
            })

    # CSV
    csv_path = OUT_DIR / '15_compute_costs.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['model_id','model_name','where','provider','gpu_sku','mode','hourly_eur','tps','eur_per_1m_tokens'])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # MD summary
    md_path = OUT_DIR / '15_compute_costs.md'
    lines: List[str] = []
    lines.append('# Compute Kosten (Onderzoek)')
    lines.append('')
    lines.append('Bron: research/*.yaml → tools/compute_costs.py. Gebruik deze resultaten om var_kosten_per_eenheid (€/per 1M tokens) te onderbouwen, zowel on‑prem als cloud/dedicated.')
    lines.append('')
    by_model: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_model.setdefault(r['model_id'], []).append(r)
    for mid, lst in by_model.items():
        lst_sorted = sorted(lst, key=lambda x: (Decimal(x['eur_per_1m_tokens']), x['where']))
        top = lst_sorted[:5]
        lines.append(f'## Model: {top[0]["model_name"]} ({mid})')
        lines.append('')
        lines.append('| Waar | Provider | GPU | Mode | €/uur | t/s | €/ per 1M tokens |')
        lines.append('|------|----------|-----|------|------:|----:|------------------:|')
        for r in top:
            lines.append(f"| {r['where']} | {r['provider']} | {r['gpu_sku']} | {r['mode']} | {Decimal(r['hourly_eur']):.2f} | {Decimal(r['tps'])} | {Decimal(r['eur_per_1m_tokens']):.2f} |")
        lines.append('')
    md_path.write_text('\n'.join(lines), encoding='utf-8')
    print('Research outputs written:', csv_path.name, md_path.name)

if __name__ == '__main__':
    main()
