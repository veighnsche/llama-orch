from __future__ import annotations

from typing import Any, Dict, List, Tuple

from finplan.common.calendar import month_str, parse_month, add_months


def _carry_forward(arr: List[Any], length: int) -> List[Any]:
    if not arr:
        return [0] * length
    out = list(arr[:length])
    if len(out) < length:
        last = out[-1]
        out.extend([last] * (length - len(out)))
    return out


def months_range_str(start_ym: str, horizon: int) -> List[str]:
    start = parse_month(start_ym)
    return [month_str(add_months(start, i)) for i in range(horizon)]


def validate_and_default(cfg: Dict[str, Any], months: int) -> Tuple[Dict[str, Any], List[str]]:
    """Apply defaults and normalize arrays to the given horizon.
    Returns (normalized_cfg, months_list).
    """
    d = {**cfg}
    bedrijf = d.setdefault('bedrijf', {})
    bedrijf.setdefault('naam', 'Onbekend')
    bedrijf.setdefault('rechtsvorm', 'IB')
    bedrijf.setdefault('start_maand', '2025-01')
    bedrijf.setdefault('valuta', 'EUR')

    horizon = int(d.get('horizon_maanden') or months or 12)
    d['horizon_maanden'] = horizon
    months_list = months_range_str(bedrijf['start_maand'], horizon)

    d.setdefault('scenario', 'base')
    d.setdefault('vat_period', 'monthly')

    btw = d.setdefault('btw', {})
    btw.setdefault('btw_pct', 21)
    btw.setdefault('model', 'omzet_enkel')
    btw.setdefault('kor', False)
    btw.setdefault('btw_vrij', False)
    btw.setdefault('mkb_vrijstelling_pct', 0)

    # omzetstromen arrays
    streams = d.setdefault('omzetstromen', [])
    for s in streams:
        s.setdefault('naam', 'Stroom')
        s.setdefault('prijs', 0)
        s.setdefault('var_kosten_per_eenheid', 0)
        s['volume_pm'] = _carry_forward(list(s.get('volume_pm') or []), horizon)
        s.setdefault('btw_pct', btw.get('btw_pct', 21))
        s.setdefault('dso_dagen', d.get('werkkapitaal', {}).get('dso_dagen', 30))

    # OPEX structure
    opex = d.setdefault('opex_vast_pm', {})
    if not isinstance(opex.get('personeel'), list):
        opex['personeel'] = []
    opex.setdefault('marketing', 0)
    opex.setdefault('software', 0)
    opex.setdefault('huisvesting', 0)
    opex.setdefault('overig', 0)

    # Investeringen
    inv = d.setdefault('investeringen', [])
    for it in inv:
        it.setdefault('omschrijving', 'Investering')
        it.setdefault('levensduur_mnd', 36)
        it.setdefault('start_maand', bedrijf['start_maand'])

    # Financiering
    fin = d.setdefault('financiering', {})
    fin.setdefault('eigen_inbreng', 0)
    fin.setdefault('leningen', [])
    for ln in fin['leningen']:
        ln.setdefault('verstrekker', 'Onbekend')
        ln.setdefault('hoofdsom', 0)
        ln.setdefault('rente_nominaal_jr', ln.get('rente_nominaal_jr_pct', 0))
        ln.setdefault('looptijd_mnd', 0)
        ln.setdefault('grace_mnd', 0)
        ln.setdefault('alleen_rente_in_grace', True)

    # Werkkapitaal
    wc = d.setdefault('werkkapitaal', {})
    wc.setdefault('dso_dagen', 30)
    wc.setdefault('dpo_dagen', 14)
    wc.setdefault('dio_dagen', 0)
    wc.setdefault('vooruitbetaald_pm', 0)
    wc.setdefault('deferred_revenue_pm', 0)

    # Assumpties
    asm = d.setdefault('assumpties', {})
    asm.setdefault('discount_rate_pct', 10.0)
    asm.setdefault('kas_buffer_norm', 2500)

    return d, months_list
