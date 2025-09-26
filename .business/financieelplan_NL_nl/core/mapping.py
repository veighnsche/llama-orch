from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any, Dict, List, Tuple

from .money import D, CENT, ZERO, money as fmt_money


def _sum(vals):
    s = ZERO
    for v in vals:
        s += D(str(v))
    return s.quantize(CENT)


def _pct(n: Decimal, d: Decimal) -> str:
    if d == 0:
        return "0%"
    return f"{(n / d * D('100')).quantize(D('0.01'))}%"


def _ratio_pct(a: Decimal, b: Decimal) -> str:
    tot = a + b
    if tot == 0:
        return "0% / 0%"
    ap = (a / tot * D('100')).quantize(D('0.01'))
    bp = (b / tot * D('100')).quantize(D('0.01'))
    return f"{ap}% / {bp}%"


def _avg(vals: List[Decimal]) -> Decimal:
    if not vals:
        return ZERO
    return (_sum(vals) / D(str(len(vals)))).quantize(CENT)


def _best_worst(result_by_month: Dict[str, Decimal]) -> Tuple[Tuple[str, Decimal], Tuple[str, Decimal]]:
    best = ("", None)
    worst = ("", None)
    for ym, v in result_by_month.items():
        if best[1] is None or v > best[1]:
            best = (ym, v)
        if worst[1] is None or v < worst[1]:
            worst = (ym, v)
    return (best[0], best[1] or ZERO), (worst[0], worst[1] or ZERO)


def _build_amort_index(amort_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Decimal]]:
    idx: Dict[str, Dict[str, Decimal]] = defaultdict(lambda: {"rente_totaal": ZERO, "restschuld_einde": ZERO})
    last_month: Dict[str, str] = {}
    for r in amort_rows:
        v = D(str(r.get('rente_pm', 0)))
        verstrekker = str(r.get('verstrekker', 'Onbekend'))
        idx[verstrekker]["rente_totaal"] += v
        last_month[verstrekker] = str(r.get('maand'))
        idx[verstrekker]["restschuld_einde"] = D(str(r.get('restschuld', 0)))
    return idx


def _breakeven_omzet_pm(revenue: Dict[str, Decimal], cogs: Dict[str, Decimal], opex_total: Dict[str, Decimal]) -> Decimal:
    rev_avg = _avg([revenue[m] for m in revenue])
    gm_ratio = ZERO
    if rev_avg > 0:
        gm_ratio = ((rev_avg - _avg([cogs[m] for m in cogs])) / rev_avg).quantize(D('0.0001'))
    fixed_costs = _avg([opex_total[m] for m in opex_total])
    if gm_ratio <= 0:
        return ZERO
    return (fixed_costs / gm_ratio).quantize(CENT)


def _runway(cash_end: Dict[str, Decimal]) -> int:
    max_streak = 0
    cur = 0
    for ym in cash_end.keys():
        if cash_end[ym] >= ZERO:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0
    return max_streak


def _build_streams(streams_cfg: List[Dict[str, Any]], months: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Decimal]]:
    out_list: List[Dict[str, Any]] = []
    totals: Dict[str, Decimal] = {m: ZERO for m in months}
    for s in streams_cfg or []:
        prijs = D(str(s.get('prijs', 0)))
        var_eh = D(str(s.get('var_kosten_per_eenheid', 0)))
        vols = s.get('volume_pm') or [0] * len(months)
        omzet_pm = ZERO
        marge_pm = ZERO
        for i, ym in enumerate(months):
            vol = D(str(vols[i])) if i < len(vols) else ZERO
            omzet = (prijs * vol).quantize(CENT)
            marge = ((prijs - var_eh) * vol).quantize(CENT)
            omzet_pm += omzet
            marge_pm += marge
            totals[ym] += omzet
        out_list.append({
            'naam': s.get('naam', 'Stroom'),
            'prijs': f"{prijs}",
            'volume_pm': f"{_avg([D(str(v)) for v in vols]).quantize(D('0.01'))}",
            'omzet_pm': fmt_money((omzet_pm / D(str(len(months)))).quantize(CENT)),
            'marge_pm': fmt_money((marge_pm / D(str(len(months)))).quantize(CENT)),
        })
    return out_list, totals


def build_template_context(model: Dict[str, Any]) -> Dict[str, Any]:
    cfg = model.get('config', {})
    months = model['months']
    bedrijf = cfg.get('bedrijf', {})
    btw_cfg = cfg.get('btw', {})
    fin = cfg.get('financiering', {})
    loans = fin.get('leningen', []) or []

    revenue = model['revenue']
    cogs = model['cogs']
    opex_total = model['opex_total']

    total_invest = model['total_invest']
    eigen_inbreng = model['eigen_inbreng']
    debt_total = _sum([ln.get('hoofdsom', 0) for ln in loans])
    total_financiering = (eigen_inbreng + debt_total).quantize(CENT)

    first_month = months[0] if months else ''
    last_month = months[-1] if months else ''

    # P&L aggregates
    omzet_totaal = _sum([revenue[m] for m in months])
    cogs_totaal = _sum([cogs[m] for m in months])
    brutomarge_totaal = (omzet_totaal - cogs_totaal).quantize(CENT)
    opex_totaal = _sum([opex_total[m] for m in months])
    afschrijving_totaal = _sum([model['depreciation'][m] for m in months])
    rente_totaal = _sum([model['interest'][m] for m in months])
    resultaat_vb_by_month: Dict[str, Decimal] = {
        m: (revenue[m] - cogs[m] - opex_total[m] - model['depreciation'][m] - model['interest'][m]).quantize(CENT)
        for m in months
    }
    resultaat_vb_totaal = _sum(resultaat_vb_by_month.values())
    ebitda_pm_avg = _avg([model['ebitda'][m] for m in months])

    best, worst = _best_worst(resultaat_vb_by_month)
    verlies_maanden = sum(1 for m in months if resultaat_vb_by_month[m] < ZERO)
    verlies_verdict = 'zorgpunt' if verlies_maanden > 0 else 'stabiel'

    # DSCR
    dscr_min = model['dscr_min']
    dscr_maand_min = model['dscr_maand_min']
    dscr_below_1_count = model['dscr_below_1_count']
    dscr_verdict = 'voldoende' if dscr_min >= D('1') else 'onvoldoende'

    # Breakeven
    be_omzet = _breakeven_omzet_pm(revenue, cogs, opex_total)
    omzet_basis_pm = _avg([revenue[m] for m in months])
    veiligheidsmarge_pct = _pct((omzet_basis_pm - be_omzet).quantize(CENT), be_omzet if be_omzet > 0 else D('1'))

    # Liquidity
    lowest_cash = model['lowest_cash']
    lowest_cash_month = model['lowest_cash_month']
    kas_negatief_count = sum(1 for m in months if model['cash_end'][m] < ZERO)
    kas_negatief_verdict = 'negatief' if kas_negatief_count > 0 else 'OK'

    # Amort summary per loan
    amort_idx = _build_amort_index(model.get('amort_rows', []) or [])
    leningen_ctx: List[Dict[str, Any]] = []
    for ln in loans:
        verstrekker = ln.get('verstrekker', 'Onbekend')
        hoofdsom = D(str(ln.get('hoofdsom', 0))).quantize(CENT)
        jr = D(str(ln.get('rente_nominaal_jr', ln.get('rente_nominaal_jr_pct', 0))))
        r_pm = (jr / D('12')).quantize(D('0.01'))
        looptijd = int(ln.get('looptijd_mnd', 0) or 0)
        grace = int(ln.get('grace_mnd', 0) or 0)
        term = '0.00'
        if looptijd > 0:
            # The CSV includes computed term; for MD we show string
            from .loan import annuity_payment
            term = str(annuity_payment(hoofdsom, (jr / D('100')) / D('12'), max(0, looptijd - grace)))
        entry = {
            'verstrekker': verstrekker,
            'hoofdsom': fmt_money(hoofdsom),
            'rente_nominaal_jr': f"{jr}",
            'rente_nominaal_maand': f"{r_pm}",
            'looptijd_mnd': looptijd,
            'grace_mnd': grace,
            'termijn_bedrag': term,
            'rente_effectief_jaar': f"{jr}",
            'rente_totaal': fmt_money(amort_idx.get(verstrekker, {}).get('rente_totaal', ZERO)),
            'restschuld_einde': fmt_money(amort_idx.get(verstrekker, {}).get('restschuld_einde', ZERO)),
        }
        leningen_ctx.append(entry)

    # Streams context for 60_pricing
    streams_ctx, _ = _build_streams(cfg.get('omzetstromen', []) or [], months)
    # First stream baseline for single-key outputs
    if cfg.get('omzetstromen'):
        s0 = cfg['omzetstromen'][0]
        prijs_basis = D(str(s0.get('prijs', 0)))
        var_eh = D(str(s0.get('var_kosten_per_eenheid', 0)))
        marge_eh = (prijs_basis - var_eh).quantize(CENT)
        vol_avg = _avg([D(str(v)) for v in s0.get('volume_pm') or [0] * len(months)])
    else:
        prijs_basis = ZERO
        var_eh = ZERO
        marge_eh = ZERO
        vol_avg = ZERO

    # Runway
    runway = _runway(model['cash_end'])

    # Compose context
    data: Dict[str, Any] = {
        # General
        'horizon_maanden': len(months),
        'start_maand': bedrijf.get('start_maand', first_month),
        'eerste_maand': first_month,
        'laatste_maand': last_month,
        'scenario': cfg.get('scenario', 'base'),
        'regime': bedrijf.get('rechtsvorm', 'IB'),
        'btw_model': btw_cfg.get('model', 'omzet_enkel'),
        'vat_period': cfg.get('vat_period', 'monthly'),
        'total_investering': fmt_money(total_invest),
        'total_financiering': fmt_money(total_financiering),
        'eigen_inbreng': fmt_money(eigen_inbreng),
        'eigen_inbreng_pct': _pct(eigen_inbreng, total_financiering if total_financiering > 0 else D('1')),
        'debt_equity_pct': _ratio_pct(debt_total, eigen_inbreng),
        'eerste_kas': fmt_money(eigen_inbreng + debt_total),
        'laagste_kas': fmt_money(lowest_cash),
        'kas_diepste_maand': lowest_cash_month,
        'kas_waarschuwing': 'OK' if lowest_cash >= ZERO else 'negatief',
        'kas_eind': fmt_money(model['cash_end'][last_month] if last_month else ZERO),
        'maandlast_dekking': f"{(model['coverage'].quantize(D('0.01')) if hasattr(model['coverage'],'quantize') else D(str(model['coverage'])).quantize(D('0.01')))}x",
        'dekking_verdict': 'voldoende' if model['coverage'] >= D('1') else 'onvoldoende',
        'irr_project': 'n.v.t.',
        'discount_rate_pct': cfg.get('assumpties', {}).get('discount_rate_pct', 10.0),
        'payback_maanden': 0,
        'runway_maanden_base': runway,
        'runway_maanden_worst': runway,  # placeholder until stress computed
        # Investeringen & financiering
        'inv_count': len(model.get('invest_items', []) or []),
        'inv_totaal': fmt_money(total_invest),
        'inv_avg_levensduur': int(round(sum(int(it['levensduur_mnd']) for it in (model.get('invest_items') or [])) / max(1, len(model.get('invest_items') or [])))),
        'inv_max_omschrijving': max((model.get('invest_items') or []), key=lambda it: D(str(it['bedrag'])) if it else ZERO).get('omschrijving') if model.get('invest_items') else '',
        'inv_max_bedrag': fmt_money(max([D(str(it['bedrag'])) for it in (model.get('invest_items') or [])] or [ZERO])),
        'fin_eigen': fmt_money(eigen_inbreng),
        'fin_schuld': fmt_money(debt_total),
        'lening_count': len(loans),
        'lening_avg_looptijd': int(round(sum(int(ln.get('looptijd_mnd', 0) or 0) for ln in loans) / max(1, len(loans)))),
        'lening_maandlasten_gem': '0.00',  # can compute from terms if needed
        'inv_pct_eigen': _pct(eigen_inbreng, total_invest if total_invest > 0 else D('1')),
        'inv_pct_schuld': _pct(debt_total, total_invest if total_invest > 0 else D('1')),
        'leningen': leningen_ctx,
        # Liquiditeit
        'kas_begin': fmt_money(model['cash_begin'][first_month] if first_month else ZERO),
        'kas_negatief_count': kas_negatief_count,
        'kas_negatief_verdict': kas_negatief_verdict,
        'btw_totaal': fmt_money(_sum([model['vat_payment'][m] for m in months])),
        'btw_max_bedrag': fmt_money(max([model['vat_payment'][m] for m in months] or [ZERO])),
        'btw_max_maand': max(months, key=lambda m: model['vat_payment'][m]) if months else '',
        'kas_buffer_norm': cfg.get('assumpties', {}).get('kas_buffer_norm', 0),
        'stress_laagste_kas': fmt_money(ZERO),
        'stress_runway_maanden': runway,
        # Exploitatie
        'omzet_totaal': fmt_money(omzet_totaal),
        'brutomarge_totaal': fmt_money(brutomarge_totaal),
        'brutomarge_pct': _pct(brutomarge_totaal, omzet_totaal if omzet_totaal > 0 else D('1')),
        'opex_totaal': fmt_money(opex_totaal),
        'afschrijving_totaal': fmt_money(afschrijving_totaal),
        'rente_totaal': fmt_money(rente_totaal),
        'resultaat_vb_totaal': fmt_money(resultaat_vb_totaal),
        'ebitda_pm_avg': fmt_money(ebitda_pm_avg),
        'nettowinst_pct': _pct(resultaat_vb_totaal, omzet_totaal if omzet_totaal > 0 else D('1')),
        'resultaat_best': fmt_money(best[1]),
        'maand_best': best[0],
        'resultaat_worst': fmt_money(worst[1]),
        'maand_worst': worst[0],
        'verlies_maanden': verlies_maanden,
        'verlies_verdict': verlies_verdict,
        'stress_result_avg': fmt_money(ZERO),
        'stress_verlies_maanden': 0,
        'stress_verdict': 'n.v.t.',
        'breakeven_omzet_pm': fmt_money(be_omzet),
        'veiligheidsmarge_pct': veiligheidsmarge_pct,
        # Qredits / maandlasten
        'dscr_min': f"{model['dscr_min'].quantize(D('0.01'))}",
        'dscr_maand_min': dscr_maand_min,
        'dscr_below_1_count': dscr_below_1_count,
        'dscr_verdict': dscr_verdict,
        'stress_dscr_min': '0.00',
        'stress_dscr_below_1_count': 0,
        'stress_dscr_verdict': 'n.v.t.',
        # Belastingen
        'btw_pct': btw_cfg.get('btw_pct', 21),
        'kor': btw_cfg.get('kor', False),
        'btw_vrij': btw_cfg.get('btw_vrij', False),
        'mkb_vrijstelling_pct': cfg.get('btw', {}).get('mkb_vrijstelling_pct', 0),
        'belastingdruk_pct': 0,
        # Section expects a list of objects containing the variable of the same name
        'indicatieve_heffing_jaar': [{'indicatieve_heffing_jaar': 0}],
        'stress_heffing_range': 'n.v.t.',
        'stress_kasimpact': fmt_money(ZERO),
        # Pricing (first stream baseline)
        'prijs_per_eenheid': f"{prijs_basis}",
        'eenheden_pm': f"{vol_avg}",
        'omzet_basis': fmt_money(omzet_basis_pm),
        'marge_per_eenheid': fmt_money(marge_eh),
        'cac_eur': 0,
        'payback_maanden': 0,
        'omzetstromen': streams_ctx,
        'prijs_min10': f"{(prijs_basis * D('0.90')).quantize(D('0.01'))}",
        'prijs_basis': f"{prijs_basis}",
        'prijs_plus10': f"{(prijs_basis * D('1.10')).quantize(D('0.01'))}",
        'omzet_min10': fmt_money((omzet_basis_pm * D('0.90')).quantize(CENT)),
        'omzet_plus10': fmt_money((omzet_basis_pm * D('1.10')).quantize(CENT)),
        'marge_min10': fmt_money(((marge_eh * D('0.90')) * vol_avg).quantize(CENT)),
        'marge_basis': fmt_money((marge_eh * vol_avg).quantize(CENT)),
        'marge_plus10': fmt_money(((marge_eh * D('1.10')) * vol_avg).quantize(CENT)),
        'runway_min10': runway,
        'runway_basis': runway,
        'runway_plus10': runway,
        'volume_basis': f"{vol_avg}",
        'volume_min10': f"{(vol_avg * D('0.90')).quantize(D('0.01'))}",
        'volume_plus15': f"{(vol_avg * D('1.15')).quantize(D('0.01'))}",
        'omzet_pricevol1': fmt_money(((prijs_basis * D('0.90')) * (vol_avg * D('1.15'))).quantize(CENT)),
        'omzet_pricevol2': fmt_money(((prijs_basis * D('1.10')) * (vol_avg * D('0.90'))).quantize(CENT)),
        'marge_pricevol1': fmt_money((((prijs_basis - var_eh) * D('0.90')) * (vol_avg * D('1.15'))).quantize(CENT)),
        'marge_pricevol2': fmt_money((((prijs_basis - var_eh) * D('1.10')) * (vol_avg * D('0.90'))).quantize(CENT)),
        # Break-even detailed
        'vaste_kosten_pm': fmt_money(_avg([opex_total[m] for m in months])),
        'variabele_kosten_per_eenheid': fmt_money(var_eh),
        'marge_pct': _pct(marge_eh, prijs_basis if prijs_basis > 0 else D('1')),
        'breakeven_eenheden_pm': f"{(be_omzet / (prijs_basis if prijs_basis > 0 else D('1'))).quantize(D('0.01'))}",
        'breakeven_omzet_plus10_opex': fmt_money((_avg([opex_total[m] for m in months]) * D('1.10') / ( ( (omzet_basis_pm - _avg([cogs[m] for m in months])) / (omzet_basis_pm if omzet_basis_pm>0 else D('1')) ) or D('1'))).quantize(CENT)),
        'breakeven_omzet_min10_marge': fmt_money((be_omzet * D('1.1111')).quantize(CENT)),
        'dekking_pct': _pct((vol_avg * marge_eh).quantize(CENT), _avg([opex_total[m] for m in months]) if months else D('1')),
        # Working capital extras (basic placeholders)
        'debiteuren_openstaand': fmt_money(ZERO),
        'crediteuren_openstaand': fmt_money(ZERO),
        'voorraad': fmt_money(ZERO),
        'borg_depositos': fmt_money(ZERO),
        'vooruitbetaald': fmt_money(ZERO),
        'deferred_revenue': fmt_money(ZERO),
        'werkkapitaal_totaal': fmt_money(ZERO),
        'stress_omzet_min30': '−30%',
        'stress_dso_plus30': '+30 dagen',
        'stress_opex_plus20': '+20%',
        'kas_na_stress_omzet': fmt_money(ZERO),
        'kas_na_stress_dso': fmt_money(ZERO),
        'kas_na_stress_opex': fmt_money(ZERO),
        # Unit economics extra defaults
        'ltv_eur': fmt_money(ZERO),
        'ltv_cac_ratio': 'n.v.t.',
        'ltv_cac_varplus10': 'n.v.t.',
        'ltv_cac_pricemin10': 'n.v.t.',
    }

    return data
