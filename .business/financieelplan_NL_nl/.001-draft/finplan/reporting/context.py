from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Tuple

from finplan.common.money import D, CENT, ZERO, money as fmt_money
from finplan.compute.loan import annuity_payment
from finplan.common.metrics import avg_dec, sum_dec


def build_streams(streams_cfg: List[Dict[str, Any]], months: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Decimal]]:
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
            'volume_pm': f"{avg_dec([D(str(v)) for v in vols]).quantize(D('0.01'))}",
            'omzet_pm': fmt_money((omzet_pm / D(str(len(months)))).quantize(CENT)),
            'marge_pm': fmt_money((marge_pm / D(str(len(months)))).quantize(CENT)),
        })
    return out_list, totals


def pricing_baseline(cfg: Dict[str, Any], months: List[str]) -> Tuple[Decimal, Decimal, Decimal, Decimal]:
    if cfg.get('omzetstromen'):
        s0 = cfg['omzetstromen'][0]
        prijs_basis = D(str(s0.get('prijs', 0)))
        var_eh = D(str(s0.get('var_kosten_per_eenheid', 0)))
        marge_eh = (prijs_basis - var_eh).quantize(CENT)
        vol_avg = avg_dec([D(str(v)) for v in s0.get('volume_pm') or [0] * len(months)])
    else:
        prijs_basis = ZERO
        var_eh = ZERO
        marge_eh = ZERO
        vol_avg = ZERO
    return prijs_basis, var_eh, marge_eh, vol_avg


def loan_contexts(loans: List[Dict[str, Any]], amort_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from collections import defaultdict
    idx: Dict[str, Dict[str, Decimal]] = defaultdict(lambda: {"rente_totaal": ZERO, "restschuld_einde": ZERO})
    for r in amort_rows or []:
        v = D(str(r.get('rente_pm', 0)))
        verstrekker = str(r.get('verstrekker', 'Onbekend'))
        idx[verstrekker]["rente_totaal"] += v
        idx[verstrekker]["restschuld_einde"] = D(str(r.get('restschuld', 0)))

    out: List[Dict[str, Any]] = []
    for ln in loans or []:
        verstrekker = ln.get('verstrekker', 'Onbekend')
        hoofdsom = D(str(ln.get('hoofdsom', 0))).quantize(CENT)
        jr = D(str(ln.get('rente_nominaal_jr', ln.get('rente_nominaal_jr_pct', 0))))
        r_pm = (jr / D('12')).quantize(D('0.01'))
        looptijd = int(ln.get('looptijd_mnd', 0) or 0)
        grace = int(ln.get('grace_mnd', 0) or 0)
        term = '0.00'
        if looptijd > 0:
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
            'rente_totaal': fmt_money(idx.get(verstrekker, {}).get('rente_totaal', ZERO)),
            'restschuld_einde': fmt_money(idx.get(verstrekker, {}).get('restschuld_einde', ZERO)),
        }
        out.append(entry)
    return out


def compute_pnl_aggregates(model: Dict[str, Any]) -> Dict[str, Any]:
    months = model['months']
    revenue = model['revenue']
    cogs = model['cogs']
    opex_total = model['opex_total']
    res_by_month: Dict[str, Decimal] = {}
    for m in months:
        res_by_month[m] = (revenue[m] - cogs[m] - opex_total[m] - model['depreciation'][m] - model['interest'][m]).quantize(CENT)
    return {
        'omzet_totaal': sum_dec([revenue[m] for m in months]),
        'cogs_totaal': sum_dec([cogs[m] for m in months]),
        'brutomarge_totaal': (sum_dec([revenue[m] for m in months]) - sum_dec([cogs[m] for m in months])).quantize(CENT),
        'opex_totaal': sum_dec([opex_total[m] for m in months]),
        'afschrijving_totaal': sum_dec([model['depreciation'][m] for m in months]),
        'rente_totaal': sum_dec([model['interest'][m] for m in months]),
        'resultaat_vb_by_month': res_by_month,
        'resultaat_vb_totaal': sum_dec(res_by_month.values()),
        'ebitda_pm_avg': avg_dec([model['ebitda'][m] for m in months]),
    }
