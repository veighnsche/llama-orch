"""IO helpers: read YAML/JSON, write CSV/MD, report assembly (Phase 2)
Note: Template rendering will be wired in Phase 3. For now we keep fixed prose
writers similar to the pre-refactor implementation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from .money import D, CENT, money as fmt_money, pct_from_ratio as fmt_pct

# Optional YAML support; runtime must not require it.
try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


# ---------- Filesystem ----------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    ensure_dir(path.parent)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding='utf-8')


# ---------- Config loading ----------

def load_input(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() in ('.yml', '.yaml') and yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError('Top-level config must be a mapping')
    return data


# ---------- Report writers (Phase 2 prose; will move to templates in Phase 3) ----------

def overview_md(model: Dict[str, Any]) -> str:
    months = model['months']
    first_cash = (model['eigen_inbreng'] + sum((D(str(ln['hoofdsom'])) for ln in model['config'].get('financiering', {}).get('leningen', []) or []), D('0'))).quantize(CENT) if months else D('0')
    lowest_cash = model['lowest_cash']
    lowest_cash_month = model['lowest_cash_month'] or (months[0] if months else '')
    coverage = model['coverage']

    bel = model['config'].get('belastingen', {})
    vat_model = bel.get('btw_model', 'omzet_enkel')
    regime = bel.get('regime', 'IB')

    return (
        f"# Overzicht\n\n"
        f"- Start kas (eigen inbreng + leningen in maand 1): {fmt_money(first_cash)}\n"
        f"- Laagste kasstand: {fmt_money(lowest_cash)} (maand {lowest_cash_month})\n"
        f"- Maandlast-dekking (gem. EBITDA / gem. schuldendienst): {(coverage.quantize(D('0.01')) if hasattr(coverage,'quantize') else D(str(coverage)).quantize(D('0.01')))}x\n"
        f"- Belastingregime: {regime}, BTW-model: {vat_model}\n"
        f"_Indicatief; geen fiscale advisering._\n"
    )


def inv_fin_md(model: Dict[str, Any]) -> str:
    total_invest = model['total_invest']
    eigen = model['eigen_inbreng']
    debt = sum((D(str(ln['hoofdsom'])) for ln in model['config'].get('financiering', {}).get('leningen', []) or []), D('0'))
    debt_pct = (debt / (eigen + debt) * D('100')) if (eigen + debt) > 0 else D('0')
    equity_pct = D('100') - debt_pct if (eigen + debt) > 0 else D('0')
    return (
        f"# Investeringen & Financiering\n\n"
        f"- Totale investering: {fmt_money(total_invest)}\n"
        f"- Eigen inbreng: {fmt_money(eigen)}\n"
        f"- Vreemd vermogen: {fmt_money(debt)}\n"
        f"- Debt/Equity: {fmt_pct(debt_pct/ D('100'))} / {fmt_pct(equity_pct/ D('100'))}\n"
    )


def exploitatie_md() -> str:
    return (
        f"# Exploitatie (maandelijks)\n\n"
        f"Toelichting: omzet → COGS → marge → OPEX → afschrijving → rente → resultaat (v/bel).\n"
    )


def liquiditeit_md() -> str:
    return (
        f"# Liquiditeit\n\n"
        f"Kasstroomtabel met DSO/DPO verschuivingen en BTW-afdracht per periode.\n"
    )


def qredits_md() -> str:
    return (
        f"# Qredits / Maandlasten\n\n"
        f"Volledig aflosschema; beoordeling van maandlast-dekking in Overzicht.\n"
    )


def tax_md(model: Dict[str, Any]) -> str:
    bel = model['config'].get('belastingen', {})
    regime = bel.get('regime', 'IB')
    kor = bool(bel.get('kor', False))
    btw_vrij = bool(bel.get('btw_vrij', False))
    btw_note = 'Geen BTW-afdracht (KOR of vrijgesteld).' if (kor or btw_vrij) else 'BTW volgens gekozen model.'
    return (
        f"# Belastingen (indicatief)\n\n"
        f"Regime: {regime}. {btw_note} Geen fiscale advisering.\n"
    )


def schema_md() -> str:
    return (
        "# Schema (mapping input → output)\n\n"
        "- bedrijf.start_maand → tijdlijn (maanden)\n"
        "- investeringen[*] → afschrijving per maand; 10_investering.csv\n"
        "- financiering.{eigen_inbreng,leningen[*]} → kasinstroom m1; 40_amortisatie.csv\n"
        "- omzetmodel.{omzet_pm,cogs_pct,opex_pm,*} → exploitatie & liquiditeit\n"
        "- belastingen.{btw_pct,btw_vrij,kor,btw_model,vat_period} → BTW-afdracht\n"
    )


def write_reports(out_dir: Path, model: Dict[str, Any]) -> None:
    from .money import D, CENT
    ensure_dir(out_dir)

    # 00_overview.md
    write_text(out_dir / '00_overview.md', overview_md(model))

    # 10_investering_financiering.md + CSVs
    write_text(out_dir / '10_investering_financiering.md', inv_fin_md(model))
    inv_rows = [[it['omschrijving'], f"{it['levensduur_mnd']}", f"{it['start_maand']}", f"{D(str(it['afschrijving_pm'])).quantize(CENT)}", f"{D(str(it['bedrag'])).quantize(CENT)}"] for it in model['invest_items']]
    write_csv(out_dir / '10_investering.csv', ['omschrijving','levensduur_mnd','start_maand','afschrijving_pm','bedrag'], inv_rows)
    fin_rows = []
    fin = model['config'].get('financiering', {})
    fin_rows.append(['eigen_inbreng', f"{model['eigen_inbreng'].quantize(CENT)}"])
    for ln in fin.get('leningen', []) or []:
        fin_rows.append([ln.get('verstrekker','Onbekend'), f"{D(str(ln['hoofdsom'])).quantize(CENT)}", f"{ln.get('rente_nominaal_jr_pct')}%", ln.get('looptijd_mnd'), ln.get('grace_mnd',0)])
    write_csv(out_dir / '10_financiering.csv', ['bron','bedrag','rente_nominaal_jr_pct','looptijd_mnd','grace_mnd'], fin_rows)

    # 20_liquiditeit.md + CSV
    write_text(out_dir / '20_liquiditeit.md', liquiditeit_md())
    liq_rows = []
    for ym in model['months']:
        liq_rows.append([
            ym,
            f"{model['cash_begin'][ym].quantize(CENT)}",
            f"{model['cash_in_revenue'][ym].quantize(CENT)}",
            f"{model['inflow_other'][ym].quantize(CENT)}",
            f"{model['cash_out_cogs'][ym].quantize(CENT)}",
            f"{model['opex_total'][ym].quantize(CENT)}",
            f"{model['vat_payment'][ym].quantize(CENT)}",
            f"{model['interest'][ym].quantize(CENT)}",
            f"{model['principal'][ym].quantize(CENT)}",
            f"{model['cash_end'][ym].quantize(CENT)}",
        ])
    write_csv(out_dir / '20_liquiditeit_monthly.csv', ['maand','begin_kas','in_omzet','in_overig','uit_cogs','uit_opex','uit_btw','uit_rente','uit_aflossing','eind_kas'], liq_rows)

    # 30_exploitatie.md + CSV
    write_text(out_dir / '30_exploitatie.md', exploitatie_md())
    opex_keys = sorted(list(model.get('opex_lines', {}).keys()))
    exp_header = ['maand','omzet','cogs','marge'] + [f"opex_{k}" for k in opex_keys] + ['opex_totaal','afschrijving','rente','resultaat_vb']
    exp_rows = []
    for ym in model['months']:
        row = [
            ym,
            f"{model['revenue'][ym].quantize(CENT)}",
            f"{model['cogs'][ym].quantize(CENT)}",
            f"{(model['revenue'][ym]-model['cogs'][ym]).quantize(CENT)}",
        ]
        for k in opex_keys:
            row.append(f"{model['opex_lines'][k][ym].quantize(CENT)}")
        row.extend([
            f"{model['opex_total'][ym].quantize(CENT)}",
            f"{model['depreciation'][ym].quantize(CENT)}",
            f"{model['interest'][ym].quantize(CENT)}",
            f"{(model['revenue'][ym]-model['cogs'][ym]-model['opex_total'][ym]-model['depreciation'][ym]-model['interest'][ym]).quantize(CENT)}",
        ])
        exp_rows.append(row)
    write_csv(out_dir / '30_exploitatie.csv', exp_header, exp_rows)

    # 40_qredits_maandlasten.md + amortisatie.csv
    write_text(out_dir / '40_qredits_maandlasten.md', qredits_md())
    amort = model['amort_rows']
    write_csv(out_dir / '40_amortisatie.csv', ['maand','verstrekker','rente_pm','aflossing_pm','restschuld'], [
        [r['maand'], r['verstrekker'], f"{D(str(r['rente_pm'])).quantize(CENT)}", f"{D(str(r['aflossing_pm'])).quantize(CENT)}", f"{D(str(r['restschuld'])).quantize(CENT)}"] for r in amort
    ])

    # 50_belastingen.md + tax.csv (indicatief)
    write_text(out_dir / '50_belastingen.md', tax_md(model))
    bel = model['config'].get('belastingen', {})
    tax_rows = [[model['config'].get('belastingen', {}).get('regime','IB'), bel.get('btw_pct',21), bel.get('btw_model','omzet_enkel'), bel.get('kor', False), bel.get('btw_vrij', False)]]
    write_csv(out_dir / '50_tax.csv', ['regime','btw_pct','btw_model','kor','btw_vrij'], tax_rows)

    # zz_schema.md
    write_text(out_dir / 'zz_schema.md', schema_md())
