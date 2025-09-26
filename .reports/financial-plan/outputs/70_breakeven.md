# 70 · Break-even Analyse (v1 — deterministisch)

Brondata: `../dataset.v1.json`
Disclaimers: Indicatief; prijspeil 2025; excl. BTW; geen fiscale advisering.

## 1) Vaste lasten (OPEX) — per maand

Payroll (bruto):
- Founder: €3.500
- Contractor 0,5 FTE (pro rata): €1.750
- Subtotaal bruto: €5.250

Opslagen op payroll:
- Werkgeverslasten 25% × €5.250 = €1.312,50
- Vakantiegeld 8% × €5.250 = €420,00
- Payroll totaal: €5.250 + €1.312,50 + €420,00 = **€6.982,50**

Overige vaste kosten:
- Marketing €500,00 • SaaS €150,00 • Accounting €125,00 • Verzekeringen €75,00 • Huur €0,00 • Overig €100,00
- Overige vaste kosten totaal: **€950,00**

Totaal vaste OPEX per maand:
- **€6.982,50 + €950,00 = €7.932,50**

## 2) Brutowinstmarge (GM) aannames

- Public Tap (API, prepaid): **55% GM** (COGS 45%)
- Private Tap (dedicated cloud): **40% GM** (doel)

## 3) Break-even omzet per maand (EBITDA ~ 0)

Formule: `Omzet_break_even = Vaste_OPEX / GM_blended`

Scenario’s:
- 100% Public Tap (GM 55%): 7.932,50 / 0,55 = **€14.423,64**
- 100% Private Tap (GM 40%): 7.932,50 / 0,40 = **€19.831,25**
- 50/50 mix (GM (0,55+0,40)/2 = 0,475): 7.932,50 / 0,475 = **€16.698,95**

Opmerkingen:
- Werkelijke mix en prijzen beïnvloeden GM_blended; bovenstaande is deterministisch t.o.v. de dataset.
- CAPEX-afschrijving is niet als cash‑out meegenomen in EBITDA; zie 80_unit_economics.md voor context.
