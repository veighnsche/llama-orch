# TEMPLATE_KEYS (generated)

## 00_overview.md.tpl

Placeholder | Kind | Required | Description
---|---|---|---
breakeven_omzet_pm | var | yes | 
btw_model | var | yes | BTW-berekeningsmodel.
debt_equity_pct | var | yes | verhouding schuld/eigen vermogen.
dekking_verdict | var | yes | tekstsignaal (voldoende/onvoldoende).
discount_rate_pct | var | yes | disconteringsvoet voor NPV.
dscr_below_1_count | var | yes | 
dscr_maand_min | var | yes | 
dscr_min | var | yes | 
eerste_kas | var | yes | kas in maand 1 na inbreng + leningontvangst.
eerste_maand | var | yes | 
eigen_inbreng | var | yes | 
eigen_inbreng_pct | var | yes | 
horizon_maanden | var | yes | aantal maanden in de planning.
irr_project | var | yes | IRR van projectcashflows.
kas_diepste_maand | var | yes | 
kas_eind | var | yes | eindkas na horizon.
kas_waarschuwing | var | yes | korte tekstsignaal (OK/negatief).
laagste_kas | var | yes | 
laatste_maand | var | yes | 
maandlast_dekking | var | yes | Gem. DSCR (EBITDA/schuldendienst).
payback_maanden | var | yes | maanden tot cumulatieve cash ≥ 0.
regime | var | yes | fiscaal regime: IB of VPB.
runway_maanden_base | var | yes | 
runway_maanden_worst | var | yes | 
scenario | var | yes | gekozen scenario (base/best/worst).
start_maand | var | yes | 
total_financiering | var | yes | 
total_investering | var | yes | 
vat_period | var | yes | BTW-afdrachtfrequentie (maand/kwartaal).
veiligheidsmarge_pct | var | yes | 

## 05_qredits_aanvraag.md.tpl

Placeholder | Kind | Required | Description
---|---|---|---
dscr_min | var | yes | 
dscr_verdict | var | yes | 
fin_schuld | var | yes | 
inv_totaal | var | yes | 
lening_avg_looptijd | var | yes | 
lening_count | var | yes | 
runway_maanden_base | var | yes | 
stress_dscr_min | var | yes | 
stress_laagste_kas | var | yes | 
stress_runway_maanden | var | yes | 
termijn_bedrag | var | yes | annuïtaire maandlast.

## 10_investering_financiering.md.tpl

Placeholder | Kind | Required | Description
---|---|---|---
debt_equity_pct | var | yes | verhouding schuld/eigen vermogen.
fin_eigen | var | yes | 
fin_schuld | var | yes | 
grace_mnd | var | yes | maanden alleen rente.
hoofdsom | var | yes | geleend bedrag.
inv_avg_levensduur | var | yes | gemiddelde levensduur (mnd).
inv_count | var | yes | 
inv_max_bedrag | var | yes | 
inv_max_omschrijving | var | yes | 
inv_pct_eigen | var | yes | 
inv_pct_schuld | var | yes | 
inv_totaal | var | yes | 
lening_avg_looptijd | var | yes | 
lening_count | var | yes | 
lening_maandlasten_gem | var | yes | 
looptijd_mnd | var | yes | looptijd in maanden.
maandlast_dekking | var | yes | Gem. DSCR (EBITDA/schuldendienst).
rente_effectief_jaar | var | yes | effectieve jaarrente (APR, indicatief).
rente_nominaal_jr | var | yes | 
rente_nominaal_maand | var | yes | 
rente_totaal | var | yes | totale rente over looptijd.
termijn_bedrag | var | yes | annuïtaire maandlast.
verstrekker | var | yes | kredietverstrekker.
leningen | section | conditional | 

## 20_liquiditeit.md.tpl

Placeholder | Kind | Required | Description
---|---|---|---
btw_max_bedrag | var | yes | 
btw_max_maand | var | yes | 
btw_model | var | yes | BTW-berekeningsmodel.
btw_totaal | var | yes | som BTW-betalingen over horizon.
eerste_maand | var | yes | 
kas_begin | var | yes | 
kas_buffer_norm | var | yes | gewenste minimale buffer.
kas_diepste_maand | var | yes | 
kas_eind | var | yes | eindkas na horizon.
kas_negatief_count | var | yes | 
kas_negatief_verdict | var | yes | 
kas_waarschuwing | var | yes | korte tekstsignaal (OK/negatief).
laagste_kas | var | yes | 
laatste_maand | var | yes | 
runway_maanden_base | var | yes | 
stress_laagste_kas | var | yes | 
stress_runway_maanden | var | yes | 
vat_period | var | yes | BTW-afdrachtfrequentie (maand/kwartaal).

## 30_exploitatie.md.tpl

Placeholder | Kind | Required | Description
---|---|---|---
afschrijving_totaal | var | yes | 
breakeven_omzet_pm | var | yes | 
brutomarge_pct | var | yes | brutomarge% van omzet.
brutomarge_totaal | var | yes | 
ebitda_pm_avg | var | yes | gemiddelde EBITDA per maand.
horizon_maanden | var | yes | aantal maanden in de planning.
maand_best | var | yes | 
maand_worst | var | yes | 
nettowinst_pct | var | yes | (indicatief) nettomarge%.
omzet_totaal | var | yes | 
opex_totaal | var | yes | 
rente_totaal | var | yes | totale rente over looptijd.
resultaat_best | var | yes | 
resultaat_vb_totaal | var | yes | resultaat vóór belasting totaal.
resultaat_worst | var | yes | 
stress_result_avg | var | yes | 
stress_verdict | var | yes | 
stress_verlies_maanden | var | yes | 
veiligheidsmarge_pct | var | yes | 
verlies_maanden | var | yes | 
verlies_verdict | var | yes | 

## 40_qredits_maandlasten.md.tpl

Placeholder | Kind | Required | Description
---|---|---|---
dscr_below_1_count | var | yes | 
dscr_maand_min | var | yes | 
dscr_min | var | yes | 
dscr_verdict | var | yes | 
grace_mnd | var | yes | maanden alleen rente.
hoofdsom | var | yes | geleend bedrag.
lening_count | var | yes | 
looptijd_mnd | var | yes | looptijd in maanden.
maandlast_dekking | var | yes | Gem. DSCR (EBITDA/schuldendienst).
rente_effectief_jaar | var | yes | effectieve jaarrente (APR, indicatief).
rente_nominaal_jr | var | yes | 
rente_nominaal_maand | var | yes | 
rente_totaal | var | yes | totale rente over looptijd.
restschuld_einde | var | yes | restschuld aan het einde (meestal 0).
stress_dscr_below_1_count | var | yes | 
stress_dscr_min | var | yes | 
stress_dscr_verdict | var | yes | 
termijn_bedrag | var | yes | annuïtaire maandlast.
verstrekker | var | yes | kredietverstrekker.
leningen | section | conditional | 

## 50_belastingen.md.tpl

Placeholder | Kind | Required | Description
---|---|---|---
belastingdruk_pct | var | yes | heffing als % van resultaat v/bel (indicatief).
btw_model | var | yes | BTW-berekeningsmodel.
btw_pct | var | yes | 
btw_totaal | var | yes | som BTW-betalingen over horizon.
btw_vrij | var | yes | 
indicatieve_heffing_jaar | var | yes | totaal indicatieve heffing.
kor | var | yes | 
mkb_vrijstelling_pct | var | yes | 
regime | var | yes | fiscaal regime: IB of VPB.
stress_heffing_range | var | yes | 
stress_kasimpact | var | yes | 
vat_period | var | yes | BTW-afdrachtfrequentie (maand/kwartaal).
indicatieve_heffing_jaar | section | conditional | totaal indicatieve heffing.

## 60_pricing.md.tpl

Placeholder | Kind | Required | Description
---|---|---|---
cac_eur | var | yes | 
eenheden_pm | var | yes | 
marge_basis | var | yes | expanded from wildcard
marge_min10 | var | yes | expanded from wildcard
marge_per_eenheid | var | yes | 
marge_plus10 | var | yes | expanded from wildcard
marge_pm | var | yes | 
marge_pricevol1 | var | yes | 
marge_pricevol2 | var | yes | expanded from pair shorthand
naam | var | yes | 
omzet_basis | var | yes | 
omzet_min10 | var | yes | expanded from wildcard
omzet_plus10 | var | yes | expanded from wildcard
omzet_pm | var | yes | 
omzet_pricevol1 | var | yes | 
omzet_pricevol2 | var | yes | expanded from pair shorthand
payback_maanden | var | yes | maanden tot cumulatieve cash ≥ 0.
prijs | var | yes | 
prijs_basis | var | yes | 
prijs_min10 | var | yes | 
prijs_per_eenheid | var | yes | 
prijs_plus10 | var | yes | 
runway_basis | var | yes | expanded from wildcard
runway_min10 | var | yes | expanded from wildcard
runway_plus10 | var | yes | expanded from wildcard
volume_basis | var | yes | 
volume_min10 | var | yes | 
volume_plus15 | var | yes | 
volume_pm | var | yes | 
omzetstromen | section | conditional | 

## 70_breakeven.md.tpl

Placeholder | Kind | Required | Description
---|---|---|---
breakeven_eenheden_pm | var | yes | 
breakeven_omzet_min10_marge | var | yes | 
breakeven_omzet_plus10_opex | var | yes | 
breakeven_omzet_pm | var | yes | 
marge_pct | var | yes | 
marge_per_eenheid | var | yes | 
omzet_basis | var | yes | 
prijs_per_eenheid | var | yes | 
variabele_kosten_per_eenheid | var | yes | 
vaste_kosten_pm | var | yes | 
veiligheidsmarge_pct | var | yes | 

## 80_unit_economics.md.tpl

Placeholder | Kind | Required | Description
---|---|---|---
breakeven_eenheden_pm | var | yes | 
cac_eur | var | yes | 
contrib_basis | var | yes | 
contrib_min10 | var | yes | explicitly included common key
contrib_plus10 | var | yes | explicitly included common key
contribution_margin | var | yes | marge na variabele saleskosten/CAC.
dekking_pct | var | yes | explicitly included common key
eenheden_pm | var | yes | 
ltv_cac_pricemin10 | var | yes | explicitly included common key
ltv_cac_ratio | var | yes | 
ltv_cac_varplus10 | var | yes | explicitly included common key
ltv_eur | var | yes | 
marge_basis | var | yes | expanded from wildcard
marge_min10 | var | yes | expanded from wildcard
marge_pct | var | yes | 
marge_pct_basis | var | yes | explicitly included common key
marge_pct_min10 | var | yes | explicitly included common key
marge_pct_plus10 | var | yes | explicitly included common key
marge_per_eenheid | var | yes | 
marge_plus10 | var | yes | expanded from wildcard
marge_pricemin10 | var | yes | 
marge_varplus10 | var | yes | 
payback_maanden | var | yes | maanden tot cumulatieve cash ≥ 0.
prijs_basis | var | yes | 
prijs_min10 | var | yes | 
prijs_per_eenheid | var | yes | 
prijs_plus10 | var | yes | 
var_basis | var | yes | explicitly included common key
variabele_kosten_per_eenheid | var | yes | 
vaste_kosten_pm | var | yes | 

## 90_working_capital.md.tpl

Placeholder | Kind | Required | Description
---|---|---|---
borg_depositos | var | yes | 
crediteuren_openstaand | var | yes | 
debiteuren_openstaand | var | yes | 
kas_begin | var | yes | 
kas_na_stress_dso | var | yes | 
kas_na_stress_omzet | var | yes | 
kas_na_stress_opex | var | yes | 
stress_dso_plus30 | var | yes | explicitly included common key
stress_omzet_min30 | var | yes | explicitly included common key
stress_opex_plus20 | var | yes | explicitly included common key
voorraad | var | yes | 
werkkapitaal_totaal | var | yes | netto WC behoefte.

## zz_schema.md.tpl

Placeholder | Kind | Required | Description
---|---|---|---
in | var | yes | 
out | var | yes | 
mapping | section | conditional | 
