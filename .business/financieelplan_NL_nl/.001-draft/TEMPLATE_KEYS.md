# TEMPLATE_KEYS — Allowed placeholders per template

This file enumerates allowed placeholders for each template. The renderer must fail fast if a template contains unknown placeholders, and if required placeholders are missing.

## templates/00_overview.md.tpl

Required:
- start_maand
- scenario
- vat_period
- total_investering
- total_financiering
- eigen_inbreng
- eigen_inbreng_pct
- debt_equity_pct
- eerste_kas
- laagste_kas
- maandlast_dekking
- horizon_maanden
- eerste_maand
- laatste_maand
- regime
- btw_model

Optional: none

## templates/10_investering_financiering.md.tpl

Required:
- inv_totaal
- inv_count
- fin_eigen
- fin_schuld
- lening_count

Optional (within {{#leningen}} ... {{/leningen}} block):
- leningen (list)
- verstrekker
- hoofdsom
- rente_nominaal_jr
- looptijd_mnd
- grace_mnd

## templates/20_liquiditeit.md.tpl

Required:
- horizon_maanden
- eerste_maand
- laatste_maand
- start_maand
- btw_model
- vat_period
- kas_begin
- kas_eind
- laagste_kas
- kas_diepste_maand

Optional:
- kas_waarschuwing

## templates/30_exploitatie.md.tpl

Required:
- omzet_totaal
- brutomarge_totaal
- opex_totaal
- afschrijving_totaal
- rente_totaal
- resultaat_vb_totaal

Optional: none

## templates/40_qredits_maandlasten.md.tpl

Required (within {{#leningen}} ... {{/leningen}} block):
- leningen (list)
- verstrekker
- termijn_bedrag
- rente_nominaal_jr
- rente_nominaal_maand
- rente_effectief_jaar
- looptijd_mnd
- grace_mnd

Optional: none

## templates/50_belastingen.md.tpl

Required:
- regime
- btw_pct
- btw_model
- vat_period
- kor
- btw_vrij
- mkb_vrijstelling_pct
- aftrek_zelfstandig
- aftrek_starters

Optional:
- indicatieve_heffing_jaar

## templates/zz_schema.md.tpl

Required:
- mapping_block

Optional: none
