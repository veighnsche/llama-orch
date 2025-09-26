# Qredits / Maandlasten

Dit onderdeel beschrijft de maandelijkse verplichtingen voortvloeiend uit de opgenomen leningen. Het laat per lening zien wat de aflossingen en rentelasten zijn, en welke totale maandlast dit oplevert. Hiermee wordt inzichtelijk of de onderneming structureel aan haar schuldverplichtingen kan voldoen.

## Kernpunten

- Aantal leningen: {{lening_count}}
- Totale gemiddelde maandlast (rente + aflossing): {{termijn_bedrag}}
- Gemiddelde schuldendienst-dekking (EBITDA / schuldendienst): {{maandlast_dekking}}
- Laagste DSCR: {{dscr_min}} in maand {{dscr_maand_min}}
- Aantal maanden met DSCR < 1: {{dscr_below_1_count}} ({{dscr_verdict}})

## Specificatie per lening

{{#leningen}}
- **{{verstrekker}}**  
  Hoofdsom: {{hoofdsom}}  
  Looptijd: {{looptijd_mnd}} maanden  
  Nominale rente: {{rente_nominaal_jr}}% per jaar ({{rente_nominaal_maand}}% per maand)  
  Grace-periode: {{grace_mnd}} maanden  
  Maandlast: {{termijn_bedrag}}  
  Totale rentelasten over looptijd: {{rente_totaal}}  
  Effectieve jaarrente (APR, indicatief): {{rente_effectief_jaar}}%  
  Restschuld einde looptijd: {{restschuld_einde}}
{{/leningen}}

## Toelichting

- Maandlasten bestaan uit rente en aflossing volgens annuïtaire berekening (tenzij anders opgegeven).  
- Tijdens de grace-periode wordt alleen rente betaald; daarna start aflossing.  
- Maandlasten zijn volledig doorvertaald naar het kasstroomoverzicht in `20_liquiditeit_monthly.csv`.  
- Bij meerdere leningen is de gecombineerde schuldendienst zichtbaar in de samenvattende DSCR-cijfers.  
- Het risico op knelpunten wordt zichtbaar gemaakt via de DSCR-analyse.

## Stress-signaal (indicatief)

Bij een scenario met **EBITDA −30%** en gelijkblijvende maandlasten:
- Laagste DSCR: {{stress_dscr_min}}  
- Aantal maanden met DSCR < 1: {{stress_dscr_below_1_count}}  
- Implicatie: {{stress_dscr_verdict}}

## Detailtabel

Het volledige aflossingsschema per lening is opgenomen in `40_amortisatie.csv` met kolommen:

- `maand`
- `verstrekker`
- `rente_pm`
- `aflossing_pm`
- `restschuld`

---

_Alle bedragen zijn indicatief en gebaseerd op opgegeven aannames. Dit document vormt geen fiscale of juridische advisering._  
_Herleidbaarheid: alle cijfers zijn deterministisch afgeleid uit `dataset.v1.json`._
