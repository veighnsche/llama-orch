# Investeringen & Financiering

Dit onderdeel beschrijft de benodigde investeringen voor de onderneming en de manier waarop deze gefinancierd worden. Het geeft inzicht in de omvang van de investeringen, de verhouding tussen eigen middelen en vreemd vermogen, en de belangrijkste kenmerken van leningen.

## Investeringen

- Aantal investeringsposten: {{inv_count}}
- Totaal investeringsbedrag: {{inv_totaal}}
- Gemiddelde afschrijvingstermijn: {{inv_avg_levensduur}} maanden
- Grootste investering: {{inv_max_omschrijving}} ({{inv_max_bedrag}})

Elke investering wordt lineair afgeschreven over de opgegeven levensduur in maanden. Indien een startmaand is gespecificeerd, wordt prorata afgeschreven vanaf die maand.  
Dit geeft een realistisch beeld van jaarlijkse lasten en voorkomt dat kosten in één keer drukken op het resultaat.

_De volledige detailtabel is opgenomen in `10_investering.csv`._

## Financiering

- Eigen inbreng: {{fin_eigen}}
- Vreemd vermogen: {{fin_schuld}}
- Aantal leningen: {{lening_count}}
- Debt/Equity-verhouding: {{debt_equity_pct}}
- Gemiddelde looptijd leningen: {{lening_avg_looptijd}} maanden

### Specificatie leningen

{{#leningen}}
- **{{verstrekker}}**  
  Hoofdsom: {{hoofdsom}}  
  Looptijd: {{looptijd_mnd}} maanden  
  Rente: {{rente_nominaal_jr}}% per jaar (nominaal), {{rente_nominaal_maand}}% per maand  
  Grace-periode: {{grace_mnd}} maanden  
  Termijnbedrag: {{termijn_bedrag}} per maand  
  Effectieve jaarrente (APR): {{rente_effectief_jaar}}%  
  Totale rentelasten over looptijd: {{rente_totaal}}  
{{/leningen}}

De berekeningen zijn op annuïtaire basis uitgevoerd, tenzij anders aangegeven. Bij grace-periodes wordt rente tijdens de grace doorberekend, waarna volledige aflossing start.

_De volledige detailtabel is opgenomen in `10_financiering.csv`._

## Samenvattende ratio’s

- Totale investeringen gedekt door eigen inbreng: {{inv_pct_eigen}}  
- Totale investeringen gedekt door vreemd vermogen: {{inv_pct_schuld}}  
- Gemiddelde maandlasten alle leningen: {{lening_maandlasten_gem}}  
- Schuldendienst-dekking (DSCR, gem.): {{maandlast_dekking}}

---

_Alle bedragen zijn indicatief en gebaseerd op opgegeven aannames. Dit document vormt geen fiscale of juridische advisering._  
_Herleidbaarheid: alle cijfers zijn deterministisch afgeleid uit `dataset.v1.json`._
