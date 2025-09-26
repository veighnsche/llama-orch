# Investeringen & Financiering

Dit onderdeel beschrijft de benodigde investeringen voor de onderneming en de manier waarop deze gefinancierd worden. Het geeft inzicht in de omvang van de investeringen, de verhouding tussen eigen middelen en vreemd vermogen, en de belangrijkste kenmerken van leningen.

## Investeringen

- Aantal investeringsposten: 2
- Totaal investeringsbedrag: € 5.500,00
- Gemiddelde afschrijvingstermijn: 30 maanden
- Grootste investering: Website & branding (€ 3.000,00)

Elke investering wordt lineair afgeschreven over de opgegeven levensduur in maanden. Indien een startmaand is gespecificeerd, wordt prorata afgeschreven vanaf die maand.  
Dit geeft een realistisch beeld van jaarlijkse lasten en voorkomt dat kosten in één keer drukken op het resultaat.

_De volledige detailtabel is opgenomen in `10_investering.csv`._

## Financiering

- Eigen inbreng: € 5.000,00
- Vreemd vermogen: € 20.000,00
- Aantal leningen: 1
- Debt/Equity-verhouding: 80.00% / 20.00%
- Gemiddelde looptijd leningen: 48 maanden

### Specificatie leningen


- **Qredits**  
  Hoofdsom: € 20.000,00  
  Looptijd: 48 maanden  
  Rente: 9.0% per jaar (nominaal), 0.75% per maand  
  Grace-periode: 0 maanden  
  Termijnbedrag: 497.70 per maand  
  Effectieve jaarrente (APR): 9.0%  
  Totale rentelasten over looptijd: € 1.623,49  


De berekeningen zijn op annuïtaire basis uitgevoerd, tenzij anders aangegeven. Bij grace-periodes wordt rente tijdens de grace doorberekend, waarna volledige aflossing start.

_De volledige detailtabel is opgenomen in `10_financiering.csv`._

## Samenvattende ratio’s

- Totale investeringen gedekt door eigen inbreng: 90.91%  
- Totale investeringen gedekt door vreemd vermogen: 363.64%  
- Gemiddelde maandlasten alle leningen: 0.00  
- Schuldendienst-dekking (DSCR, gem.): -3.22x

---

_Alle bedragen zijn indicatief en gebaseerd op opgegeven aannames. Dit document vormt geen fiscale of juridische advisering._  
_Herleidbaarheid: alle cijfers zijn deterministisch afgeleid uit `dataset.v1.json`._
