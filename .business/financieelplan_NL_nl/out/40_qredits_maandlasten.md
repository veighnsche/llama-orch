# Qredits / Maandlasten

Dit onderdeel beschrijft de maandelijkse verplichtingen voortvloeiend uit de opgenomen leningen. Het laat per lening zien wat de aflossingen en rentelasten zijn, en welke totale maandlast dit oplevert. Hiermee wordt inzichtelijk of de onderneming structureel aan haar schuldverplichtingen kan voldoen.

## Kernpunten

- Aantal leningen: 1
- Totale gemiddelde maandlast (rente + aflossing): € 339,26
- Gemiddelde schuldendienst-dekking (EBITDA / schuldendienst): 5.08x
- Laagste DSCR: -10.29 in maand 2026-01
- Aantal maanden met DSCR < 1: 2 (onvoldoende)

## Specificatie per lening


- **Qredits**  
  Hoofdsom: € 20.000,00  
  Looptijd: 60 maanden  
  Nominale rente: 7.0% per jaar (0.58% per maand)  
  Grace-periode: 3 maanden  
  Maandlast: 413.45  
  Totale rentelasten over looptijd: € 1.336,82  
  Effectieve jaarrente (APR, indicatief): 7.0%  
  Restschuld einde looptijd: € 17.265,76


## Toelichting

- Maandlasten bestaan uit rente en aflossing volgens annuïtaire berekening (tenzij anders opgegeven).  
- Tijdens de grace-periode wordt alleen rente betaald; daarna start aflossing.  
- Maandlasten zijn volledig doorvertaald naar het kasstroomoverzicht in `20_liquiditeit_monthly.csv`.  
- Bij meerdere leningen is de gecombineerde schuldendienst zichtbaar in de samenvattende DSCR-cijfers.  
- Het risico op knelpunten wordt zichtbaar gemaakt via de DSCR-analyse.

## Stress-signaal (indicatief)

Bij een scenario met **EBITDA −30%** en gelijkblijvende maandlasten:
- Laagste DSCR: 0.00  
- Aantal maanden met DSCR < 1: 0  
- Implicatie: n.v.t.

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
