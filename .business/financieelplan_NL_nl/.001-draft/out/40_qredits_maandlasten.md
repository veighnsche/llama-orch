# Qredits / Maandlasten

Dit onderdeel beschrijft de maandelijkse verplichtingen voortvloeiend uit de opgenomen leningen. Het laat per lening zien wat de aflossingen en rentelasten zijn, en welke totale maandlast dit oplevert. Hiermee wordt inzichtelijk of de onderneming structureel aan haar schuldverplichtingen kan voldoen.

## Kernpunten

- Aantal leningen: 1
- Totale gemiddelde maandlast (rente + aflossing): € 530,17
- Gemiddelde schuldendienst-dekking (EBITDA / schuldendienst): -0.10x
- Laagste DSCR: -10.22 in maand 2025-10
- Aantal maanden met DSCR < 1: 7 (onvoldoende)

## Specificatie per lening


- **Qredits**  
  Hoofdsom: € 30.000,00  
  Looptijd: 48 maanden  
  Nominale rente: 9.0% per jaar (0.75% per maand)  
  Grace-periode: 6 maanden  
  Maandlast: 835.34  
  Totale rentelasten over looptijd: € 2.630,64  
  Effectieve jaarrente (APR, indicatief): 9.0%  
  Restschuld einde looptijd: € 26.268,60


## Toelichting

- Maandlasten bestaan uit rente en aflossing volgens annuïtaire berekening (tenzij anders opgegeven).  
- Tijdens de grace-periode wordt alleen rente betaald; daarna start aflossing.  
- Maandlasten zijn volledig doorvertaald naar het kasstroomoverzicht in `20_liquiditeit_monthly.csv`.  
- Bij meerdere leningen is de gecombineerde schuldendienst zichtbaar in de samenvattende DSCR-cijfers.  
- Het risico op knelpunten wordt zichtbaar gemaakt via de DSCR-analyse.

## Stress-signaal (indicatief)

Bij een scenario met **EBITDA −30%** en gelijkblijvende maandlasten:
- Laagste DSCR: -13.82  
- Aantal maanden met DSCR < 1: 12  
- Implicatie: onvoldoende

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
