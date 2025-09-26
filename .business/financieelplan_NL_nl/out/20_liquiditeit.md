# Liquiditeit

Dit kasstroomoverzicht geeft inzicht in de ontvangsten en uitgaven per maand. Hierin zijn verschuivingen verwerkt door betalingstermijnen van debiteuren (DSO) en crediteuren (DPO), en de afdracht van BTW. Het laat zien of de onderneming gedurende de looptijd over voldoende liquiditeit beschikt om verplichtingen tijdig na te komen.

## Kerncijfers

- Begin kaspositie: € 0,00 in maand 2026-01
- Eind kaspositie: € 1.219,94 in maand 2026-12
- Laagste kasstand: -€ 151,36 in maand 2026-06 (negatief)
- Runway (maanden kas ≥ 0): 6 in het basisscenario
- Aantal maanden negatief: 1 (negatief)

## BTW-afdracht

- BTW-model: omzet_enkel
- Afdrachtfrequentie: monthly
- Totale BTW-afdrachten over horizon: € 21.609,00
- Grootste afdracht in één maand: € 2.058,00 (2026-06)

BTW wordt berekend over de omzet minus aftrekbare kosten (afhankelijk van het gekozen model). De afdracht vindt plaats in de laatste maand van de periode (maandelijks of per kwartaal).

## Toelichting

- **Inkomsten** omvatten kasontvangsten uit omzet (inclusief DSO-verschuiving) en overige ontvangsten zoals subsidies of eigen inbreng.  
- **Uitgaven** omvatten betalingen voor inkoop (COGS, inclusief DPO), OPEX, rente, aflossingen en BTW-afdracht.  
- **Negatieve kasposities** worden expliciet zichtbaar gemaakt in de maandtabel en zijn voorzien van een waarschuwing.  
- Een kasbuffer van minimaal 2500 wordt aanbevolen; de feitelijke laagste stand is -€ 151,36.

## Stress-case signaal

Indicatieve impact bij een neerwaarts scenario (omzet −30%, DSO +30 dagen, OPEX +10%):
- Laagste kasstand: € 0,00  
- Runway: 6 maanden  

Dit geeft een indicatie van de weerbaarheid bij tegenvallende resultaten.

_De detailtabel is opgenomen in `20_liquiditeit_monthly.csv`._

---

_Alle bedragen zijn indicatief en gebaseerd op opgegeven aannames. Dit document vormt geen fiscale of juridische advisering._  
_Herleidbaarheid: alle cijfers zijn deterministisch afgeleid uit `dataset.v1.json`._
