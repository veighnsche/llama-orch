# Liquiditeit

Dit kasstroomoverzicht geeft inzicht in de ontvangsten en uitgaven per maand. Hierin zijn verschuivingen verwerkt door betalingstermijnen van debiteuren (DSO) en crediteuren (DPO), en de afdracht van BTW. Het laat zien of de onderneming gedurende de looptijd over voldoende liquiditeit beschikt om verplichtingen tijdig na te komen.

## Kerncijfers

- Begin kaspositie: {{kas_begin}} in maand {{eerste_maand}}
- Eind kaspositie: {{kas_eind}} in maand {{laatste_maand}}
- Laagste kasstand: {{laagste_kas}} in maand {{kas_diepste_maand}} ({{kas_waarschuwing}})
- Runway (maanden kas ≥ 0): {{runway_maanden_base}} in het basisscenario
- Aantal maanden negatief: {{kas_negatief_count}} ({{kas_negatief_verdict}})

## BTW-afdracht

- BTW-model: {{btw_model}}
- Afdrachtfrequentie: {{vat_period}}
- Totale BTW-afdrachten over horizon: {{btw_totaal}}
- Grootste afdracht in één maand: {{btw_max_bedrag}} ({{btw_max_maand}})

BTW wordt berekend over de omzet minus aftrekbare kosten (afhankelijk van het gekozen model). De afdracht vindt plaats in de laatste maand van de periode (maandelijks of per kwartaal).

## Toelichting

- **Inkomsten** omvatten kasontvangsten uit omzet (inclusief DSO-verschuiving) en overige ontvangsten zoals subsidies of eigen inbreng.  
- **Uitgaven** omvatten betalingen voor inkoop (COGS, inclusief DPO), OPEX, rente, aflossingen en BTW-afdracht.  
- **Negatieve kasposities** worden expliciet zichtbaar gemaakt in de maandtabel en zijn voorzien van een waarschuwing.  
- Een kasbuffer van minimaal {{kas_buffer_norm}} wordt aanbevolen; de feitelijke laagste stand is {{laagste_kas}}.

## Stress-case signaal

Indicatieve impact bij een neerwaarts scenario (omzet −30%, DSO +30 dagen, OPEX +10%):
- Laagste kasstand: {{stress_laagste_kas}}  
- Runway: {{stress_runway_maanden}} maanden  

Dit geeft een indicatie van de weerbaarheid bij tegenvallende resultaten.

_De detailtabel is opgenomen in `20_liquiditeit_monthly.csv`._

---

_Alle bedragen zijn indicatief en gebaseerd op opgegeven aannames. Dit document vormt geen fiscale of juridische advisering._  
_Herleidbaarheid: alle cijfers zijn deterministisch afgeleid uit `dataset.v1.json`._
