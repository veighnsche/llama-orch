# Pricing & Omzetgevoeligheid

Dit onderdeel geeft inzicht in de gekozen prijsstrategie, de onderbouwing van tarieven, en de gevoeligheid van omzet bij prijswijzigingen. Het laat zien hoe robuust het bedrijfsmodel is bij veranderingen in prijs of volume.

## Kernpunten (basis-scenario)

- Huidige verkoopprijs per eenheid: 49.0
- Verkochte eenheden per maand: 175.00
- Omzet per maand: € 8.575,00
- Brutomarge per eenheid: € 39,00
- Customer Acquisition Cost (CAC): 0  
- Payback-tijd (maanden tot terugverdienen CAC): 0  

## Segmentatie (optioneel)


- **API** — Prijs: 49.0, Volume: 175.00, Omzet p/m: € 8.575,00, Marge: € 6.825,00


Dit maakt duidelijk hoe omzet verdeeld is over verschillende product-/dienstcategorieën.

## Gevoeligheidsanalyse (prijsvariatie)

Onderstaande tabel toont de impact van prijsvariaties op omzet en marge bij gelijkblijvend volume:

| Prijsvariant | Prijs per eenheid | Omzet per maand | Brutomarge per maand | Runway (maanden) |
|--------------|------------------:|----------------:|---------------------:|-----------------:|
| -10%         | 44.10   | € 7.717,50 | € 6.142,50      | 6 |
| Basis        | 49.0   | € 8.575,00 | € 6.825,00      | 6 |
| +10%         | 53.90  | € 9.432,50| € 7.507,50     | 6|

## Gevoeligheidsanalyse (prijs-volume elasticiteit)

Indien prijswijzigingen leiden tot volumeverandering, geeft onderstaande tabel de gecombineerde effecten:

| Scenario        | Prijs | Volume p/m | Omzet p/m | Brutomarge p/m |
|-----------------|------:|-----------:|----------:|---------------:|
| -10% prijs, +15% volume | 44.10 | 201.25 | € 8.875,13 | € 7.063,88 |
| Basis           | 49.0 | 175.00 | € 8.575,00 | € 6.825,00 |
| +10% prijs, -10% volume | 53.90 | 157.50 | € 8.489,25 | € 6.756,75 |

## Toelichting

- **Prijsstrategie:** gebaseerd op marktvergelijking, kostprijs per eenheid, en gewenste marge.  
- **Elasticiteit:** laat zien hoe omzet/marge bewegen bij prijs- en volume-aanpassingen.  
- **Runway-impact:** bij lagere prijzen kan de kas sneller negatief worden; dit is in de gevoeligheidstabel zichtbaar.  
- **Koppeling naar break-even:** zie `70_breakeven.md` voor de minimale omzet per maand die nodig is om alle vaste lasten te dekken.  

---

_Disclaimer: Dit overzicht is indicatief. Werkelijke prijzen, volumes en klantgedrag kunnen afwijken._  
_Herleidbaarheid: alle cijfers zijn deterministisch afgeleid uit `dataset.v1.json`._
