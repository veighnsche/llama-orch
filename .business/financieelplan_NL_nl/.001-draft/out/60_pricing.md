# Pricing & Omzetgevoeligheid

Dit onderdeel geeft inzicht in de gekozen prijsstrategie, de onderbouwing van tarieven, en de gevoeligheid van omzet bij prijswijzigingen. Het laat zien hoe robuust het bedrijfsmodel is bij veranderingen in prijs of volume.

## Kernpunten (basis-scenario)

- Huidige verkoopprijs per eenheid (bijv. EUR per 1M tokens): 1.2
- Verkochte eenheden per maand (bijv. 1M tokens): 8416.67
- Omzet per maand: € 10.100,00
- Brutomarge per eenheid: € 0,35
- Customer Acquisition Cost (CAC): 0  
- Payback-tijd (maanden tot terugverdienen CAC): 0  

## Segmentatie (optioneel)


- **Public Tap (credits)** — Prijs: 1.2, Volume: 8416.67, Omzet p/m: € 10.100,00, Marge: € 2.945,83


Dit maakt duidelijk hoe omzet verdeeld is over verschillende product-/dienstcategorieën.

## Gevoeligheidsanalyse (prijsvariatie)

Onderstaande tabel toont de impact van prijsvariaties op omzet en marge bij gelijkblijvend volume (eenheid = per 1M tokens bij credits-first):

| Prijsvariant | Prijs per eenheid | Omzet per maand | Brutomarge per maand | Runway (maanden) |
|--------------|------------------:|----------------:|---------------------:|-----------------:|
| -10%         | 1.08   | € 9.090,00 | € 2.651,25      | 10 |
| Basis        | 1.2   | € 10.100,00 | € 2.945,83      | 10 |
| +10%         | 1.32  | € 11.110,00| € 3.240,42     | 10|

## Gevoeligheidsanalyse (prijs-volume elasticiteit)

Indien prijswijzigingen leiden tot volumeverandering, geeft onderstaande tabel de gecombineerde effecten (eenheid = per 1M tokens bij credits-first):

| Scenario        | Prijs | Volume p/m | Omzet p/m | Brutomarge p/m |
|-----------------|------:|-----------:|----------:|---------------:|
| -10% prijs, +15% volume | 1.08 | 9679.17 | € 10.453,50 | € 3.048,94 |
| Basis           | 1.2 | 8416.67 | € 10.100,00 | € 2.945,83 |
| +10% prijs, -10% volume | 1.32 | 7575.00 | € 9.999,00 | € 2.916,38 |

## Toelichting

- **Prijsstrategie:** gebaseerd op marktvergelijking, kostprijs per eenheid, en gewenste marge.  
- **Elasticiteit:** laat zien hoe omzet/marge bewegen bij prijs- en volume-aanpassingen.  
- **Runway-impact:** bij lagere prijzen kan de kas sneller negatief worden; dit is in de gevoeligheidstabel zichtbaar.  
- **Koppeling naar break-even:** zie `70_breakeven.md` voor de minimale omzet per maand die nodig is om alle vaste lasten te dekken.  

---

_Disclaimer: Dit overzicht is indicatief. Werkelijke prijzen, volumes en klantgedrag kunnen afwijken._  
_Herleidbaarheid: alle cijfers zijn deterministisch afgeleid uit `dataset.v1.json`._
