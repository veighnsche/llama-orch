# Pricing & Omzetgevoeligheid

Dit onderdeel geeft inzicht in de gekozen prijsstrategie, de onderbouwing van tarieven, en de gevoeligheid van omzet bij prijswijzigingen. Het laat zien hoe robuust het bedrijfsmodel is bij veranderingen in prijs of volume.

## Kernpunten (basis-scenario)

- Huidige verkoopprijs per eenheid (bijv. EUR per 1M tokens): 1.2
- Verkochte eenheden per maand (bijv. 1M tokens): 187.50
- Omzet per maand: € 225,00
- Brutomarge per eenheid: € 0,35
- Customer Acquisition Cost (CAC): 0  
- Payback-tijd (maanden tot terugverdienen CAC): 0  

## Segmentatie (optioneel)


- **Public Tap (credits)** — Prijs: 1.2, Volume: 187.50, Omzet p/m: € 225,00, Marge: € 65,63


Dit maakt duidelijk hoe omzet verdeeld is over verschillende product-/dienstcategorieën.

## Gevoeligheidsanalyse (prijsvariatie)

Onderstaande tabel toont de impact van prijsvariaties op omzet en marge bij gelijkblijvend volume (eenheid = per 1M tokens bij credits-first):

| Prijsvariant | Prijs per eenheid | Omzet per maand | Brutomarge per maand | Runway (maanden) |
|--------------|------------------:|----------------:|---------------------:|-----------------:|
| -10%         | 1.08   | € 202,50 | € 59,06      | 9 |
| Basis        | 1.2   | € 225,00 | € 65,63      | 9 |
| +10%         | 1.32  | € 247,50| € 72,19     | 9|

## Gevoeligheidsanalyse (prijs-volume elasticiteit)

Indien prijswijzigingen leiden tot volumeverandering, geeft onderstaande tabel de gecombineerde effecten (eenheid = per 1M tokens bij credits-first):

| Scenario        | Prijs | Volume p/m | Omzet p/m | Brutomarge p/m |
|-----------------|------:|-----------:|----------:|---------------:|
| -10% prijs, +15% volume | 1.08 | 215.63 | € 232,88 | € 67,92 |
| Basis           | 1.2 | 187.50 | € 225,00 | € 65,63 |
| +10% prijs, -10% volume | 1.32 | 168.75 | € 222,75 | € 64,97 |

## Toelichting

- **Prijsstrategie:** gebaseerd op marktvergelijking, kostprijs per eenheid, en gewenste marge.  
- **Elasticiteit:** laat zien hoe omzet/marge bewegen bij prijs- en volume-aanpassingen.  
- **Runway-impact:** bij lagere prijzen kan de kas sneller negatief worden; dit is in de gevoeligheidstabel zichtbaar.  
- **Koppeling naar break-even:** zie `70_breakeven.md` voor de minimale omzet per maand die nodig is om alle vaste lasten te dekken.  

---

_Disclaimer: Dit overzicht is indicatief. Werkelijke prijzen, volumes en klantgedrag kunnen afwijken._  
_Herleidbaarheid: alle cijfers zijn deterministisch afgeleid uit `dataset.v1.json`._
