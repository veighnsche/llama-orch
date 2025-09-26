# Pricing & Omzetgevoeligheid

Dit onderdeel geeft inzicht in de gekozen prijsstrategie, de onderbouwing van tarieven, en de gevoeligheid van omzet bij prijswijzigingen. Het laat zien hoe robuust het bedrijfsmodel is bij veranderingen in prijs of volume.

## Kernpunten (basis-scenario)

- Huidige verkoopprijs per eenheid: {{prijs_per_eenheid}}
- Verkochte eenheden per maand: {{eenheden_pm}}
- Omzet per maand: {{omzet_basis}}
- Brutomarge per eenheid: {{marge_per_eenheid}}
- Customer Acquisition Cost (CAC): {{cac_eur}}  
- Payback-tijd (maanden tot terugverdienen CAC): {{payback_maanden}}  

## Segmentatie (optioneel)

{{#omzetstromen}}
- **{{naam}}** — Prijs: {{prijs}}, Volume: {{volume_pm}}, Omzet p/m: {{omzet_pm}}, Marge: {{marge_pm}}
{{/omzetstromen}}

Dit maakt duidelijk hoe omzet verdeeld is over verschillende product-/dienstcategorieën.

## Gevoeligheidsanalyse (prijsvariatie)

Onderstaande tabel toont de impact van prijsvariaties op omzet en marge bij gelijkblijvend volume:

| Prijsvariant | Prijs per eenheid | Omzet per maand | Brutomarge per maand | Runway (maanden) |
|--------------|------------------:|----------------:|---------------------:|-----------------:|
| -10%         | {{prijs_min10}}   | {{omzet_min10}} | {{marge_min10}}      | {{runway_min10}} |
| Basis        | {{prijs_basis}}   | {{omzet_basis}} | {{marge_basis}}      | {{runway_basis}} |
| +10%         | {{prijs_plus10}}  | {{omzet_plus10}}| {{marge_plus10}}     | {{runway_plus10}}|

## Gevoeligheidsanalyse (prijs-volume elasticiteit)

Indien prijswijzigingen leiden tot volumeverandering, geeft onderstaande tabel de gecombineerde effecten:

| Scenario        | Prijs | Volume p/m | Omzet p/m | Brutomarge p/m |
|-----------------|------:|-----------:|----------:|---------------:|
| -10% prijs, +15% volume | {{prijs_min10}} | {{volume_plus15}} | {{omzet_pricevol1}} | {{marge_pricevol1}} |
| Basis           | {{prijs_basis}} | {{volume_basis}} | {{omzet_basis}} | {{marge_basis}} |
| +10% prijs, -10% volume | {{prijs_plus10}} | {{volume_min10}} | {{omzet_pricevol2}} | {{marge_pricevol2}} |

## Toelichting

- **Prijsstrategie:** gebaseerd op marktvergelijking, kostprijs per eenheid, en gewenste marge.  
- **Elasticiteit:** laat zien hoe omzet/marge bewegen bij prijs- en volume-aanpassingen.  
- **Runway-impact:** bij lagere prijzen kan de kas sneller negatief worden; dit is in de gevoeligheidstabel zichtbaar.  
- **Koppeling naar break-even:** zie `70_breakeven.md` voor de minimale omzet per maand die nodig is om alle vaste lasten te dekken.  

---

_Disclaimer: Dit overzicht is indicatief. Werkelijke prijzen, volumes en klantgedrag kunnen afwijken._  
_Herleidbaarheid: alle cijfers zijn deterministisch afgeleid uit `dataset.v1.json`._
