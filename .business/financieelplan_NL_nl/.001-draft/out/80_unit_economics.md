# Unit Economics

Dit onderdeel geeft inzicht in de winstgevendheid per verkochte eenheid. Het laat zien welke bijdrage elke verkoop levert aan het dekken van vaste kosten en het genereren van winst. Hiermee kan worden beoordeeld of het bedrijfsmodel schaalbaar en duurzaam is.

## Kernpunten (basis-scenario)

- Verkoopprijs per eenheid (bijv. EUR per 1M tokens): 1.2
- Variabele kosten per eenheid (bijv. per 1M tokens): € 0,85
- Brutomarge per eenheid (bijv. per 1M tokens): € 0,35 (29.17% van omzet)
- Contribution margin per eenheid (marge na variabele saleskosten/CAC): € 0,35
- Customer Acquisition Cost (CAC): 0
- Lifetime Value (LTV, op basis van churn en marge): € 0,00
- LTV/CAC-ratio: n.v.t.
- Payback-periode (maanden om CAC terug te verdienen): 0

## Bijdrage per eenheid

Elke verkochte eenheid (bijv. 1M tokens) levert een bijdrage van € 0,35 aan het dekken van vaste kosten en daarna aan winst.  

Formule:
```

Verkoopprijs – Variabele kosten = Brutomarge per eenheid
Brutomarge – variabele saleskosten = Contribution margin

```

## Scenario’s (prijsvariatie)

| Scenario   | Verkoopprijs | Variabele kosten | Marge per eenheid | Marge % | Contribution margin |
|------------|-------------:|-----------------:|------------------:|--------:|---------------------:|
| Basis      | 1.2 | 0.85 | € 2.945,83 | 29.17% | € 2.945,83 |
| -10% prijs | 1.08 | 0.85 | € 2.651,25 | 21.30% | € 1.935,83 |
| +10% prijs | 1.32 | 0.85 | € 3.240,42 | 35.61% | € 3.955,83 |

## Schaalbaarheid

- Aantal eenheden nodig om vaste kosten van € 3.000,00 te dekken: 8570.45  
- Bij 8416.67 eenheden (basis): dekking vaste kosten = 98.19%%

## Stress-signaal (indicatief)

- +10% variabele kosten → marge per eenheid: € 0,27, LTV/CAC: n.v.t.  
- −10% verkoopprijs → marge per eenheid: € 0,23, LTV/CAC: n.v.t.  

## Toelichting

- **Brutomarge per eenheid** geeft de basisbijdrage aan winst.  
- **Contribution margin** toont hoeveel werkelijk bijdraagt na verkoopkosten.  
- **LTV/CAC** laat zien of klanten duurzaam rendabel zijn.  
- **Payback-periode** maakt duidelijk hoe snel investeringen in acquisitie terugverdiend worden.  
- **Stress-analyse** toont de gevoeligheid voor kosten- of prijsveranderingen.  

---

_Disclaimer: Dit overzicht is indicatief en gebaseerd op ingevoerde aannames. Voor een definitieve beoordeling van unit economics zijn marktdata en klantgedrag bepalend._  
_Herleidbaarheid: alle cijfers zijn deterministisch afgeleid uit `dataset.v1.json`._
