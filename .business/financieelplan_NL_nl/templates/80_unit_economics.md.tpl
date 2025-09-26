# Unit Economics

Dit onderdeel geeft inzicht in de winstgevendheid per verkochte eenheid. Het laat zien welke bijdrage elke verkoop levert aan het dekken van vaste kosten en het genereren van winst. Hiermee kan worden beoordeeld of het bedrijfsmodel schaalbaar en duurzaam is.

## Kernpunten (basis-scenario)

- Verkoopprijs per eenheid: {{prijs_per_eenheid}}
- Variabele kosten per eenheid: {{variabele_kosten_per_eenheid}}
- Brutomarge per eenheid: {{marge_per_eenheid}} ({{marge_pct}} van omzet)
- Contribution margin per eenheid (marge na variabele saleskosten/CAC): {{contribution_margin}}
- Customer Acquisition Cost (CAC): {{cac_eur}}
- Lifetime Value (LTV, op basis van churn en marge): {{ltv_eur}}
- LTV/CAC-ratio: {{ltv_cac_ratio}}
- Payback-periode (maanden om CAC terug te verdienen): {{payback_maanden}}

## Bijdrage per eenheid

Elke verkochte eenheid levert een bijdrage van {{marge_per_eenheid}} aan het dekken van vaste kosten en daarna aan winst.  

Formule:
```

Verkoopprijs – Variabele kosten = Brutomarge per eenheid
Brutomarge – variabele saleskosten = Contribution margin

```

## Scenario’s (prijsvariatie)

| Scenario   | Verkoopprijs | Variabele kosten | Marge per eenheid | Marge % | Contribution margin |
|------------|-------------:|-----------------:|------------------:|--------:|---------------------:|
| Basis      | {{prijs_basis}} | {{var_basis}} | {{marge_basis}} | {{marge_pct_basis}} | {{contrib_basis}} |
| -10% prijs | {{prijs_min10}} | {{var_basis}} | {{marge_min10}} | {{marge_pct_min10}} | {{contrib_min10}} |
| +10% prijs | {{prijs_plus10}} | {{var_basis}} | {{marge_plus10}} | {{marge_pct_plus10}} | {{contrib_plus10}} |

## Schaalbaarheid

- Aantal eenheden nodig om vaste kosten van {{vaste_kosten_pm}} te dekken: {{breakeven_eenheden_pm}}  
- Bij {{eenheden_pm}} eenheden (basis): dekking vaste kosten = {{dekking_pct}}%

## Stress-signaal (indicatief)

- +10% variabele kosten → marge per eenheid: {{marge_varplus10}}, LTV/CAC: {{ltv_cac_varplus10}}  
- −10% verkoopprijs → marge per eenheid: {{marge_pricemin10}}, LTV/CAC: {{ltv_cac_pricemin10}}  

## Toelichting

- **Brutomarge per eenheid** geeft de basisbijdrage aan winst.  
- **Contribution margin** toont hoeveel werkelijk bijdraagt na verkoopkosten.  
- **LTV/CAC** laat zien of klanten duurzaam rendabel zijn.  
- **Payback-periode** maakt duidelijk hoe snel investeringen in acquisitie terugverdiend worden.  
- **Stress-analyse** toont de gevoeligheid voor kosten- of prijsveranderingen.  

---

_Disclaimer: Dit overzicht is indicatief en gebaseerd op ingevoerde aannames. Voor een definitieve beoordeling van unit economics zijn marktdata en klantgedrag bepalend._  
_Herleidbaarheid: alle cijfers zijn deterministisch afgeleid uit `dataset.v1.json`._
