# Belastingen (indicatief)

Dit onderdeel geeft een indicatieve weergave van het belastingregime en de BTW-afdracht. Alle berekeningen zijn gebaseerd op opgegeven aannames en vormen geen fiscale advisering. Het doel is uitsluitend om te laten zien dat de onderneming rekening houdt met fiscale verplichtingen en de kasimpact daarvan.

## Kernpunten

- Regime: {{regime}}  
- BTW-tarief: {{btw_pct}}%  
- BTW-model: {{btw_model}}, afdracht per {{vat_period}}  
- KOR van toepassing: {{kor}}  
- BTW-vrijstelling: {{btw_vrij}}  
- MKB-winstvrijstelling: {{mkb_vrijstelling_pct}}%  
- Indicatieve belastingdruk (% van resultaat v/bel): {{belastingdruk_pct}}  
- Totale BTW-afdrachten over horizon: {{btw_totaal}}  

## Toelichting

- **Belastingregime:** dit plan gaat uit van {{regime}} (eenmanszaak/IB of BV/VPB). De exacte verschuldigde belasting hangt af van persoonlijke situatie en actuele tarieven.  
- **BTW:** de omzet wordt belast met {{btw_pct}}% BTW volgens het opgegeven model. BTW-afdrachten zijn verwerkt in het liquiditeitsoverzicht.  
- **Kleineondernemersregeling (KOR):** als {{kor}}, wordt geen BTW-afdracht berekend. Dit verlaagt administratieve lasten, maar beperkt de mogelijkheid om voorbelasting te verrekenen.  
- **BTW-vrijstelling:** indien {{btw_vrij}}, zijn omzet en kosten buiten de heffing gehouden.  
- **MKB-winstvrijstelling:** vermindert het belastbare resultaat met {{mkb_vrijstelling_pct}}%.  
- **Overige fiscale regelingen:** optioneel kunnen regelingen zoals zelfstandigenaftrek (IB) of innovatieaftrek/WBSO (VPB) van invloed zijn; deze zijn hier niet doorberekend, maar kunnen aanvullend voordeel opleveren.  

{{#indicatieve_heffing_jaar}}
- **Indicatieve totale belastingheffing over de projectieperiode:** {{indicatieve_heffing_jaar}}
{{/indicatieve_heffing_jaar}}

## Stress-signaal (indicatief)

Bij een winst 30% lager/hoger:
- Indicatieve belastingheffing daalt/stijgt naar: {{stress_heffing_range}}  
- Kasimpact op liquiditeit in piekmaanden: {{stress_kasimpact}}  

Dit laat zien dat de onderneming rekening houdt met variatie in fiscale lasten.

## Detailtabel

De indicatieve belastinggegevens zijn opgenomen in `50_tax.csv` met kolommen:

- `regime`
- `btw_pct`
- `btw_model`
- `kor`
- `btw_vrij`
- `mkb_vrijstelling_pct`
- `belastingdruk_pct`
- `indicatieve_heffing_jaar`

---

_Disclaimer: Dit overzicht is uitsluitend bedoeld als rekenvoorbeeld. Voor een definitieve beoordeling van fiscale verplichtingen is altijd professioneel advies vereist._  
_Herleidbaarheid: alle cijfers zijn deterministisch afgeleid uit `dataset.v1.json`._
