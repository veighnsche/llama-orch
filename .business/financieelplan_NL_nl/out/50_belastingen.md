# Belastingen (indicatief)

Dit onderdeel geeft een indicatieve weergave van het belastingregime en de BTW-afdracht. Alle berekeningen zijn gebaseerd op opgegeven aannames en vormen geen fiscale advisering. Het doel is uitsluitend om te laten zien dat de onderneming rekening houdt met fiscale verplichtingen en de kasimpact daarvan.

## Kernpunten

- Regime: eenmanszaak  
- BTW-tarief: 21%  
- BTW-model: omzet_enkel, afdracht per monthly  
- KOR van toepassing: False  
- BTW-vrijstelling: False  
- MKB-winstvrijstelling: 0%  
- Indicatieve belastingdruk (% van resultaat v/bel): -0.00%  
- Totale BTW-afdrachten over horizon: € 567,00  

## Toelichting

- **Belastingregime:** dit plan gaat uit van eenmanszaak (eenmanszaak/IB of BV/VPB). De exacte verschuldigde belasting hangt af van persoonlijke situatie en actuele tarieven.  
- **BTW:** de omzet wordt belast met 21% BTW volgens het opgegeven model. BTW-afdrachten zijn verwerkt in het liquiditeitsoverzicht.  
- **Kleineondernemersregeling (KOR):** als False, wordt geen BTW-afdracht berekend. Dit verlaagt administratieve lasten, maar beperkt de mogelijkheid om voorbelasting te verrekenen.  
- **BTW-vrijstelling:** indien False, zijn omzet en kosten buiten de heffing gehouden.  
- **MKB-winstvrijstelling:** vermindert het belastbare resultaat met 0%.  
- **Overige fiscale regelingen:** optioneel kunnen regelingen zoals zelfstandigenaftrek (IB) of innovatieaftrek/WBSO (VPB) van invloed zijn; deze zijn hier niet doorberekend, maar kunnen aanvullend voordeel opleveren.  


- **Indicatieve totale belastingheffing over de projectieperiode:** € 0,00


## Stress-signaal (indicatief)

Bij een winst 30% lager/hoger:
- Indicatieve belastingheffing daalt/stijgt naar: n.v.t.  
- Kasimpact op liquiditeit in piekmaanden: € 0,00  

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
