# Overzicht

Dit overzicht geeft de kerncijfers van het financieel plan. Het laat in één oogopslag zien welke investeringen en financiering nodig zijn, wat de kaspositie is gedurende de looptijd, en of de maandlasten haalbaar zijn.

- **Periode:** 12 maanden (van 2025-10 tot en met 2026-09)
- **Scenario:** base
- **Belastingregime:** eenmanszaak
- **BTW-model:** omzet_enkel, afdracht per monthly

## Investeringen en Financiering

- Totale investering: € 2.500,00
- Totale financiering: € 30.000,00
- Eigen inbreng: € 0,00 (0.00%)
- Verhouding vreemd vermogen / eigen vermogen: 100.00% / 0.00%
- Indicatieve terugverdientijd investering: 0 maanden
- Interne rentabiliteit (IRR): n.v.t. bij disconteringsvoet 10.0%

## Kaspositie

- Start kas (eigen inbreng + ontvangen leningen in maand 2025-10): € 30.000,00
- Laagste kasstand: -€ 4.964,04 in maand 2026-09 (negatief)
- Eind kaspositie na 12 maanden: -€ 4.964,04
- Runway (maanden kas ≥ 0): 10 (base), 10 (worst)

## Dekking maandlasten

- Gemiddelde EBITDA gedeeld door gemiddelde schuldendienst: -0.10x
- Laagste DSCR: -10.22 in maand 2025-10 (maanden <1: 7)
- Beoordeling: maandlasten zijn hiermee **onvoldoende**

## Break-even signaal

- Break-even omzet: € 10.284,54 per maand
- Veiligheidsmarge boven break-even: -1.79%

---

_Alle bedragen zijn indicatief en gebaseerd op opgegeven aannames. Dit document vormt geen fiscale of juridische advisering._  

_Herleidbaarheid: alle cijfers komen deterministisch uit `dataset.v1.json`._


# Qredits – Aanvraag: Motivatie & Terugbetalingsplan

Dit onderdeel motiveert waarom financiering noodzakelijk is, hoe de middelen worden ingezet (use of funds), en hoe de terugbetaling verantwoord en haalbaar is gegeven het bedrijfsmodel en de ramingen.

## Samenvatting van de aanvraag

- Gevraagde financiering (totaal vreemd vermogen): € 30.000,00
- Aantal leningen: 1
- Gemiddelde looptijd (mnd): 48
- Gemiddelde maandlast (rente + aflossing): € 530,17
- Huidige dekking maandlasten (DSCR-min): -10.22 (onvoldoende)
- Runway (maanden ≥ 0 kaspositie, basis): 10
- Runway stress (omzet −30%, DSO +30d, OPEX +20%): 5

## Waarom financieren (use of funds)

- Overbrugging vaste lasten in de opstartfase (marketing, software, huisvesting, verzekeringen/administratie) totdat omzet uit credits opschaalt.
- Kleine, noodzakelijke CAPEX voor werkmiddelen.
- Werkkapitaalbuffer voor pieken in BTW-afdrachten en seizoensschommelingen.

Indicatieve componenten:
- CAPEX totaal: € 2.500,00
- Maandlasten lening (gemiddeld): € 530,17

## Terugbetalingsplan

- Omzetmodel is credits-first (prepaid), waardoor debiteurenrisico en DSO minimaal zijn en kasontvangsten direct plaatsvinden.
- Prijszetting per 1M tokens met oplopende vraag zorgt voor gecontroleerde schaalbaarheid.
- Maandlasten (rente + aflossing) zijn inbegrepen in de exploitatie en cashflowprojecties.
- DSCR en runway worden maandelijks gemeten; in de latere maanden verbetert de dekking naarmate volumes stijgen.

### Risico, stress en mitigaties

- Stress-case (omzet −30%, DSO +30d, OPEX +20%):
  - Laagste kasstand: -€ 25.213,44
  - Runway: 5 maanden
  - Laagste DSCR (indicatief): -13.82
- Mitigaties:
  - Prepaid credits en directe incasso minimaliseren DSO en credit risk.
  - OPEX is flexibel: marketing/software schaalbaar; personeelslasten pas later toevoegen.
  - Maandlastenmonitoring en triggers bij DSCR < 1 voor bijsturing (pricing, kosten, of tempo groei).

## Conclusie

De aangevraagde financiering wordt primair gebruikt om de opstartfase te overbruggen (vaste lasten, beperkte CAPEX) tot het creditsvolume het break-evenpunt passeert en de maandlasten structureel gedekt zijn. Dankzij prepaid credits en een gecontroleerde volumegroei is de terugbetaling realistisch en gecontroleerd, met duidelijke stress-signalen en mitigerende maatregelen.


# Investeringen & Financiering

Dit onderdeel beschrijft de benodigde investeringen voor de onderneming en de manier waarop deze gefinancierd worden. Het geeft inzicht in de omvang van de investeringen, de verhouding tussen eigen middelen en vreemd vermogen, en de belangrijkste kenmerken van leningen.

## Investeringen

- Aantal investeringsposten: 1
- Totaal investeringsbedrag: € 2.500,00
- Gemiddelde afschrijvingstermijn: 36 maanden
- Grootste investering: Laptop (€ 2.500,00)

Elke investering wordt lineair afgeschreven over de opgegeven levensduur in maanden. Indien een startmaand is gespecificeerd, wordt prorata afgeschreven vanaf die maand.  
Dit geeft een realistisch beeld van jaarlijkse lasten en voorkomt dat kosten in één keer drukken op het resultaat.

_De volledige detailtabel is opgenomen in `10_investering.csv`._

## Financiering

- Eigen inbreng: € 0,00
- Vreemd vermogen: € 30.000,00
- Aantal leningen: 1
- Debt/Equity-verhouding: 100.00% / 0.00%
- Gemiddelde looptijd leningen: 48 maanden

### Specificatie leningen


- **Qredits**  
  Hoofdsom: € 30.000,00  
  Looptijd: 48 maanden  
  Rente: 9.0% per jaar (nominaal), 0.75% per maand  
  Grace-periode: 6 maanden  
  Termijnbedrag: 835.34 per maand  
  Effectieve jaarrente (APR): 9.0%  
  Totale rentelasten over looptijd: € 2.630,64  


De berekeningen zijn op annuïtaire basis uitgevoerd, tenzij anders aangegeven. Bij grace-periodes wordt rente tijdens de grace doorberekend, waarna volledige aflossing start.

_De volledige detailtabel is opgenomen in `10_financiering.csv`._

## Samenvattende ratio’s

- Totale investeringen gedekt door eigen inbreng: 0.00%  
- Totale investeringen gedekt door vreemd vermogen: 1200.00%  
- Gemiddelde maandlasten alle leningen: 0.00  
- Schuldendienst-dekking (DSCR, gem.): -0.10x

---

_Alle bedragen zijn indicatief en gebaseerd op opgegeven aannames. Dit document vormt geen fiscale of juridische advisering._  
_Herleidbaarheid: alle cijfers zijn deterministisch afgeleid uit `dataset.v1.json`._


# Liquiditeit

Dit kasstroomoverzicht geeft inzicht in de ontvangsten en uitgaven per maand. Hierin zijn verschuivingen verwerkt door betalingstermijnen van debiteuren (DSO) en crediteuren (DPO), en de afdracht van BTW. Het laat zien of de onderneming gedurende de looptijd over voldoende liquiditeit beschikt om verplichtingen tijdig na te komen.

## Kerncijfers

- Begin kaspositie: € 0,00 in maand 2025-10
- Eind kaspositie: -€ 4.964,04 in maand 2026-09
- Laagste kasstand: -€ 4.964,04 in maand 2026-09 (negatief)
- Runway (maanden kas ≥ 0): 10 in het basisscenario
- Aantal maanden negatief: 2 (negatief)

## BTW-afdracht

- BTW-model: omzet_enkel
- Afdrachtfrequentie: monthly
- Totale BTW-afdrachten over horizon: € 25.452,00
- Grootste afdracht in één maand: € 3.024,00 (2026-06)

BTW wordt berekend over de omzet minus aftrekbare kosten (afhankelijk van het gekozen model). De afdracht vindt plaats in de laatste maand van de periode (maandelijks of per kwartaal).

## Toelichting

- **Inkomsten** omvatten kasontvangsten uit omzet (inclusief DSO-verschuiving) en overige ontvangsten zoals subsidies of eigen inbreng.  
- **Uitgaven** omvatten betalingen voor inkoop (COGS, inclusief DPO), OPEX, rente, aflossingen en BTW-afdracht.  
- **Negatieve kasposities** worden expliciet zichtbaar gemaakt in de maandtabel en zijn voorzien van een waarschuwing.  
- Een kasbuffer van minimaal 2500 wordt aanbevolen; de feitelijke laagste stand is -€ 4.964,04.

## Stress-case signaal

Indicatieve impact bij een neerwaarts scenario (omzet −30%, DSO +30 dagen, OPEX +10%):
- Laagste kasstand: -€ 25.213,44  
- Runway: 5 maanden  

Dit geeft een indicatie van de weerbaarheid bij tegenvallende resultaten.

_De detailtabel is opgenomen in `20_liquiditeit_monthly.csv`._

---

_Alle bedragen zijn indicatief en gebaseerd op opgegeven aannames. Dit document vormt geen fiscale of juridische advisering._  
_Herleidbaarheid: alle cijfers zijn deterministisch afgeleid uit `dataset.v1.json`._


# Exploitatie (maandelijks)

Dit exploitatieoverzicht laat de resultatenrekening zien per maand. Hierin worden omzet, kostprijs van de omzet (COGS), brutomarge, operationele kosten, afschrijvingen, rente en het resultaat voor belasting weergegeven. Het overzicht maakt duidelijk hoe robuust de marge is en of de onderneming structureel winstgevend kan opereren.

## Kerncijfers (totaal over 12 maanden)

- Totale omzet: € 121.200,00
- Totale brutomarge: € 35.350,00 (29.17%)
- Totale operationele kosten (OPEX): € 36.000,00
- Totale afschrijvingen: € 833,28
- Totale rentelasten: € 2.630,64
- Resultaat voor belasting: -€ 4.113,92
- Gemiddelde EBITDA per maand: -€ 54,17
- Nettomarge (%): -3.39%

## Trendanalyse

- Beste maand resultaat: € 928,79 in 2026-09
- Slechtste maand resultaat: -€ 2.594,44 in 2025-10
- Aantal verlieslatende maanden: 6 (zorgpunt)

## Toelichting op de opbouw

- **Omzet**: verwachte verkoopopbrengsten inclusief eventuele seizoenspatronen en scenario-effecten.  
- **COGS (kostprijs omzet)**: directe inkoopkosten die samenhangen met de omzet (bijv. GPU-uren, API-fees).  
- **Brutomarge**: verschil tussen omzet en COGS, uitgedrukt als bedrag en percentage.  
- **OPEX**: operationele kosten uitgesplitst naar categorie (personeel, marketing, huisvesting, software, overige overhead).  
- **Afschrijving**: lineaire afschrijving van investeringen volgens opgegeven levensduur.  
- **Rente**: lasten op opgenomen leningen.  
- **EBITDA**: brutomarge minus OPEX.  
- **Resultaat v/bel**: resultaat voor belastingen, na rente en afschrijvingen.

De relatie kan schematisch worden weergegeven als:

```

Omzet → COGS → Brutomarge → OPEX → EBITDA → Afschrijving → Rente → Resultaat (v/bel)

```

## Break-even signaal

- Benodigde omzet per maand om break-even te draaien: € 10.284,54
- Veiligheidsmarge t.o.v. break-even: -1.79%

## Stress-case signaal (indicatief)

Bij een scenario met omzet −30% en OPEX +10%:
- Gemiddeld resultaat: -€ 1.826,58
- Aantal verlieslatende maanden: 12
- Implicatie: zorgpunt

## Detailtabel

De maandelijkse exploitatiecijfers zijn opgenomen in `30_exploitatie.csv` met de volgende kolommen:

- `maand`
- `omzet`
- `cogs`
- `marge`
- `opex_<categorie>...`
- `opex_totaal`
- `afschrijving`
- `rente`
- `ebitda`
- `resultaat_vb`

---

_Alle bedragen zijn indicatief en gebaseerd op opgegeven aannames. Dit document vormt geen fiscale of juridische advisering._  
_Herleidbaarheid: alle cijfers zijn deterministisch afgeleid uit `dataset.v1.json`._

# Qredits / Maandlasten

Dit onderdeel beschrijft de maandelijkse verplichtingen voortvloeiend uit de opgenomen leningen. Het laat per lening zien wat de aflossingen en rentelasten zijn, en welke totale maandlast dit oplevert. Hiermee wordt inzichtelijk of de onderneming structureel aan haar schuldverplichtingen kan voldoen.

## Kernpunten

- Aantal leningen: 1
- Totale gemiddelde maandlast (rente + aflossing): € 530,17
- Gemiddelde schuldendienst-dekking (EBITDA / schuldendienst): -0.10x
- Laagste DSCR: -10.22 in maand 2025-10
- Aantal maanden met DSCR < 1: 7 (onvoldoende)

## Specificatie per lening


- **Qredits**  
  Hoofdsom: € 30.000,00  
  Looptijd: 48 maanden  
  Nominale rente: 9.0% per jaar (0.75% per maand)  
  Grace-periode: 6 maanden  
  Maandlast: 835.34  
  Totale rentelasten over looptijd: € 2.630,64  
  Effectieve jaarrente (APR, indicatief): 9.0%  
  Restschuld einde looptijd: € 26.268,60


## Toelichting

- Maandlasten bestaan uit rente en aflossing volgens annuïtaire berekening (tenzij anders opgegeven).  
- Tijdens de grace-periode wordt alleen rente betaald; daarna start aflossing.  
- Maandlasten zijn volledig doorvertaald naar het kasstroomoverzicht in `20_liquiditeit_monthly.csv`.  
- Bij meerdere leningen is de gecombineerde schuldendienst zichtbaar in de samenvattende DSCR-cijfers.  
- Het risico op knelpunten wordt zichtbaar gemaakt via de DSCR-analyse.

## Stress-signaal (indicatief)

Bij een scenario met **EBITDA −30%** en gelijkblijvende maandlasten:
- Laagste DSCR: -13.82  
- Aantal maanden met DSCR < 1: 12  
- Implicatie: onvoldoende

## Detailtabel

Het volledige aflossingsschema per lening is opgenomen in `40_amortisatie.csv` met kolommen:

- `maand`
- `verstrekker`
- `rente_pm`
- `aflossing_pm`
- `restschuld`

---

_Alle bedragen zijn indicatief en gebaseerd op opgegeven aannames. Dit document vormt geen fiscale of juridische advisering._  
_Herleidbaarheid: alle cijfers zijn deterministisch afgeleid uit `dataset.v1.json`._


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
- Totale BTW-afdrachten over horizon: € 25.452,00  

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


# Break-even Analyse

Dit onderdeel toont het punt waarop de onderneming quitte speelt: de omzet die nodig is om alle kosten (vaste en variabele) te dekken. Dit geeft inzicht in de risicogrens en de benodigde volumes voor continuïteit.

## Kernpunten

- Vaste kosten per maand: € 3.000,00
- Variabele kosten per eenheid: € 0,85
- Verkoopprijs per eenheid: 1.2
- Brutomarge per eenheid: € 0,35 (29.17%)
- Break-even aantal eenheden per maand: 8570.45
- Break-even omzet per maand: € 10.284,54
- Veiligheidsmarge boven break-even: -1.79%

## Visualisatie

```

Aantal eenheden → Omzet → Kosten → Resultaat
Break-even bij: {{breakeven\_eenheden\_pm}} eenheden / {{breakeven\_omzet\_pm}}

```

## Toelichting

- **Vaste kosten:** kosten die onafhankelijk zijn van productie of verkoopvolume (zoals huur, salarissen, vaste softwarekosten).  
- **Variabele kosten:** kosten die direct gekoppeld zijn aan het aantal verkochte eenheden (bijv. GPU-uren, API-fees, support per klant).  
- **Break-even punt:** het volume waarbij omzet = totale kosten. Daarboven ontstaat winstgevendheid.  
- **Veiligheidsmarge:** de mate waarin geplande omzet € 10.100,00 boven break-even ligt. Dit laat zien of de onderneming voldoende buffer heeft.  
- **Payback vs. maandlasten:** het break-even punt moet niet alleen kosten dekken, maar ook voldoende kas opleveren om financieringslasten (zie Qredits/maandlasten) te dragen.  

## Stress-signaal (indicatief)

Bij scenario’s met hogere kosten of lagere prijzen:  
- +10% vaste kosten → nieuw break-even omzet: € 11.314,30  
- −10% marge per eenheid → nieuw break-even omzet: € 11.427,15  

Hiermee wordt zichtbaar hoe gevoelig de continuïteit is voor afwijkingen.

---

_Disclaimer: Dit overzicht is indicatief en gebaseerd op ingevoerde aannames. Voor een definitieve beoordeling zijn realistische marktdata en professioneel advies vereist._  
_Herleidbaarheid: alle cijfers zijn deterministisch afgeleid uit `dataset.v1.json`._


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


# Werkkapitaal & Stress-test

Dit onderdeel geeft inzicht in het werkkapitaal dat nodig is om de onderneming draaiende te houden. Daarnaast wordt een stress-test weergegeven waarin wordt nagegaan wat er gebeurt bij ongunstige omstandigheden.

## Kernpunten Werkkapitaal

- Begin kaspositie: € 0,00
- Openstaande debiteuren (DSO): € 0,00
- Openstaande crediteuren (DPO): € 0,00
- Voorraadwaarde: € 0,00
- Borgsommen / deposito’s: € 0,00
- Netto werkkapitaalbehoefte: € 0,00

## Stress-test Scenario

| Scenario              | Aanname                           | Effect op kaspositie |
|-----------------------|-----------------------------------|---------------------:|
| Omzet –30%            | −30%            | -€ 7.933,44 |
| Betaaltermijn +30d    | +30 dagen             | -€ 19.364,04 |
| OPEX +20%             | +20%            | -€ 12.164,04 |

## Toelichting

- **Werkkapitaal**: omvat de middelen die vastzitten in voorraad en debiteuren, minus uitgestelde betalingen aan crediteuren.  
- **Stress-test**: maakt zichtbaar hoe gevoelig de kaspositie is voor omzetdaling, langere betalingstermijnen of hogere kosten.  
- **Gebruik**: geeft inzicht in risico’s en laat zien of aanvullende buffers of kredietlijnen nodig zijn.

---

_Disclaimer: Dit overzicht is indicatief. Werkelijke kasstromen en werkkapitaalbehoefte kunnen afwijken._
