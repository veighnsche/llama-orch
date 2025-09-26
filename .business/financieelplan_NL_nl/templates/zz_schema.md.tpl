# Schema (mapping input → output)

Dit schema laat zien hoe de invoervelden in de YAML/JSON-configuratie worden vertaald naar de tabellen en rapportages in dit financieel plan. Het biedt transparantie en maakt duidelijk hoe aannames doorwerken in de uitkomsten.

## Mapping

{{#mapping}}
- **{{in}}** → **{{out}}**
{{/mapping}}

## Voorbeeldcategorieën

- **Bedrijf** → Rechtsvorm, startmaand, urencriterium, partner-ondernemer  
- **Privé** → Inkomsten, uitgaven per maand, vermogenspositie, privé-onttrekkingen  
- **Investeringen** → `10_investering.csv` (omschrijving, levensduur, start, afschrijving, bedrag)  
- **Financiering** → `10_financiering.csv` (bron, bedrag, rente, looptijd, grace)  
- **Omzetmodel** → `30_exploitatie.csv` (omzet, COGS, marge, OPEX, resultaat)  
- **Liquiditeit** → `20_liquiditeit_monthly.csv` (begin kas, ontvangsten, uitgaven, eind kas)  
- **Leningen** → `40_amortisatie.csv` (rente, aflossing, restschuld per maand)  
- **Belastingen** → `50_tax.csv` (regime, BTW, KOR, vrijstelling, MKB-vrijstelling)

## Toelichting

- Elke inputvariabele heeft één of meerdere uitwerkingen in de rapportages.  
- Tabellen en Markdown-rapporten worden volledig deterministisch gegenereerd.  
- Alle berekeningen gebruiken dezelfde Decimal-context (afgerond op centen).  
- Onbekende of ontbrekende placeholders leiden tot een foutmelding, zodat consistentie gegarandeerd blijft.

---

_Disclaimer: Dit schema is bedoeld voor transparantie en controleerbaarheid. Het heeft geen commerciële of juridische status._
