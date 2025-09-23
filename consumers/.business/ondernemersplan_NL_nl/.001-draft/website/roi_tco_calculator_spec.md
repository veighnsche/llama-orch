# ROI/TCO Calculator — Specificatie (Draft)

Doel: eenvoudig vergelijk on‑prem (templates + SLA) vs publieke API kosten, om beslissers comfort te geven en intent te verhogen.

## Inputs
- Tokens per maand (in/out)
- API prijs per 1K tokens (gemiddeld)
- On‑prem kosten: implementatie (afschrijving) + SLA p/m + energie/schatting

## Outputs
- Maandelijkse kosten (API vs on‑prem)
- Breakeven point (maanden)
- Jaarbesparing (conservatief)

## Implementatie
- SSG: statische calculator met eenvoudige JS (no framework) of prerender tabelvarianten
- Gated download (CSV/MD) met de berekening per inputscenario
