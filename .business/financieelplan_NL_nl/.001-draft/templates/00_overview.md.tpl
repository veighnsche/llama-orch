# Overzicht

Dit overzicht geeft de kerncijfers van het financieel plan. Het laat in één oogopslag zien welke investeringen en financiering nodig zijn, wat de kaspositie is gedurende de looptijd, en of de maandlasten haalbaar zijn.

- **Periode:** {{horizon_maanden}} maanden (van {{eerste_maand}} tot en met {{laatste_maand}})
- **Scenario:** {{scenario}}
- **Belastingregime:** {{regime}}
- **BTW-model:** {{btw_model}}, afdracht per {{vat_period}}

## Investeringen en Financiering

- Totale investering: {{total_investering}}
- Totale financiering: {{total_financiering}}
- Eigen inbreng: {{eigen_inbreng}} ({{eigen_inbreng_pct}})
- Verhouding vreemd vermogen / eigen vermogen: {{debt_equity_pct}}
- Indicatieve terugverdientijd investering: {{payback_maanden}} maanden
- Interne rentabiliteit (IRR): {{irr_project}} bij disconteringsvoet {{discount_rate_pct}}%

## Kaspositie

- Start kas (eigen inbreng + ontvangen leningen in maand {{start_maand}}): {{eerste_kas}}
- Laagste kasstand: {{laagste_kas}} in maand {{kas_diepste_maand}} ({{kas_waarschuwing}})
- Eind kaspositie na {{horizon_maanden}} maanden: {{kas_eind}}
- Runway (maanden kas ≥ 0): {{runway_maanden_base}} (base), {{runway_maanden_worst}} (worst)

## Dekking maandlasten

- Gemiddelde EBITDA gedeeld door gemiddelde schuldendienst: {{maandlast_dekking}}
- Laagste DSCR: {{dscr_min}} in maand {{dscr_maand_min}} (maanden <1: {{dscr_below_1_count}})
- Beoordeling: maandlasten zijn hiermee **{{dekking_verdict}}**

## Break-even signaal

- Break-even omzet: {{breakeven_omzet_pm}} per maand
- Veiligheidsmarge boven break-even: {{veiligheidsmarge_pct}}

---

_Alle bedragen zijn indicatief en gebaseerd op opgegeven aannames. Dit document vormt geen fiscale of juridische advisering._  

_Herleidbaarheid: alle cijfers komen deterministisch uit `dataset.v1.json`._
