# Break-even Analyse

Dit onderdeel toont het punt waarop de onderneming quitte speelt: de omzet die nodig is om alle kosten (vaste en variabele) te dekken. Dit geeft inzicht in de risicogrens en de benodigde volumes voor continuïteit.

## Kernpunten

- Vaste kosten per maand: {{vaste_kosten_pm}}
- Variabele kosten per eenheid: {{variabele_kosten_per_eenheid}}
- Verkoopprijs per eenheid: {{prijs_per_eenheid}}
- Brutomarge per eenheid: {{marge_per_eenheid}} ({{marge_pct}})
- Break-even aantal eenheden per maand: {{breakeven_eenheden_pm}}
- Break-even omzet per maand: {{breakeven_omzet_pm}}
- Veiligheidsmarge boven break-even: {{veiligheidsmarge_pct}}

## Visualisatie

```

Aantal eenheden → Omzet → Kosten → Resultaat
Break-even bij: {{breakeven\_eenheden\_pm}} eenheden / {{breakeven\_omzet\_pm}}

```

## Toelichting

- **Vaste kosten:** kosten die onafhankelijk zijn van productie of verkoopvolume (zoals huur, salarissen, vaste softwarekosten).  
- **Variabele kosten:** kosten die direct gekoppeld zijn aan het aantal verkochte eenheden (bijv. GPU-uren, API-fees, support per klant).  
- **Break-even punt:** het volume waarbij omzet = totale kosten. Daarboven ontstaat winstgevendheid.  
- **Veiligheidsmarge:** de mate waarin geplande omzet {{omzet_basis}} boven break-even ligt. Dit laat zien of de onderneming voldoende buffer heeft.  
- **Payback vs. maandlasten:** het break-even punt moet niet alleen kosten dekken, maar ook voldoende kas opleveren om financieringslasten (zie Qredits/maandlasten) te dragen.  

## Stress-signaal (indicatief)

Bij scenario’s met hogere kosten of lagere prijzen:  
- +10% vaste kosten → nieuw break-even omzet: {{breakeven_omzet_plus10_opex}}  
- −10% marge per eenheid → nieuw break-even omzet: {{breakeven_omzet_min10_marge}}  

Hiermee wordt zichtbaar hoe gevoelig de continuïteit is voor afwijkingen.

---

_Disclaimer: Dit overzicht is indicatief en gebaseerd op ingevoerde aannames. Voor een definitieve beoordeling zijn realistische marktdata en professioneel advies vereist._  
_Herleidbaarheid: alle cijfers zijn deterministisch afgeleid uit `dataset.v1.json`._
