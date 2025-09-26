# Break-even Analyse

Dit onderdeel toont het punt waarop de onderneming quitte speelt: de omzet die nodig is om alle kosten (vaste en variabele) te dekken. Dit geeft inzicht in de risicogrens en de benodigde volumes voor continuïteit.

## Kernpunten

- Vaste kosten per maand: € 5.100,00
- Variabele kosten per eenheid: € 10,00
- Verkoopprijs per eenheid: 49.0
- Brutomarge per eenheid: € 39,00 (79.59%)
- Break-even aantal eenheden per maand: 130.77
- Break-even omzet per maand: € 6.407,84
- Veiligheidsmarge boven break-even: 33.82%

## Visualisatie

```

Aantal eenheden → Omzet → Kosten → Resultaat
Break-even bij: {{breakeven\_eenheden\_pm}} eenheden / {{breakeven\_omzet\_pm}}

```

## Toelichting

- **Vaste kosten:** kosten die onafhankelijk zijn van productie of verkoopvolume (zoals huur, salarissen, vaste softwarekosten).  
- **Variabele kosten:** kosten die direct gekoppeld zijn aan het aantal verkochte eenheden (bijv. GPU-uren, API-fees, support per klant).  
- **Break-even punt:** het volume waarbij omzet = totale kosten. Daarboven ontstaat winstgevendheid.  
- **Veiligheidsmarge:** de mate waarin geplande omzet € 8.575,00 boven break-even ligt. Dit laat zien of de onderneming voldoende buffer heeft.  
- **Payback vs. maandlasten:** het break-even punt moet niet alleen kosten dekken, maar ook voldoende kas opleveren om financieringslasten (zie Qredits/maandlasten) te dragen.  

## Stress-signaal (indicatief)

Bij scenario’s met hogere kosten of lagere prijzen:  
- +10% vaste kosten → nieuw break-even omzet: € 7.048,46  
- −10% marge per eenheid → nieuw break-even omzet: € 7.119,75  

Hiermee wordt zichtbaar hoe gevoelig de continuïteit is voor afwijkingen.

---

_Disclaimer: Dit overzicht is indicatief en gebaseerd op ingevoerde aannames. Voor een definitieve beoordeling zijn realistische marktdata en professioneel advies vereist._  
_Herleidbaarheid: alle cijfers zijn deterministisch afgeleid uit `dataset.v1.json`._
