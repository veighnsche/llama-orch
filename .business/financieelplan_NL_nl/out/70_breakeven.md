# Break-even Analyse

Dit onderdeel toont het punt waarop de onderneming quitte speelt: de omzet die nodig is om alle kosten (vaste en variabele) te dekken. Dit geeft inzicht in de risicogrens en de benodigde volumes voor continuïteit.

## Kernpunten

- Vaste kosten per maand: € 1.670,00
- Variabele kosten per eenheid: € 0,85
- Verkoopprijs per eenheid: 1.2
- Brutomarge per eenheid: € 0,35 (29.17%)
- Break-even aantal eenheden per maand: 4772.52
- Break-even omzet per maand: € 5.727,02
- Veiligheidsmarge boven break-even: -96.07%

## Visualisatie

```

Aantal eenheden → Omzet → Kosten → Resultaat
Break-even bij: {{breakeven\_eenheden\_pm}} eenheden / {{breakeven\_omzet\_pm}}

```

## Toelichting

- **Vaste kosten:** kosten die onafhankelijk zijn van productie of verkoopvolume (zoals huur, salarissen, vaste softwarekosten).  
- **Variabele kosten:** kosten die direct gekoppeld zijn aan het aantal verkochte eenheden (bijv. GPU-uren, API-fees, support per klant).  
- **Break-even punt:** het volume waarbij omzet = totale kosten. Daarboven ontstaat winstgevendheid.  
- **Veiligheidsmarge:** de mate waarin geplande omzet € 225,00 boven break-even ligt. Dit laat zien of de onderneming voldoende buffer heeft.  
- **Payback vs. maandlasten:** het break-even punt moet niet alleen kosten dekken, maar ook voldoende kas opleveren om financieringslasten (zie Qredits/maandlasten) te dragen.  

## Stress-signaal (indicatief)

Bij scenario’s met hogere kosten of lagere prijzen:  
- +10% vaste kosten → nieuw break-even omzet: € 6.298,77  
- −10% marge per eenheid → nieuw break-even omzet: € 6.363,29  

Hiermee wordt zichtbaar hoe gevoelig de continuïteit is voor afwijkingen.

---

_Disclaimer: Dit overzicht is indicatief en gebaseerd op ingevoerde aannames. Voor een definitieve beoordeling zijn realistische marktdata en professioneel advies vereist._  
_Herleidbaarheid: alle cijfers zijn deterministisch afgeleid uit `dataset.v1.json`._
