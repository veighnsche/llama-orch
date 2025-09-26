# Werkkapitaal & Stress-test

Dit onderdeel geeft inzicht in het werkkapitaal dat nodig is om de onderneming draaiende te houden. Daarnaast wordt een stress-test weergegeven waarin wordt nagegaan wat er gebeurt bij ongunstige omstandigheden.

## Kernpunten Werkkapitaal

- Begin kaspositie: {{kas_begin}}
- Openstaande debiteuren (DSO): {{debiteuren_openstaand}}
- Openstaande crediteuren (DPO): {{crediteuren_openstaand}}
- Voorraadwaarde: {{voorraad}}
- Borgsommen / deposito’s: {{borg_depositos}}
- Netto werkkapitaalbehoefte: {{werkkapitaal_totaal}}

## Stress-test Scenario

| Scenario              | Aanname                           | Effect op kaspositie |
|-----------------------|-----------------------------------|---------------------:|
| Omzet –30%            | {{stress_omzet_min30}}            | {{kas_na_stress_omzet}} |
| Betaaltermijn +30d    | {{stress_dso_plus30}}             | {{kas_na_stress_dso}} |
| OPEX +20%             | {{stress_opex_plus20}}            | {{kas_na_stress_opex}} |

## Toelichting

- **Werkkapitaal**: omvat de middelen die vastzitten in voorraad en debiteuren, minus uitgestelde betalingen aan crediteuren.  
- **Stress-test**: maakt zichtbaar hoe gevoelig de kaspositie is voor omzetdaling, langere betalingstermijnen of hogere kosten.  
- **Gebruik**: geeft inzicht in risico’s en laat zien of aanvullende buffers of kredietlijnen nodig zijn.

---

_Disclaimer: Dit overzicht is indicatief. Werkelijke kasstromen en werkkapitaalbehoefte kunnen afwijken._
