# Mermaid Visuals — Draft

## Investeringsverdeling (pie)

```mermaid
pie showData
  title Investeringsverdeling (€30.700)
  "Hardware" : 5300
  "Ontwikkeling" : 8000
  "Observability" : 1200
  "Marketing" : 3000
  "Acquisitie (B2B)" : 7200
  "Werkkapitaal" : 6000
```

Toelichting (doel, leeswijzer, bewijswaarde)
- Doel: laat in één oogopslag zien waar de lening in wordt geïnvesteerd en waarom dit omzetversnelling en leveringszekerheid oplevert.
- Leeswijzer: de grootste posten (ontwikkeling en acquisitie) versnellen tijd‑tot‑waarde en afsprakenstroom; werkkapitaal dekt tijdelijke schommelingen.
- Wat dit aantoont: besteding is direct gekoppeld aan omzetgeneratie (acquisitie/marketing) en betrouwbare levering (ontwikkeling/observability/hardware), met buffer voor continuïteit.

## Roadmap 12 maanden (gantt)

```mermaid
gantt
    title Roadmap 12 maanden (indicatief)
    dateFormat  YYYY-MM-DD
    excludes    weekends

    section Productisering
    Spec/Contracts & determinisme :done, p1, 2025-01-01, 2025-02-15
    SSE/cancel/observability core  :active, p2, 2025-02-16, 2025-03-31

    section Referenties & Delivery
    RTX Dev/Obs/Provisioning       :p3, 2025-03-15, 2025-04-30
    Ref-implementaties (2–3)       :p4, 2025-05-01, 2025-07-31

    section Adapters & Scheduling
    vLLM/TGI/llama.cpp adapters    :p5, 2025-05-15, 2025-08-15
    Heterogene GPU scheduling      :p6, 2025-08-01, 2025-09-30

    section Go-to-market
    Partners & campagnes (verticals) :p7, 2025-03-01, 2025-06-30
    Cases/Benchmarks & proof bundle  :p8, 2025-06-01, 2025-09-30
```

Toelichting
- Doel: toont fasering van product → referenties → partners/cases zodat waardecreatie snel en beheerst verloopt.
- Leeswijzer: overlappende balken geven parallel werk aan; referenties/cases vallen na de eerste leveringen.
- Wat dit aantoont: planmatige uitvoering en voorspelbare milestones die direct aansluiten op tranche‑gates en omzetopbouw.

## Funnel (flowchart)

```mermaid
flowchart LR
    A[Impressions] -->|CTR 1.5–2.5%| B(Website Visits)
    B -->|Lead 3–6%| C{Leads}
    C -->|SQL 35–45%| D[Qualified]
    D -->|Meetings 70–80%| E[Meetings]
    E -->|Offers 60–70%| F[Offers]
    F -->|Win 35–45%| G[Won Deals]
    style G fill:#a3e635,stroke:#14532d,stroke-width:2px
```

Toelichting
- Doel: visualiseert de conversiestappen en streefwaarden zodat acquisitie meetbaar en bijstuurbaar is.
- Leeswijzer: de percentages zijn doelbandbreedtes; per campagne sturen we op CTR, CPL, SQL→meeting en winrate.
- Wat dit aantoont: pipeline is niet willekeurig; met KPI’s en retainer/partners borgen we voldoende afsprakenstroom voor dekking en groei.

## SLA Bundels (mindmap)

```mermaid
mindmap
  root((SLA Bundels))
    Essential
      Monitoring
      Updates: monthly
      Support: office hours
      R/T: 8/24h
    Advanced
      Observability: extended
      Updates: bi-weekly
      R/T: 4/12h
    Enterprise
      On-call: optional
      Updates: weekly
      Reports: extended
      R/T: 2/6h
```

Toelichting
- Doel: in één beeld de service‑niveaus en responstijden, zodat verwachtingen helder zijn.
- Leeswijzer: elk pakket bouwt voort op de vorige (meer observability, snellere R/T, uitgebreidere rapportage).
- Wat dit aantoont: voorspelbare, schaalbare service die maandlasten dekt en kwaliteit borgt (relevant voor terugbetaalbaarheid).

## Liquidity & Ramp (timeline)

```mermaid
timeline
  title Liquidity & Ramp (indicatief)
  2025 Q1 : 3→6 SLA (€1.35k→€2.7k) : Projects 1–2
  2025 Q2 : 7→8 SLA (€3.15k→€3.6k) : Projects 1
  2025 Q3 : 9→11 SLA (€4.05k→€4.95k) : Projects 2
  2025 Q4 : 11 SLA (≈€4.95k) : Projects 1
```

Toelichting
- Doel: laat zien hoe SLA‑ramp gecombineerd met projecten voldoende cash creëert voor leninglast en privéopname.
- Leeswijzer: elke periode toont SLA‑MRR (bij €450/klant) en aantal projecten; balans tussen vaste en variabele instroom.
- Wat dit aantoont: dekkingsgraad groeit voorspelbaar; projecten versnellen bufferopbouw richting stabiele DSCR.

## Tranche besteding & gates (timeline)

```mermaid
timeline
  title Tranches & Gates (indicatief)
  T1 (Mnd 0–2) : Productisering, salesmateriaal : Gate → Demo + 2 kwalificaties
  T2 (Mnd 2–4) : GPU/dev, observability, provisioning : Gate → 1 POC of 1 implementatie gepland
  T3 (Mnd 3–6) : Marketing verticals, partnerprogramma : Gate → 4+ SQL, ≥ 3 SLA actief
```

Toelichting
- Doel: koppelt uitgaven aan concrete mijlpalen om overbesteding te voorkomen.
- Leeswijzer: elke tranche heeft een duidelijke “Gate” (bewijsstuk/KPI) voordat de volgende uitgaven starten.
- Wat dit aantoont: uitgaven zijn beheerst en direct gelinkt aan omzet‑ en leveringsmijlpalen, wat risico verlaagt.

## Architectuur (flowchart)

```mermaid
flowchart LR
  Client((Clients)) --> ORCH[orchestratord]
  ORCH --> SSE[(SSE Streaming)]
  ORCH --> ADP{{Adapters}}
  ADP --> VLLM[vLLM]
  ADP --> TGI[TGI]
  ADP --> LCPP[llama.cpp]
  VLLM --> GPU[(GPU Pool)]
  TGI --> GPU
  LCPP --> GPU
  ORCH --> OBS[(Observability)]
  ORCH --> SECR[(Secrets/Policies)]
  classDef core fill:#dbeafe,stroke:#1d4ed8,stroke-width:1.5px;
  class ORCH,SSE core;
```

Toelichting
- Doel: toont de minimalistische, controleerbare laag tussen clients en open‑source componenten met focus op determinisme en observability.
- Leeswijzer: data blijft binnen de eigen omgeving; adapters koppelen naar gekozen engines; observability en policies zijn eerste‑klas.
- Wat dit aantoont: kleine attack‑surface, duidelijke verantwoordelijkheden en meetbaarheid — voorwaarden voor veilige, betrouwbare levering.
