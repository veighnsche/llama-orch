# Open‑source AI op eigen servers. Veilig. Snel. Meetbaar.

Geharde Server Templates + Implementatie in 2–4 weken + Beheerde SLA.  
`llama‑orch` is een dunne, controleerbare laag tussen uw organisatie en open‑source componenten om privacy, veiligheid en determinisme te borgen.

CTA’s:  
- Plan een discovery call (gratis, 30 min)  
- Download de on‑prem checklist (EU/AVG‑first)

## Waarom nu

Publieke AI‑API’s zijn snel maar brengen lock‑in, onduidelijke kosten en privacyrisico’s. On‑prem/hybride geeft controle en voorspelbaarheid, maar implementatie en beheer zijn complex. Wij leveren geharde Server Templates en een gestroomlijnde implementatie zodat u binnen weken live bent — mét observability en SLA.

## Oplossing in het kort

- Server Templates (GPU): gestandaardiseerde, geharde images en configuraties (OS‑profielen, NVIDIA‑drivers, container runtime), provisioning‑scripts, security hardening, logging/metrics out‑of‑the‑box.  
- Implementatie: modellen/engines plaatsen, SSE‑token‑streaming met cancel en timeouts, observability dashboards, tests en overdracht documentatie.  
- Beheer (SLA): monitoring, updates/patching, incidentrespons en maandrapportage met KPI’s.

`llama‑orch` is bewust een minimalistische orkestratielaag: klein attack‑surface, goed te auditen, en gericht op deterministische output en heldere metrics.

## Wat u krijgt (waarde)

- EU/AVG‑first: datasoevereiniteit, DPIA‑ondersteuning, audit‑logging en datalokatie in de EU.  
- Geen vendor lock‑in: open‑source componenten, eigenaarschap bij u.  
- Meetbaar en voorspelbaar: latency/throughput/uptime en kosten per 1K tokens inzichtelijk.  
- Snel live: 2–4 weken, met referentie‑architecturen en runbooks.  
- Veiliger: geharde templates en least‑privilege defaults; updates onder regie.  
- SLA‑zekerheid: duidelijke R/T‑tijden, automatische incasso, maandelijkse rapportage.

## Hoe het werkt (stappen)

1) Discovery & technische scan (hardware, netwerken, security)
2) Server Templates plaatsen (OS, drivers, runtimes, observability)
3) `llama‑orch` configureren (SSE, determinisme, policies)
4) Modellen/engines deployen + testen (benchmarks, failover)
5) Acceptatie met KPI’s (latency, uptime, kosten/1K tokens)
6) Go‑live + beheer (updates, incidentrespons, rapportage)

## Architectuur (tekstueel)

Clients ↔ `orchestratord` (SSE streaming, cancel, admission) ↔ Adapters (vLLM/TGI/llama.cpp) ↔ GPU‑pools (gepinnde profielen).  
Observability: logs/metrics/traces met duidelijke namen en dashboards.  
Beveiliging: hardened OS, secrets handling, minimaal loggingbeleid, afgesproken retentie.

## Use‑cases

- Interne chat‑assistent met bedrijfscontext (privé).  
- Zoek/QA over interne documenten.  
- Code‑assistent of rapport‑generator met audit‑spoor.  
- Afgeschermde API voor productteams (hybride/on‑prem).

## SLA‑bundels

- Essential (€ 350 p/m): monitoring, maandelijkse updates, kantooruren support, R/T 8/24 uur.  
- Advanced (€ 600 p/m): uitgebreide observability, tweewekelijkse updates, R/T 4/12 uur.  
- Enterprise (€ 900 p/m): weekly updates, optionele on‑call, R/T 2/6 uur, maatwerk rapportage.

## Indicatieve prijzen

- Implementatie: € 4.500 – € 9.500 (scope‑afhankelijk; 2–4 weken).  
- Beheer: € 350 – € 900 p/m per omgeving.  
Prijs is inclusief documentatie, acceptatietesten en overdracht. Hardware niet inbegrepen, wel advies en inkoopbegeleiding.

## Veelgestelde vragen (selectie)

- Welke modellen/engines? Llama‑familie, Mistral, vLLM, TGI, llama.cpp (keuze per use‑case).  
- Werkt het in de cloud? Ja, on‑prem, cloud of hybride; datalokatie blijft EU‑first.  
- Hoe zit het met updates en security? Vast ritme, release‑notes, audit‑log; change‑vensters in overleg.  
- Integratie met bestaande apps? Ja, via stabiele API’s en voorbeeldclients.  
- Trainen jullie modellen? Wij richten ons op inferentie en integratie; fine‑tuning/embedding pipelines in overleg met partners.

## Volgende stap

Plan een discovery call (gratis, 30 min) of vraag een POC‑voorstel aan. Binnen 10 werkdagen na intake ontvangt u een concreet plan met scope, KPI’s en planning.

