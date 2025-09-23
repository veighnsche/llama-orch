# Server Templates + Implementatie + SLA — One‑Pager

## 1. Context & probleem

- AI‑capaciteit is nodig, maar publieke API’s impliceren lock‑in, onvoorspelbare kosten en privacyrisico’s.  
- On‑prem/hybride geeft controle, maar implementatie en beheer zijn specialistisch en tijdrovend.

## 2. Oplossing (in 2–4 weken live)

- Geharde GPU‑Server Templates: OS‑profielen, drivers, container runtime, security baseline, provisioning en observability.  
- Implementatie: modellen/engines (vLLM/TGI/llama.cpp), deterministische SSE‑streaming, policies, tests en overdracht.  
- Beheer (SLA): monitoring, updates/patching, incidentrespons en maandrapportage.

## 3. Dunne orkestratielaag (`llama‑orch`)

- Minimalistisch, controleerbaar en auditbaar; kleine attack‑surface.  
- SSE‑token‑streaming met cancel en tijdslimieten; determinisme‑opties end‑to‑end.  
- Heldere metrics/logs; open‑source transparantie; EU/AVG‑first.

## 4. Resultaat & KPI’s

- Uptime ≥ 99,5%, MTTR ≤ 6 uur (Premium).  
- Median latency ≤ 300 ms voor kernuse‑cases.  
- Acceptatie in 1e poging (≥ 95%).  
- Inzicht in kosten per 1K tokens en throughput.

## 5. Pakketten & prijzen (indicatief)

- Implementatie: € 4.500 – € 9.500 (2–4 weken, scope‑afhankelijk).  
- SLA: Essential € 350 p/m · Advanced € 600 p/m · Enterprise € 900 p/m.  
- Facturatie: maand vooruit (SLA), automatische incasso beschikbaar.  
- Hardware: niet inbegrepen; wel advies en inkoopbegeleiding.

## 6. Werkwijze (stappen)

1) Intake & technische scan  
2) Templates plaatsen + observability  
3) `llama‑orch` configureren (SSE/determinisme/policies)  
4) Modellen/engines deployen en testen  
5) Acceptatie op KPI’s  
6) Go‑live + beheer (SLA)

## 7. Differentiatie (USP’s)

- EU/AVG‑first, datasoevereiniteit en DPIA‑ondersteuning.  
- Geen vendor lock‑in; open‑source componenten; eigenaarschap bij u.  
- Deterministische output en robuuste streaming; observability standaard.  
- Snelheid door herbruikbare templates en runbooks; duidelijke SLA’s.

## 8. Roadmap (selectie)

- Adapters: vLLM/TGI/Triton; pool‑manager; heterogene GPU‑scheduling.  
- Security & policy: artifact registry, SBOM, audit‑logging, policy guardrails.  
- Observability: GPU/NVML metrics, structured tracing.

## 9. Next steps

- Plan een discovery call (gratis, 30 min).  
- Vraag een POC‑voorstel aan (tijdbox, vaste prijs, meetbare KPI’s).  
- Ontvang binnen 5–10 werkdagen een implementatieplan met scope en planning.

