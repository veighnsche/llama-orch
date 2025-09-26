# POC‑pakket (tijdbox, vaste prijs) — Draft

Doel: binnen 10 werkdagen aantonen dat “Server Templates + dunne orkestratielaag (`llama‑orch`)” uw use‑case veilig en meetbaar ondersteunt.

## Samenvatting

- Duur: 10 werkdagen (2 weken)  
- Prijs (indicatief): € 2.500 – € 4.500 (fixed, exclusief hardware)  
- Team: 1 implementatie‑specialist (opschaling mogelijk)  
- Locatie: remote + optioneel on‑site

## Deliverables

- Werkende POC‑omgeving met geharde Server Templates (OS, drivers, runtimes)  
- `llama‑orch` geconfigureerd (SSE‑streaming, cancel, determinisme)  
- 1–2 geselecteerde modellen/engines (vLLM/TGI/llama.cpp) geplaatst  
- Observability dashboards (latency/throughput/uptime, kosten/1K tokens)  
- Security baseline: least‑privilege, loggingbeleid, changelog updates  
- Testcases per use‑case + meetrapport (KPI’s)  
- Korte runbook + overdracht call

## KPI’s (voorbeelden)

- Latency: median ≤ 300 ms op kernprompts  
- Uptime: ≥ 99,5% tijdens testwindow  
- Determinisme: reproduceerbare runs (seed/instellingen vastgelegd)  
- Privacy: geen PII naar extern; logging minimal en doelgebonden  
- Kosten: inzicht per 1K tokens, budgetguardrails ingericht

## Scope

- In:  
  - Templates deployen + observability  
  - `llama‑orch` configuratie + voorbeeldclients  
  - 1–2 modellen/engines en basis‑prompts/testcases  
  - Rapportage en overdracht  
- Uitgesloten:  
  - Productie‑hardening voor internet‑exposed endpoints  
  - Integratie met identity providers (optioneel off‑scope)  
  - Custom app‑ontwikkeling (mogelijk in vervolgfase)

## Benodigd van klant

- Hardware (of akkoord op hosting/colocatie)  
- Toegang (accounts/keys) en netwerk/firewall‑inrichting  
- Voorbeelddata/prompts (gesaniteerd)  
- Beslissers aanwezig bij intake en acceptatie

## Planning (10 werkdagen)

1) Intake, security/IT‑scan, KPI’s vastleggen  
2) Templates plaatsen + observability  
3) `llama‑orch` configureren + engines/modellen  
4) Testcases draaien, metrics verzamelen  
5) Rapportage & overdracht

## Vervolg (optioneel)

- Implementatie (2–4 weken) → fixed price  
- SLA‑beheer (Essential/Advanced/Enterprise)  
- Verdere integratie met apps/IDP’s/beleid

## Juridisch

- AV, SLA en verwerkersovereenkomst beschikbaar op verzoek  
- Data blijft in EU; logging minimal/doelgebonden  
- Geheimhouding (NDA) op verzoek

