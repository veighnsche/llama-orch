# SOP — Implementatie (2–4 weken) — Draft

Doel: voorspelbare oplevering met duidelijke acceptatie‑KPI’s, minimale risico’s en overdraagbare documentatie.

## Rollen
- Lead implementatie, Klant contactpersoon, (optioneel) Security/IT, (optioneel) Partner engineer

## Stappen & checks
1) Intake & scope (dag 1–2)
   - Use‑cases, KPI’s, dataflow, security eisen, hardware/hosting
   - Beslissers & acceptatiecriteria bevestigd; NDA/VO indien nodig
2) Templates & observability (dag 3–5)
   - OS/driver/runtime profielen, provisioning, hardening
   - Dashboards: latency/throughput/uptime, logretentie ingesteld
3) `llama‑orch` configureren (dag 6)
   - SSE‑streaming (cancel), determinisme, policies/guardrails
   - Secrets/keys en RBAC (least‑privilege)
4) Engines/modellen & tests (dag 7–8)
   - vLLM/TGI/llama.cpp gekozen per use‑case
   - Testcases/benchmarks; SSE‑transcripten als bewijs
5) Acceptatie & overdracht (dag 9–10)
   - KPI‑rapport; runbook (deploy/update/incident)
   - Go‑live plan en change‑venster

## Deliverables (minimaal)
- Geconfigureerde omgeving + dashboards  
- Acceptatie‑rapport met KPI’s en transcripten  
- Runbook en configuratie snapshots  
- Backlog/roadmap voor vervolgstappen

## Exitcriteria
- KPI’s gehaald of expliciet afgeweken met akkoord  
- Documentatie overgedragen  
- Go‑live akkoord of vervolgtraject gepland

