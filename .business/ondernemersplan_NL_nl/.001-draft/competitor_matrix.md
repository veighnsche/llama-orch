# Concurrentiematrix (Draft)

| Categorie | Voorbeelden | Sterktes | Zwaktes | Wat wij anders doen |
|---|---|---|---|---|
| Publieke AI‑API’s | OpenAI, Azure OpenAI | Snel starten, ecosysteem, schaal | Data/locatie onduidelijk, lock‑in, hogere TCO bij volume, beperkte determinisme/audit | On‑prem/EU‑first, eigenaarschap data, deterministische SSE, transparante kosten/observability |
| Managed AI platforms | Anyscale, Modal, Baseten | Managed infra, auto‑scale, tooling | Cloud‑only focus, beperkte on‑prem, lock‑in | Server Templates + implementatie on‑prem/hybride; dunne, controleerbare laag |
| Point solutions (inference servers) | vLLM, TGI | Snel model serveren, performance | Geen turnkey provisioning/observability/SLA, integratiecomplexiteit | Templates + observability + SLA, deterministische streaming, policy/guardrails |
| Consultants (maatwerk) | zzp/IT‑bureaus | Flexibel, domeinkennis | Minder productgestuurd, variabele kwaliteit, langere lead time | Productized delivery (2–4 weken), SLA‑backline, runbooks/templates |

Kern‑differentiatie:  
- EU/AVG‑first on‑prem/hybride, dunne orkestratielaag (`llama‑orch`) voor veiligheid/determinisme/observability.  
- Server Templates → voorspelbare implementaties en time‑to‑value.  
- Heldere SLA‑bundels, automatische incasso, maandrapportage met KPI’s.

