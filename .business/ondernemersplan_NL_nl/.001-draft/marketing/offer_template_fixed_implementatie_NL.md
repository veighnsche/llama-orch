# Implementatie‑aanbod — Fixed Price (concept sjabloon)

Klant: [Klantnaam]  
Project: [Projectnaam]  
Datum: [Datum]  
Contact: [Naam · E‑mail · Telefoon]

## 1. Samenvatting

Doel: in 2–4 weken een veilige, meetbare on‑prem/hybride AI‑omgeving opleveren op basis van geharde Server Templates + een dunne, controleerbare orkestratielaag (`llama‑orch`).  
Resultaat: productieklare omgeving met observability en SLA‑aansluiting, zonder vendor lock‑in.

Vaste prijs: [€ …] (excl. btw, ex. hardware)  
Betaling: [40% bij start, 60% na acceptatie] (of 50/50)  
Doorlooptijd: [2–4 weken], planning in overleg.

## 2. Scope van werk

In scope:
- Server Templates (GPU): OS‑profielen, drivers, container runtime, security baseline.  
- Provisioning & hardening: scripts, policies, gebruikers/rollen, netwerk/ports.  
- `llama‑orch` configuratie: SSE‑token‑streaming, cancel, determinisme‑opties.  
- Engines/modellen: [vLLM/TGI/llama.cpp] per use‑case (1–2 varianten).  
- Observability: dashboards/logging/metrics (latency, throughput, uptime, kosten/1K tokens).  
- Testen & acceptatie: KPI‑metingen en rapportage.  
- Overdracht: korte runbook + sessie.

Optioneel (meerwerk of vervolgfase):
- Identity/SSO integratie (OIDC/SAML), netwerksegmentatie advanced, mTLS.  
- Maatwerkapplicaties, fine‑tuning pipelines, data‑engineering.  
- Internet‑exposed hardening beyond baseline (WAF, red‑team, etc.).

Buiten scope:
- Hardwarelevering (wel advies/inkoopbegeleiding).  
- Door klant aangeleverde data‑opschoning; juridische beoordeling.

## 3. Deliverables

- Geconfigureerde on‑prem/hybride omgeving (templates + `llama‑orch`).  
- 1–2 engines/modellen geplaatst en getest.  
- Observability dashboards en log/metrics‑retentie ingesteld.  
- Acceptatierapport met KPI‑metingen.  
- Runbook (deploy/update/incident) en overdracht call.

## 4. Acceptatie‑KPI’s (voorbeeld)

- Uptime tijdens acceptatie‑window: ≥ 99,5%  
- Latency (median, kernprompts): ≤ 300 ms  
- Determinisme: reproduceerbare runs (seed/instellingen vastgelegd)  
- Privacy: geen PII naar extern; logging minimal/doelgebonden  
- Kosten: inzicht per 1K tokens; budgetguardrails geconfigureerd

## 5. Planning & mijlpalen

- M1: Intake & technische/security‑scan (dag 1–2)  
- M2: Templates + observability (dag 3–5)  
- M3: `llama‑orch` configuratie + engines/modellen (dag 6–8)  
- M4: Testen & acceptatie op KPI’s (dag 9–10)  
- M5: Go‑live + overdracht; SLA‑aansluiting (dag 10)

## 6. Aannames & afhankelijkheden

- Datalocatie EU; toegang en accounts/keys beschikbaar.  
- Hardware beschikbaar of akkoord op hosting/colocatie.  
- Voorbeeldprompts/data (gesaniteerd) worden aangeleverd.  
- Beslissers aanwezig bij intake en acceptatie.

## 7. Prijs & betalingsvoorwaarden

- Vaste prijs: [€ …] (excl. btw)  
- Betaling: [40/60] via factuur (14 dagen); SLA facturatie per maand vooruit (automatische incasso mogelijk).  
- Reiskosten/locatie: in overleg; standaard remote uitvoering.

## 8. Juridisch & SLA

- AV en SLA beschikbaar; verwerkersovereenkomst op verzoek.  
- Wij leveren een minimalistische orkestratielaag (kleine attack‑surface, auditbaar).  
- Wijzigingen/scope creep: change‑control met prijsafspraak vooraf.

## 9. Risico & mitigatie (uittreksel)

- Capaciteitspieken → inzet freelancers/partners; realistische planning.  
- Security/compatibiliteit → staging, versiebeleid, regressietests.  
- Levering → WIP‑limieten, wekelijkse status, heldere exitcriteria per mijlpaal.

Geldigheid van dit aanbod: [30 dagen].

