# Checklist DPIA — LLM On‑Prem (NL) — Draft

Doel: snel toetsen of een on‑prem/hybride LLM‑implementatie privacy‑by‑design, auditbaar en AVG/NIS2‑conform is ingericht. Gebruik als intake/acceptatiehulp; geen juridisch advies.

## 1. Rollen & verantwoordelijkheid
- [ ] Verwerkingsverantwoordelijke en verwerker(s) benoemd  
- [ ] DPO/FG (indien van toepassing) betrokken  
- [ ] Contactpunten en escalatieprocedure vastgelegd

## 2. Doelen, data & grondslag
- [ ] Verwerkingsdoelen beschreven (use‑cases)  
- [ ] Categorieën persoonsgegevens (incl. mogelijk bijzonder) benoemd  
- [ ] Rechtsgrond(en) bepaald  
- [ ] Data‑minimalisatie toegepast; prompts/antwoorden gesaneerd waar nodig  
- [ ] Bewaartermijnen gedefinieerd

## 3. Datalocatie & stromen
- [ ] Datalocatie uitsluitend EU (on‑prem/colocatie/cloud EU‑regio’s)  
- [ ] Geen overdracht naar derde landen zonder passende waarborgen  
- [ ] Dataflow diagram (ingang→LLM→uitgang) beschikbaar  
- [ ] Transfer Impact Assessment (TIA) indien relevant

## 4. Beveiliging & toegang
- [ ] Hardened OS en netwerksegmentatie  
- [ ] TLS (in‑transit) en encryptie at rest  
- [ ] Secrets management en key‑beheer  
- [ ] Least‑privilege RBAC; JIT/expiring credentials  
- [ ] Patch/update beleid vastgelegd (incl. dependencies)

## 5. Logging, audit & observability
- [ ] Logging minimal en doelgebonden (geen onnodige PII)  
- [ ] Audit‑trail voor toegang, modelconfig en wijzigingen  
- [ ] Metrics/dashboards voor latency, throughput, uptime, kosten/1K tokens  
- [ ] Logretentie/anonimisering conform beleid  
- [ ] Toegangslogs privacyvriendelijk opgeslagen

## 6. Determinisme & modelgedrag
- [ ] Seed/instellingen vastgelegd voor reproduceerbaarheid  
- [ ] SSE‑streaming met cancel‑semantiek  
- [ ] Prompt redaction/guardrails beleid  
- [ ] Output review‑procedure bij risicovolle use‑cases

## 7. DPIA‑criteria & risico’s
- [ ] DPIA‑trigger(s) beoordeeld (grootschalig, systematisch, kwetsbare groepen)  
- [ ] Risico’s gecategoriseerd (privacy, security, bias, misbruik)  
- [ ] Maatregelen en rest‑risico’s gedocumenteerd  
- [ ] Goedgekeurd door bevoegde rol(len)

## 8. Rechten van betrokkenen
- [ ] Procedures voor inzage, rectificatie, verwijdering  
- [ ] Data‑export en verwijdering mogelijk  
- [ ] Contactpunt ingericht en getest

## 9. Verwerkers & contracten
- [ ] Verwerkersovereenkomst(en) aanwezig  
- [ ] Subverwerker(s) lijst en due diligence  
- [ ] SLA/KPI’s vastgelegd (uptime, R/T, security updates)  
- [ ] Acceptable Use Policy en incidentresponsbeleid

## 10. Incidenten & continuïteit
- [ ] Datalek‑proces en meldplichten  
- [ ] Backup/restore getest  
- [ ] Continuïteitsplan (capaciteit, vervanging hardware)  
- [ ] Monitoring/alerts operationeel

## 11. Evidence (bijlage)
- [ ] Architectuurdiagram en datastromen  
- [ ] Config snapshots (`llama‑orch`, engines, policies)  
- [ ] SSE‑transcripten en benchmarkrapport  
- [ ] Voorbeeld loguitdraaien (geanonimiseerd)  
- [ ] Toegangsbeleid/RBAC en key‑beheer documentatie

