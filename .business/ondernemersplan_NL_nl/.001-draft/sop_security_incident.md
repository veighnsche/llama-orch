# SOP — Security & Incident Response — Draft

Doel: snelle, proportionele respons; minimaliseren impact; voldoen aan meldplichten; structurele verbeteringen borgen.

## Rollen & severities
- Roles: Incident lead, Implementatie engineer, Klant verantwoordelijke, (optioneel) DPO/FG.
- Severity (S0–S3):
  - S0 Kritiek: data‑exfiltratie bevestigt / productie down
  - S1 Hoog: security‑bug met misbruikpotentieel / major degradatie
  - S2 Middel: beperkte impact, mitigatie beschikbaar
  - S3 Laag: observatie/verbetering, geen klantimpact

## Proces (samengevat)
1) Detectie & triage → classificeer (S0..S3)  
2) Containment → access beperken, keys roteren, segmenteren  
3) Eradicatie & herstel → patch/update, config fix  
4) Communicatie → klantstatus (R/T conform SLA), meldplichtcheck  
5) Post‑mortem → oorzaken, acties, deadlines, eigenaar  
6) Preventie → hardening/monitoring/policy updates

## Maatregelen (baseline)
- Hardening: OS, firewall, minimal services; least‑privilege RBAC.  
- Secrets: rotate/jit/expiring; versleutelde opslag.  
- Logging: minimal/doelgebonden; audit‑trail; retentiebeleid.  
- Updates: vaste cadence; kritieke patches versneld.  
- Backups/restore: periodiek testen; offline kopie.

## Meldplichten & privacy
- Datalek‑beoordeling en documentatie; AVG/NIS2 toets.  
- Meldingsproces met termijnen; betrokkenen informeren indien vereist.  
- DPIA‑update bij relevante wijzigingen.

