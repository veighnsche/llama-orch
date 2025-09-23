# 2.1 Jouw onderneming

| Veld | Inhoud |
|---|---|
| Handelsnaam | Veighnsche |
| Inschrijvingsnummer KVK | {{ kvk }} |
| Vestigingsadres | {{ street }} |
| Vestigingsnummer | {{ vat }} |
| Handelsnamen | Veighnsche · Vinch · Vinsche · Vinsch |
| Omschrijving | Ontwikkeling van software en apps |
| Rechtsvorm | [x] Eenmanszaak |
| Datum inschrijving KVK | {{ kvk_date }} |
| Administratie door | Zelf, met ondersteuning van boekhouder (kwartaalcontrole & jaarrekening) |
| Btw-tarief | [x] 21% |
| Frequentie btw-aangifte | [x] Per kwartaal |
| Btw-id nummer | {{ vat }} |
| Omzetbelastingnummer | {{ omzetbelastingnummer }} |
| Algemene voorwaarden geregeld | [x] Ja · [ ] Nee |
| Vergunningen | [x] Geen vereist · [ ] Vereist, nog niet verkregen: – · [ ] Verkregen: – |
| Subsidies | [x] Nee · [ ] Ja, namelijk: – |
| Verzekeringen | [ ] Nee · [x] Ja, namelijk: Beroepsaansprakelijkheid; Bedrijfsaansprakelijkheid; vrijwillige arbeidsongeschiktheidsvoorziening |

---

# 2.2 Jouw idee

**1. Wat ga je doen?**  
Ik lever kant-en-klare, geoptimaliseerde AI-servers en eigen software om open-source AI in productie te brengen. Eerste product is `llama-orch`: een open-source orkestrator die GPU-resources beheert, modellen plaatst en veilige, deterministische token-streaming via SSE levert. Klanten krijgen een compleet pakket: hardwareselectie/plaatsing, installatie, configuratie, monitoring en support.

**2. Waarom ga je het doen en waarom wordt jouw idee een groot succes?**  
De open-source softwaremarkt kent veel hoogwaardige AI-software. Veel klanten willen dit gebruiken maar kunnen geen (GPU)server opzetten of beheren. Er is duidelijke vraag naar on-premise en EU-compliant oplossingen (privacy, latency, kostenbeheersing, geen vendor lock-in). Door een turnkey aanbod (server + `llama-orch` + beheer) verlaag ik de drempel en verkort ik time-to-value. Succesfactoren: sterke groei van generatieve AI, toenemende voorkeur voor open-source, en mijn technische breedte (frontend/backend/infra) om snel betrouwbare implementaties te leveren.

**3. Hoe ga je het aanpakken?**  
Gefaseerde aanpak: (1) productiseren van `llama-orch` met duidelijke contracten en documentatie; (2) referentie-implementaties op GPU-servers (CachyOS/Arch-profiel), inclusief automatisering van provisioning, logging en updates; (3) aanbod in bundels (implementatie + beheerpakket met SLA); (4) distributie via website, GitHub en netwerk/partners; (5) focus op mkb en bureaus die privacy/controle wensen en snel live willen zonder eigen MLOps-team.

**Pakketten en prijzen (indicatief):**
- Essential: implementatie € 4.500 – SLA € 350 p/m (monitoring, updates maandelijks, kantooruren support).
- Advanced: implementatie € 7.500 – SLA € 600 p/m (uitgebreide observability, updates tweewekelijks, versnelde respons).
- Enterprise: implementatie € 9.500 – SLA € 900 p/m (SLA op maat, weekly updates, optionele on‑call, uitgebreid rapportagepakket).

**Onboardingproces (stappen):**
1) Intake & use‑caseverkenning → 2) Technische scan (hardware, beveiliging, connectivity) → 3) Voorstel & planning → 4) Implementatie in sprints (infra, `llama‑orch`, modellen, observability) → 5) Acceptatie & performancecheck → 6) Go‑live → 7) Nazorg & overdracht documentatie.

**SLA‑niveaus (indicatief):**
- Respons- en hersteltijden: Basic 8/24 uur, Standard 4/12 uur, Premium 2/6 uur (kantooruren; Premium met optionele on‑call).
- Monitoring & updates: proactieve alerts; beveiligingsupdates binnen afgesproken venster; geplande onderhoudsmomenten in overleg.
- Rapportage: maandrapport met uptime, incidenten, performance en verbeterpunten.

**KPI’s & resultaten:**
- Uptime ≥ 99,5%; gemiddelde latency ≤ 300 ms voor kernuse‑cases; throughput en kosten per 1K tokens inzichtelijk.
- MTTR ≤ 6 uur (Premium); klanttevredenheid ≥ 8/10; oplevering binnen 2–4 weken per implementatie (scope‑afhankelijk).

**Risico’s & mitigatie:**
- Technologische veranderingen → versiebeleid, backward‑compatibele contracten, PoC‑validatie.
- GPU‑beschikbaarheid/kosten → alternatieve leveranciers/hostingopties; schaal per behoefte; tijdige inkoop.
- Beveiliging/AVG → hardening, least‑privilege, auditlogs; DPIA‑ondersteuning; datalokatie in EU.
- Capaciteitspieken → inzet freelancers/partners; realistische planning; duidelijke scope en prioritering.

**Compliance & privacy (EU/AVG‑first):**
- Datalokatie EU, geen ongeautoriseerde data‑export; logging conform afspraak (minimaal, doelgebonden).
- Verwerkersovereenkomst en heldere SLA/AV; ondersteuning bij DPIA en security‑checks.

---

## Gebruik van de lening & fasering

Doel van de financiering is directe omzetversnelling en professionele borging van levering/support. Besteding in drie tranches, alleen na mijlpalen:
- T1 — Productisering & salesmateriaal (maand 0–2): documentatie, referentie‑implementaties, website/cases, SLA‑templates.  
- T2 — Delivery‑capaciteit (maand 2–4): ontwikkel‑/demo‑GPU, observability, testomgeving, automatisering provisioning.  
- T3 — Go‑to‑market (maand 3–6): marketing (gericht op niches), partnerprogramma, demo’s/POC’s.

Elke tranche wordt pas aangewend na het behalen van vooraf gedefinieerde doelen (case(s) live, pipeline ≥ N gekwalificeerde leads, SLA‑klanten ≥ X). Hierdoor blijft cash beheerst en aflossingscapaciteit intact.

## Tractie & bewijsstukken

- Open‑source kern: `llama‑orch` (publiek, versiebeheer, changelog).  
- Interne benchmarks en SSE‑transcripten (stabiliteit, latency, kosten per 1K tokens) als bewijs van kwaliteit.  
- Referentie‑implementatie(s) met meetbare KPI’s en klantreacties (worden toegevoegd zodra live).  
- Roadmap en release‑ritme (maandelijks), met security/observability als vaste thema’s.
