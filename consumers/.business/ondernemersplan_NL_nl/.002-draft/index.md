# Ondernemingsplan — Draft 2 (Index & Navigatie)

Doel van deze index: het volledige ondernemingsplan (exclusief financieel plan) in één samenhangende, scanbare uitleg neerzetten, met duidelijke links naar onderliggende stukken en naar de live website. Draft 2 focust op structuur, formatting en bundeling; inhoud blijft conform Draft 1.

— In scope: propositie/markt/aanbod/POC/SLA/kwaliteits- en risicobeheersing/transparantie/website.  
— Out of scope: financieel plan (staat los, zie /plans/financieel-plan op de site).

## 1. Propositie & USPs (samenvatting)
- Aanbod: “Server Templates + Implementatie + SLA” — met `llama‑orch` als dunne, controleerbare laag (determinisme, SSE, observability).  
- USPs: EU/AVG‑first, geen lock‑in, snelle implementatie (2–4 weken), transparante SLA‑bundels, radicale transparantie (roadmap/changelog/incident‑reviews publiek), GPL‑licentie (auditbaarheid).  
- Referentie: .001-draft/2_7_usps_roadmap_en_kpi.md

## 2. Doelgroep & Markt
- Doelgroepen: legal/professional services (privacy/AVG), agencies/SI (co‑delivery/white‑label).  
- Markt: on‑prem/hybride AI groeit door datasoevereiniteit en kostencontrole; open‑source stacks winnen.  
- Differentiatie: productized delivery + SLA‑backline i.p.v. losse consultancy.  
- Referenties: .001-draft/2_3_4_markt_en_swot.md, .001-draft/2_8_acquisitie_en_partners.md

## 3. Aanbod & Werkwijze
- Server Templates (gehard OS/drivers/runtimes + observability), Implementatie (plaatsing, policies, tests), SLA‑beheer (monitoring, updates, incidentrespons, rapportage).  
- Doorlooptijd: 2–4 weken (intake → templates → orchestratie → engines → test → acceptatie → go‑live).  
- Referenties: .001-draft/2_1_onderneming_en_idee.md, .001-draft/2_5_6_marketingmix_en_doelen.md, .001-draft/ops/acceptance_tests.md, .001-draft/sop_implementation.md

## 4. POC‑pakket (tijdbox)
- Doel: bewijzen in 10 werkdagen met KPI‑meting (latency/uptime/determinisme) + rapportage en overdracht.  
- Scope IN/UIT, deliverables, planning en klantbenodigdheden uitgewerkt.  
- Referentie: .001-draft/marketing/poc_pakket_NL.md

## 5. Prijzen & SLA‑bundels (indicatief)
- Implementatie: € 4.5k – € 9.5k (scope‑afhankelijk; 2–4 weken).  
- SLA’s: Essential/Advanced/Enterprise met oplopende R/T, observability en updates.  
- Transparantie + FAQ verkorten het beslistraject.  
- Referenties: .001-draft/2_5_6_marketingmix_en_doelen.md, .001-draft/marketing/landing_page_NL.md

## 6. Go‑to‑market & Acquisitie
- Retainer + partners (co‑delivery/white‑label) voor voorspelbare pipeline; ABM mini‑LP’s voor 10–20 target accounts.  
- Lead magnets (DPIA/On‑Prem Readiness) met nurture‑flows; retargeting op /poc en /pricing.  
- Sales SLA: inbound reactie binnen 24u met plannerlink, CRM‑discipline.  
- Referenties: .001-draft/2_10_advertentieplan.md, .001-draft/ops/sales_sla.md, .001-draft/marketing/email_nurture_sequences.md, .001-draft/marketing/abm_playbook.md

## 7. Website (v1 live)
- Platform: Cloudflare Pages + Hono JSX (SSG).  
- Routes: Home, Legal, Agency, POC, Pricing, Checklists, SLA, About, Contact, Status, Plans, Transparantie.  
- Plannen als MDX op /plans/* met automatische PDF‑downloads (CI via Puppeteer).  
- Status/KPI‑badges + links naar bewijsstuk‑bundel.  
- Referenties: .001-draft/website/overview.md, routes_sitemap.md, docs_mdx_pipeline.md, pdf_generation.md, build_deploy_cloudflare.md

## 8. Transparantie & Licentie
- GPL (bij voorkeur GPLv3) voor kernsoftware; code en wijzigingen publiek; geen lock‑in.  
- Radicale transparantie: roadmap/changelog/incidenten/status en plannen (MDX + PDF) op de site.  
- Referenties: .001-draft/legal/license_policy.md, .001-draft/security/oss_license_compliance.md

## 9. Kwaliteit & Risico’s
- Kwaliteit: acceptatietests, observability, post‑mortems; SLO→SLA mapping; patch/vuln‑beleid.  
- Risico’s: pipeline, piekbelasting, compat/security, leveranciers; mitigatie via retainer/partners, freelancers/WIP‑limieten, staging/hardening, alternatieve leveranciers, buffers + auto‑incasso.  
- Referenties: .001-draft/ops/slo_to_sla_mapping.md, .001-draft/security/*, .001-draft/2_9_risicoanalyse_en_mitigatie.md

## 10. Bewijs & Bijlagen (hub)
- Proof bundle: benchmarks/SSE‑transcripten/dashboards; docs control; cases/templates.  
- Bijlagen: AV, SLA, DPA, privacy/cookies, verwerkingsregister, dataretentie, subverwerkers.  
- Referenties: .001-draft/proof_bundle_index.md, .001-draft/4_bijlagen_en_bewijs.md

---

## A. Navigatie (website) in één oogopslag
- Top‑nav: Home • Legal • Agency • POC • Pricing • Checklists • SLA • About • Contact  
- Footer: Status • Plans • Bijlagen • Transparantie • Privacy • Cookies • AV • GitHub  
- Interlinks per pagina (voorbeelden):  
  - Legal → POC • Checklists • Plans#privacy  
  - POC → Pricing • Plans#acceptatie‑kpis • Contact  
  - Plans → download PDF’s • Transparantie  
- Zie: .001-draft/website/navigation_ergonomics.md

## B. Voorstel nieuwe structuur (Draft 2) — “bundels” (max diepte ≤ 4)
Doel: gelijke onderwerpen bundelen, consistent formatteren, één index per bundel. In Draft 2 voeren we primair herstructurering en formatting door; inhoud zelf verandert niet.

```
.002-draft/
  index.md        # dit document (narrative + wegwijzer)
  bundles/
    01_overview/README.md          # missie, waarden, USPs, propositie
    02_market_gtm/README.md        # markt, SWOT, acquisitie, advertentie
    03_product_delivery/README.md  # aanbod, POC, SLA, SOP implementatie
    04_quality_ops/README.md       # SLO→SLA, observability, patch/vuln, incident
    05_risk_mitigation/README.md   # risicoanalyse + mitigaties
    06_transparency_gpl/README.md  # GPL/licenties, openheid, status/metrics
    07_website/README.md           # website routes/IA, MDX/PDF, CI
    08_proof_hub/README.md         # bewijs, bijlagen, templates
```

Richtlijnen:
- Max diepte ≤ 4 (zoals hierboven): draft → bundles → topic → README.md  
- Eén README.md per bundel met verwijzingen naar relevante .001‑draft‑bronnen (bron van waarheid), tot we inhoud naar .002 verplaatsen.  
- Consistente headings (H1–H3), ToC blok, “Related links” per bundel.  
- Geen financiële inhoud in deze draft; dat blijft in het financiële plan.

## C. Wat verandert in Draft 2 (zonder herontwerp site)
- Bundelen + formatteren (consistentie, ToC, interlinks).  
- Verwijzingen naar live website en plannen (MDX + PDF).  
- Geen herontwerp website; we beschrijven kort de IA en linken naar de planbestanden.  
- Proof bundel krijgt screenshots/analytics van site en downloadknoppen.

---

## D. Korte FAQ
- “Waar vind ik het financieel plan?” → op de site onder /plans/financieel‑plan (PDF download), of in de financiële bundel (losstaand).  
- “Waarom GPL en radicale transparantie?” → auditbaarheid, vertrouwen en snellere adoptie; commerciële waarde via implementatie/SLA.  
- “Hoe snel live?” → 2–4 weken, afhankelijk van scope; POC binnen 10 werkdagen.
