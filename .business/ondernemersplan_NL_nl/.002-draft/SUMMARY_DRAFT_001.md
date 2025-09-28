# SUMMARY_DRAFT_001 — Complete, Unfiltered Synthesis (Monolithic)

Purpose: this is a gigantic, unfiltered, first-pass synthesis of every document in `.001-draft/`. It intentionally errs on the side of over-inclusion and repetition so that Draft 2 can selectively filter, reshape tone/identity, and target specific audiences with sharper messaging.

---

## Document Inventory (source map)

- 00_cover_sheet_krediet.md
- 1_de_ondernemer.md
- 2_1_onderneming_en_idee.md
- 2_3_4_markt_en_swot.md
- 2_5_6_marketingmix_en_doelen.md
- 2_7_usps_roadmap_en_kpi.md
- 2_8_acquisitie_en_partners.md
- 2_9_risicoanalyse_en_mitigatie.md
- 2_10_advertentieplan.md
- 3_financieel_plan.md
- 3_1b_12mnd_liquiditeit.md
- 4_bijlagen_en_bewijs.md
- competitor_matrix.md
- doc_control.md
- index.md (ToC)
- mermaid_visuals.md (visual summaries)
- proof_bundle_index.md
- qredits_reporting_template.md
- risk_register.md
- sop_implementation.md
- sop_security_incident.md
- unit_economics.md
- legal/* (AV, DPA, SLA, privacy/cookies, NIS2, dataretentie, verwerkingsregister, subverwerkers, licentiebeleid, DPIA voorbeeld)
- marketing/* (landing pages, checklists, nurture, pitch, POC, one-pager, outreach, ABM, kanaalbudget)
- gtm/crm_data_governance.md
- finance/sepa_mandate_template.md

---

## Identity, Entrepreneur Profile, Reliability (1_de_ondernemer.md)

- Personal profile: Vince (NL), independent founder/developer with broad front-end to AI integration experience; recent focus on open-source LLM orchestration (`llama-orch`).
- Motivation: enable SMB and (semi-)public orgs to adopt AI safely, quickly, affordably, and with full control (on‑prem/EU-first). Combine craftsmanship with measurable service (SLA, uptime, latency).
- Work history highlight: iO Digital (React/Vue for NN Group, Stedin), earlier front-end roles and freelance; since 2024 building products and PoCs around LLMs.
- Strengths: analytical, broad technical stack (frontend/backend/infra/AI), creative, disciplined processes, clear communication. Weaknesses: perfectionism, difficulty delegating, risk of over-commitment, sales network in build-up.
- Reliability & payment discipline: automatic direct debit (SEPA) for SLA; monthly VAT reserve; 3+ months operating buffer; privatized draw policy (target €2,000/m; guardrail to reduce if DSCR < 1.2×). Qredits-ready with bank statements, BKR, IB.
- Evidence mindset: runbooks, DoD, post-mortems; incident reviews. Visual mindmap for strengths/mitigations (mermaid).

---

## Proposition & Company Idea (2_1_onderneming_en_idee.md)

- Productized offering: “Server Templates + Implementatie + SLA” with `llama-orch` as a thin, auditable orchestration layer (SSE, cancel, determinism, observability) between org and open-source runtimes (vLLM, TGI, llama.cpp). Intent: keep the layer minimal, control and auditability maximal.
- Why now (success basis): open-source LLM stacks mature; customers want control (privacy, latency, cost, no lock-in). Many lack GPU/MLOps skills; desire turnkey on-prem/hybrid with SLA.
- How (phases): productize `llama-orch` with contracts/docs; create reference implementations; bundle fixed-price implementation + SLA; distribute via website/GitHub/partners; focus on SMB and agencies needing quick, safe delivery.
- Packages (indicative): Essential (€4.5k impl, €350/m SLA), Advanced (€7.5k impl, €600/m), Enterprise (€9.5k impl, €900/m). SLA options escalate in observability and response times.
- Onboarding steps: intake → scan → templates/observability → orchestration → engines/models → acceptance → go-live + runbook + change windows.
- KPIs: uptime ≥ 99.5%, median latency ≤ 300ms (core use-cases), MTTR ≤ 6h (Premium), customer sat ≥ 8/10, time-to-live 2–4 weeks.
- Risks & mitigations (shortlist): tech change → versioning/PoC; GPU cost/availability → alternates; security → hardening/policy; capacity peaks → freelancers/partners.
- Compliance: EU/EAA-first data location; DPIA support; minimal, purpose-bound logging.
- Progress: early kernel of `llama-orch` with routes/SSE and spec-first; docs/test harness; roadmap; reduces time to first paid implementations (60–90 days target).
- Transparency & licensing: radical transparency (roadmap/changelog/incident reviews), plans as MDX + PDF; GPL (pref. GPLv3) for software; services monetize implementation/SLA.

---

## Market, Trends, Competitors, SWOT (2_3_4_markt_en_swot.md, competitor_matrix.md)

- Market trajectory: strong growth in on-prem/hybrid generative AI; key drivers are privacy, latency, cost control, no lock-in, and mature open-source tooling; implementation remains non-trivial → demand for turnkey, auditable solutions.
- Trends: hybrid/on-prem, kv-cache/quantization progress, production-grade streaming/cancel, standardization in observability, EU compliance (AVG/NIS2), commercial support for open-source.
- Local/EU context: SMB/public sectors in NL/EU want control/evidence; appetite for solutions that are safe, compliant, and quick-to-value. Agencies/SIs ready to integrate on-prem AI via co-delivery.
- Competitor map:
  - Public AI APIs: fast start, ecosystem; downsides: data location, lock-in, TCO at volume, limited determinism/auditability.
  - Managed platforms: infrastructure offloaded; often cloud-only; limited on-prem option.
  - Point solutions (vLLM/TGI): great perf; lack turnkey provisioning/observability/SLA.
  - Consultants: flexible; less productized, unpredictable lead-times/quality.
- Differentiation: EU/AVG-first, on-prem/hybrid; thin orchestration layer (`llama-orch`) for deterministic SSE and observability; templates/runbooks; explicit GPU placement; clear SLA bundles; transparent costs.
- SWOT highlights:
  - Strengths: broad technical skill; open-source credibility; fast, product-driven delivery; clear KPIs.
  - Weaknesses: small founding team; commercial engine in build-up; capacity peaks.
  - Opportunities: regulation nudges on-prem; partnerships; standardized bundles; reference cases.
  - Threats: pace of change; big platform competition; GPU supply/price volatility; sales cycle length.
  - Approach: double down on strengths (open-source, measurable PoCs), professionalize sales cadence, partner programs, control costs and build buffer.

---

## Marketing Mix, Goals, and Execution (2_5_6_marketingmix_en_doelen.md, 2_10_advertentieplan.md)

- Product: server templates, implementation, and SLA management; `llama-orch` ensures determinism/streaming/observability.
- Price: implementation €4.5–€9.5k; SLA €350–€900/m; TCO competitive for sustained usage vs public APIs.
- Place: Breda remote office; on-site as needed; website/GitHub/partners/LinkedIn/events.
- Promotion: website with cases, GitHub transparency, thought leadership, partner marketing, retainer-driven outreach; landing pages and checklists.
- Personnel: no hires initially; scale with freelancers/partners; strong planning and WIP limits.
- Mission/vision: enable safe, open-source AI adoption with transparency and client ownership; in 5–10 years, be a recognized EU-first orchestrator with partner ecosystem and stable SLA MRR.
- Quarterly targets (year 1): pipeline ≥12 SQLs/quarter; ≥6 offers @ ≥40% win; 1–3 implementations/quarter; SLA Q1 3 → Q4 9–11; MRR ramps accordingly; churn ≤5%; NPS ≥ 8/10.
- Advertentieplan specifics:
  - Proposition: “Snel en veilig open‑source AI op eigen servers” with deterministic SSE and observability.
  - ICP segments: legal/professional (DPIA/audit), agencies/SIs (co-delivery), education/business services.
  - CTAs: discovery call, POC package, SLA bundles, checklist downloads.
  - Channels: LinkedIn (Awareness/Consideration/Retargeting), Search, GitHub/Dev, newsletter/webinars.
  - Vertical creatives and keywords for legal vs agencies; DPIA checklist as magnet.
  - Funnel KPIs with mermaid flow (CTR 1.5–2.5%; Leads 3–6%; SQL 35–45%; etc.).
  - Measurement & attribution: UTM, event tracking, CRM reporting cadence.
  - A/B test matrix: message (privacy vs cost vs time), creative (video/carousel), offer (POC vs checklist).
  - Budget split indicative: LI 50%, Search 30%, Retarget 10%, Experiments 10% with monthly realloc.

---

## Acquisition & Partners (2_8_acquisitie_en_partners.md, marketing/outreach_*.md, marketing/abm_playbook.md)

- Sales engine:
  - Retainer budget €1,200/m × 6 months to bootstrap predictable SQLs and meetings.
  - Deliverables per month (indicative): ≥60 relevant outreach contacts, ≥8 SQLs, ≥4 decision-maker meetings; monthly reporting and exit criteria.
- Partners:
  - Agencies/SIs with referral or co-delivery models; enablement via demo envs, runbooks, kits, webinars.
  - Joint case-building; white-label option; dedicated partner LP and referral flow.
- ABM approach: 10–20 target accounts; mini-LPs with personalization; KPI tracking per account (visits → SQL → wins).
- Outreach templates (NL): email and LinkedIn sequences for legal and agency verticals.

---

## Offering Details, Packaging & Sales Collateral (marketing/*)

- One-pager and landing copy (NL): “Open‑source AI op eigen servers. Veilig. Snel. Meetbaar.” Clear CTAs.
- Vertical LPs (legal, agencies/SIs): EU/AVG-first, DPIA support, deterministic SSE, audit logging, SLA backline, co-delivery.
- Pitch deck outline: problem, solution, product, value, cases, pricing, roadmap, partners, CTA.
- POC package (10 working days): fixed price, KPI-driven; deliverables include templates, `llama‑orch` config, 1–2 engines/models, dashboards, tests, runbook.
- Offer template (fixed price): scope in/out, deliverables, acceptance KPIs, milestones (40/60 or 50/50 billing), assumptions and legal notes.
- Checklists: On‑Prem Readiness and DPIA On‑Prem NL.
- Email nurture sequences: DPIA lead, POC intent, partner enablement (3-touch sequences).
- Channel budget plan: LI 50 / Search 30 / Retarget 10 / Experiments 10; 4‑week realloc cadence.

---

## SOPs & Delivery (sop_implementation.md, sop_security_incident.md)

- Implementation SOP (2–4 weeks): roles, steps, acceptance, deliverables, exit criteria. Emphasis on observability, SSE/determinism, RBAC/least‑privilege, tests, runbook.
- Security & Incident SOP: roles, severities (S0–S3), lifecycle (detect→contain→eradicate→communicate→post‑mortem→prevent), baseline measures, privacy and notification obligations. DPIA updates when relevant.

---

## Legal & Compliance (legal/*, gtm/crm_data_governance.md)

- AV (terms): scope/changes, availability/maintenance, fees/payments, liability caps, IP/GPL, confidentiality/data protection, term/termination, jurisdiction.
- SLA (concept): SLO/SLA tiers; incidents/severities; reporting cadence; maintenance; credits/exclusions; exit and data return.
- DPA (verwerkersovereenkomst): parties/scope; duration/nature/purpose; data categories; processor obligations; subprocessors; security measures; incident notifications; data subject rights; audits; transfers; end‑of‑term data return.
- Policies and registers: privacy policy, cookie notice, processing register, data retention policy (purpose-bound and minimal logging), subprocessor list (EU preference), NIS2 applicability note (best practices even if outside direct scope).
- License policy: GPL (pref. GPLv3) for code; services monetize delivery/support; OSS compliance and notices; “NO WARRANTY” disclaimer consistent with GPL.
- CRM data governance: fields and purposes, retention (12–24 months for leads), fulfillment of data subject rights.

---

## Risk Management (2_9_risicoanalyse_en_mitigatie.md, risk_register.md)

- Categories: commercial/pipeline, operational/capacity, technical/compatibility/security, suppliers/GPU, legal/compliance, financial/liquidity, reputation/churn.
- Mitigations: retainer/partners/ABM; freelancers for peaks; spec‑first and staging; hardening and patch policies; alternative suppliers; buffers and auto‑incasso; proactive SLA reporting; website tactics to boost SQLs.
- Risk register (concept): R‑001 pipeline < target, R‑002 security incident, R‑003 late payments; owners and triggers to be assigned.

---

## Financials, Liquidity, Unit Economics (3_financieel_plan.md, 3_1b_12mnd_liquiditeit.md, unit_economics.md)

- Financing ask (tie‑in with cover sheet): €30,000 loan (48m, ~8%), own €700; purposes: productization, GPU/dev env, observability, marketing, B2B retainer, working capital.
- Key lender metrics: loan monthly ~€731; base SLA coverage @ 6 customers €2,700/m; fixed OPEX ~€1,195 + ~€400 var; buffer ≥ 3 months; DSCR base ~1.5×, with modest quarterly project flow ~2.3–2.9×; private draw policy €2,000 with DSCR guardrail.
- Investment breakdown: hardware, development, observability, marketing, retainer, working capital (see pie chart in visuals).
- Exploitatie (year 1, indicative): projects €36k, SLA €32.4k, prototyping/consult €10k → ~€78.4k revenue; cost base ~€20k; result pre-tax ~€58k (indicative).
- Opex per month (indicative): ~€1,195 fixed + ~€400 variable (scales with growth);
- Break-even: excl. private at 6 SLA; incl. private ~10 SLA.
- Liquidity plan (12 months): SLA ramp 3→11; projects in specific months to thicken buffer; interpretation and DSCR guardrails documented.
- Scenario analysis (DSCR): downside vs base vs conservative with 1 small project/quarter; explicit mitigation tactics.
- Unit economics: SLA gross margin ~€400/m (≈89%); LTV €7.2k–€14.4k; CAC via retainer ~€600–€1,200 depending on win velocity; break-even logic reiterated; payment/dunning policy.
- Collateral & covenants (indicative): pledge of business inventory (GPU workstation), quarterly reporting, minimum liquidity covenant, notice on QoQ revenue dips.

---

## Credit Cover Sheet & Lender View (00_cover_sheet_krediet.md)

- Ask & allocation: €30k loan; total investments €30.7k (own €700); detailed table for hardware/dev/observability/marketing/retainer/working capital.
- Tranches & gates: spending only after gates (demo + 2 qualifications; 1 POC started or implement scheduled; 4+ SQL and ≥3 SLAs active). Ensures cash discipline and repayment likelihood.
- Securities and reporting: pledge of inventory; quarterly MRR/SLA, pipeline, liquidity, DSCR; covenants (buffer ≥ 2 months; notify >20% QoQ drop).
- Reliability package: payment behavior, VAT reserve, auto‑incasso, private draw guardrails; fallback through variable cost cut + freelance capacity.

---

## Proof Bundle & Attachments (4_bijlagen_en_bewijs.md, proof_bundle_index.md)

- Checklist of evidences: identity/KvK, bank/budget docs, BKR, loan terms and SEPA, product repo links/changelogs, benchmarks and SSE transcripts, observability screenshots, contracts (AV/SLA/DPA), cases, sales collateral (pitch/one‑pager/LPs).
- Ops/Security/Compliance docs: SLO→SLA mapping, patch policy, vuln mgmt, backup/RTO/RPO, keys rotation, incident comms, SBOM overview, OSS license compliance.
- Governance & finance: document control, risk register, CRM governance, SEPA mandate template.

---

## Website & Content (marketing/landing_page_NL.md, landing_vertical_*_NL.md)

- Platform (in broader plan): Cloudflare Pages + Hono JSX; pages for Legal, Agency/SI, POC, Pricing, Checklists, SLA, About, Contact, Status, Plans, Transparantie. MDX → PDF pipeline for plans.
- Copy themes: EU/AVG‑first, no lock‑in, deterministic SSE, auditable logs, predictable TCO, 2–4 week implementation, SLA backline; clear CTAs to call/checklist/POC.
- Architecture text: `orchestratord` with SSE + adapters (vLLM/TGI/llama.cpp) + GPU pools + observability and policies; hardened OS and minimal, purpose-bound logging.

---

## Visuals (mermaid_visuals.md)

- Investment pie (breakdown of €30.7k).
- 12‑month roadmap (gantt) for productization, references, adapters/scheduling, GTM.
- Funnel flow with KPI bands; SLA mindmap; liquidity timelines; tranche gates vs milestones.
- Architecture flow: thin orchestration core connected to SSE/adapters/GPU/observability/policies.

---

## Governance & Templates (doc_control.md, qredits_reporting_template.md, finance/sepa_mandate_template.md)

- Document control: semantic versioning, changelog discipline.
- Qredits quarterly report template: MRR, pipeline, funnel metrics, incidents, roadmap, and attachments.
- SEPA mandate template: consent fields and text for recurring SLA auto‑debit.

---

## Big-Picture Claims (rolled-up)

- EU/AVG-first on‑prem/hybrid AI with open-source core; thin, auditable orchestration (`llama‑orch`) designed for determinism (SSE) and strong observability; security-by-default (hardening, RBAC, minimal logging); predictable delivery (2–4 weeks) and service (SLA tiers).
- Commercial model aligns with lender comfort: auto‑incasso, buffers, tranche gates linked to milestones, DSCR guardrails, quarterly reporting, fallback levers (variable cost adjustments, freelance capacity), and acquisition scaffolding (retainer, partners, ABM) to drive SQLs and wins.
- Value proof targets: benchmarks, SSE transcripts, dashboards, cases; transparency via public roadmap/changelogs and MDX+PDF plan bundles.

---

## Cross-References (where to look for detail)

- Pricing, packaging, and SLA: 2_1_onderneming_en_idee.md, 2_5_6_marketingmix_en_doelen.md, marketing/sales_onepager_NL.md, marketing/offer_template_fixed_implementatie_NL.md, legal/SLA.md.
- Legal & privacy posture: legal/* docs, marketing/checklist_DPIA_LLM_onprem_NL.md, legal/dpia_example_case.md.
- Delivery & security: sop_implementation.md, sop_security_incident.md, marketing/onprem_readiness_checklist.md.
- Financials & risk: 3_financieel_plan.md, 3_1b_12mnd_liquiditeit.md, unit_economics.md, risk_register.md, 2_9_risicoanalyse_en_mitigatie.md, 00_cover_sheet_krediet.md.
- GTM assets: 2_10_advertentieplan.md, marketing/landing_page_NL.md, landing_vertical_*_NL.md, outreach_*.md, abm_playbook.md, pitch_deck_outline.md, email_nurture_sequences.md, channel_budget_plan.md.

---

## Notes for Draft 2 (identity-focused filtering later)

- Identity voice and selective targeting will dial up: founder’s narrative (reliability, discipline, transparency), EU/AVG-first stance, and pragmatic delivery ethos; emphasize agency/SI co‑delivery stories and legal/professional DPIA‑readiness.
- Keep technical credibility high (determinism/SSE, observability), but remove duplication and compress cross-references.
- Surface lender-friendly cues (buffers, DSCR, tranche gates, auto‑incasso) only when needed for credit packages.

---

End of SUMMARY_DRAFT_001.
