# Website Plan — Cloudflare Pages + Hono JSX (Draft, Index)

Dit plan is gemoduleerd. Gebruik onderstaande overzichtsbestanden:

- overview.md — doelen en doelgroep
- architecture.md — platform/architectuur (Cloudflare Pages + Hono JSX)
- content_sources.md — herbruikbare content uit de repo
- routes_sitemap.md — routes, sitemap en tekstuele wireframes
- components.md — componenten en UI‑bouwstenen
- dynamics_ssg.md — SSG‑vriendelijke dynamiek (status, downloads)
- data_metrics.md — metriekmapping en drempels
- seo_a11y.md — SEO, performance en toegankelijkheid
- forms_integrations.md — formulieren, e‑mail, Turnstile
- project_structure.md — mappen, scripts, prerender
- content_mapping.md — mapping repo → site
- build_deploy_cloudflare.md — build/deploy en Pages setup
- roadmap.md — MVP → + planning
- risks_mitigation.md — risico’s en mitigaties
- kpis.md — succes KPI’s
- deliverables_qredits.md — wat staat live bij indienen
- components_contracts.md — datacontracten + API shapes
- qa_checklist.md — go‑live checklist
- sprint_plan.md — 5‑daagse sprint
- docs_mdx_pipeline.md — MDX contentbron + parsing
- pdf_generation.md — PDF uit .mdx (build/CI)
- ci_pipeline.md — GitHub Actions voor build + PDF + deploy
 - navigation_ergonomics.md — navigatie & interlinking

De volledige tekststaat hieronder nog tijdelijk voor referentie.

## 1. Doelgroep & Doelen
- Doelgroep: beslissers (legal/professional services), agencies/SI’s, tech leads.
- Doelen: discovery calls, POC-aanvragen, partner calls, checklist downloads; bewijs (KPI’s/USP’s) zichtbaar maken.

## 2. Platform & Architectuur
- Hosting: Cloudflare Pages (SSG) + Pages Functions (voor lichte API’s)
- Framework: Hono + JSX templates (SSR/SSG tijdens build)
- Statische assets: bundling via esbuild/Vite (CSS minimal, geen framework CSS nodig)
- Analyses: Cloudflare Web Analytics (cookieless) + optioneel Plausible (EU)
- Security: HTTPS, security headers (CSP licht), Cloudflare Turnstile op formulieren

## 3. Contentbronnen (repo)
- Hergebruik uit .001-draft/marketing/* (LP’s, POC, offer template), en kern uit ondernemersplan (USP’s, SLA tiers):
  - Hero/Propositie: marketing/landing_page_NL.md
  - Vertical LP’s: marketing/landing_vertical_legal_NL.md, marketing/landing_vertical_agency_SI_NL.md
  - POC: marketing/poc_pakket_NL.md
  - USP’s/roadmap/KPI’s: 2_7_usps_roadmap_en_kpi.md
  - SLA tiers: 2_5_6_marketingmix_en_doelen.md (SLA sectie)
  - DPIA checklist: marketing/checklist_DPIA_LLM_onprem_NL.md
  - Offer template link in bijlagen voor sales follow-up

## 4. Navigatie & Routes (SSG)
- / (Home): Hero, waardepropositie, stappen (2–4 weken), SLA-kaarten, KPI/bewijs, CTA’s (call/POC/checklist)
- /legal (Vertical): variant voor legal/professional services
- /agency (Vertical): variant voor agencies/SI’s (partnerpropositie)
- /poc (POC pakket): scope, KPI’s, deliverables, CTA aanvragen
- /pricing (Prijzen): implementatie bandbreedte + SLA-bundels + FAQ
- /checklists (Downloads): DPIA & On‑Prem Readiness; e‑mailgate optioneel
- /about (Over): missie/waardes/roadmap/vertrouwen
- /sla (SLA samenvatting): pakketten, R/T, rapportage, incidentproces
- /contact (Contact): formulier (Turnstile) of mailto; alternatief: book a call
- /status (Status/metrics): eenvoudige widget/kaart met uptime/latency (light fetch)
- /legal/{privacy,cookies,av}: juridische pagina’s

## 5. Componenten (JSX, minimal JS)
- Hero, ValueProps (USP’s), Steps (Intake→Go‑live), SLA TierCards, KPI Cards
- ProofStrip (logos/benchmarks links), Testimonial (optioneel), CTA Banner
- Forms: ContactForm (Turnstile), DownloadGate (email gate → mail service/CF worker)
- StatusWidget (client-only fetch) — non-blocking; SSR fallback met laatste bekende getallen

## 6. Dynamiek (SSG-compatibel)
- Live stats: client fetch naar /api/status (Pages Function) met simpele JSON (uptime_pct, p50_latency_ms, sla_clients_count)
  - Bronopties: Cloudflare KV/R2 of een handmatig geüpdatete JSON (als interim)
  - Fallback SSR: render ‘laatst bijgewerkt’ waarden vanaf build JSON
- Download counts (optioneel): client fetch + KV increment; toont populariteit van checklist
- Status badges: simpele kleur (groen/oranje/rood) o.b.v. thresholds (metrics_mapping.md)

## 7. Data & Metrics (koppeling)
- Namen en betekenissen volgens proof/metrics_mapping.md  
- Mapping → UI: latency_ms (p50) → “Median latency”, uptime_pct → “Uptime (30d)”
- Drempels: latency p50 ≤ 300 ms (groen); uptime ≥ 99.5% (Premium)

## 8. SEO/Performance/Toegankelijkheid
- SEO: titels/meta, OpenGraph, canonicals, sitemap.xml, robots.txt
- Schema.org: Product/Service + Organization + FAQ schema op pricing/faq
- Performance: < 50kB CSS/JS, images webp/avif, lazy loading; no JS above fold
- A11y: alt-teksten, kleurcontrast, toetsenbordnavigatie, landmark roles

## 9. Forms & Integraties
- Contact: POST naar Pages Function → e‑mail (MailChannels/SendGrid) of KV-queue
- Download gates: e‑mail + toestemming → stuur PDF/MD link, log in CRM later
- Book a call: link naar extern (Cal.com/Calendly) of simpel contactformulier

## 10. Technische structuur (voorstel)
```
website/
  package.json
  src/
    app.tsx            # Hono setup + routes
    pages/             # JSX pages (Home, Legal, Agency, POC, Pricing, ...)
    components/        # Hero, ValueProps, Steps, TierCards, KPI, Forms, StatusWidget
    styles/            # CSS (vanilla/utility)
    data/              # Static JSON (fallback metrics, SLA tiers)
    api/status.ts      # Pages Function (returns JSON metrics)
  public/
    images/
```

## 11. Content mapping (uit repo → site)
- Hero/value/KPI → uit marketing/landing_page_NL.md
- Verticale koppen/USP’s → landing_vertical_*_NL.md
- SLA kaarten → prijzen uit 2_5_6_marketingmix_en_doelen.md
- POC copy → poc_pakket_NL.md
- Checklists (DPIA/On‑Prem) → direct linken of mailen na gate

## 12. Build & Deploy
- Build: `pnpm build` → prerender alle routes (SSG) met Hono
- Deploy: Cloudflare Pages (CI), Functions voor `/api/status` en formulieren
- Secrets: Turnstile keys + mail API keys via Pages Environment Variables

## 13. Roadmap (MVP → +)
- MVP (Week 1): Home, Legal, Agency, POC, Pricing, SLA, Contact, Checklists, Status (static JSON), Forms (email)
- +Week 2–3: Gate + KV analytics, Blog/News (MD→HTML), Case studies
- +Week 4: Multilingual (EN), sitemap automation, simple CMS (MD in repo)

## 14. Risico’s & Mitigatie
- Te veel dynamiek → plan SSG + mini‑islands; fallback SSR data  
- Cookie/consent-complexiteit → Cloudflare Analytics + cookieless default  
- Content drift vs repo → bouwscript dat MD→HTML rendert en single‑source bewaakt

## 15. Succes KPI’s
- CTR LP→Contact ≥ 2.5%, POC aanvragen ≥ 3/maand, TTFB < 100 ms (edge), LCP < 2.5s, CLS < 0.1

---

## 16. Leverbaar voor Qredits (klaar bij indienen)

Minimale, publiek werkende site met bewijs en lead‑capture:
- Home (/): propositie, stappen (2–4 weken), SLA‑kaarten, KPI’s, CTA’s (call/POC/checklist), statuswidget (fallback statisch)
- Vertical (/legal, /agency): uitgewerkte varianten met CTA’s (POC/partner)
- POC (/poc): tijdbox, deliverables, KPI’s, prijsbandbreedte, aanvraagformulier
- Pricing (/pricing): implementatiebandbreedte + SLA’s + FAQ
- Checklists (/checklists): DPIA + On‑Prem download (mail gate optioneel, anders direct)
- SLA (/sla), About (/about), Contact (/contact), Legal (/legal/privacy|cookies|av)
- Status (/status): eenvoudige publieke badge/kaart (uptime/latency, met “laatst bijgewerkt”)

Bij oplevering inbegrepen:
- Cloudflare Pages project + custom domain + HTTPS
- Cloudflare Web Analytics geactiveerd (cookieless)
- Turnstile op ContactForm (env keys ingevoerd)
- Mail sending via Pages Functions (MailChannels of SendGrid)

---

## 17. Sitemap & Wireframes (tekstueel)

- Home
  - Hero (headline + sub + CTA: call/POC)
  - Value props (USP’s 3–5 bullets) + Proof strip (benchmarks/cases links)
  - Steps (Intake → Go‑live in 2–4 weken)
  - SLA tier cards (Essential/Advanced/Enterprise)
  - KPI cards (Latency ≤ 300ms, Uptime ≥ 99.5%, Determinisme, Kosten/1K)
  - CTA banner (POC of Checklist)
  - Status widget (uptime/latency badge + “laatst bijgewerkt”)
- Legal variant
  - Sector‑pijnpunten (privacy/DPIA/audit)
  - Use‑cases (contract assistent, QA, rapportage)
  - KPI/acceptatie (privacy/logging/latency)
  - CTA (POC)
- Agency variant
  - Partnerpropositie (co‑delivery/white‑label)
  - Templates/runbooks → sneller live
  - SLA‑backline (we doen de backline)
  - CTA (partner call)
- POC
  - Samenvatting, deliverables, KPI’s, scope IN/UIT, planning, benodigd van klant
  - Formulier (POC aanvraag) met Turnstile
- Pricing
  - Implementatie: €4.5–€9.5k (2–4 weken)
  - SLA’s: €350/€600/€900 p/m (kaarten)
  - FAQ (modellen, cloud/on‑prem, updates/security)
- Checklists
  - DPIA, On‑Prem Readiness (download of gated)
- SLA / About / Contact / Legal / Status

---

## 18. Componenten & Datacontracten

- TierCard: `{ name: 'Essential'|'Advanced'|'Enterprise', price: number, features: string[], rt: string }`
- KPIBadge: `{ label: string, value: string, status: 'ok'|'warn'|'err' }`
- StatusWidget: `{ uptime_pct: number, p50_latency_ms: number, updated_at: string, sla_clients_count?: number }`
- ContactForm: `{ name, email, company, message }` → POST `/api/contact`
- DownloadGate (optioneel): `{ email, consent }` → POST `/api/download?doc=dpiA|onprem`

Status API (JSON):
```json
{
  "uptime_pct": 99.7,
  "p50_latency_ms": 280,
  "sla_clients_count": 6,
  "updated_at": "2025-01-05T10:00:00Z"
}
```

---

## 19. Copy mapping (repo → site)

- Hero/value/KPI → marketing/landing_page_NL.md
- Legal/Agency secties → marketing/landing_vertical_*_NL.md
- POC → marketing/poc_pakket_NL.md
- SLA kaarten/prijzen → 2_5_6_marketingmix_en_doelen.md
- Checklists → marketing/checklist_DPIA_LLM_onprem_NL.md + marketing/onprem_readiness_checklist.md
- Proof/KPI definities → 2_7_usps_roadmap_en_kpi.md + proof/metrics_mapping.md

---

## 20. Projectstructuur & scripts (concreet)

package.json (indicatief):
```json
{
  "name": "website",
  "private": true,
  "scripts": {
    "dev": "hono --serve src/app.tsx",
    "build": "node scripts/prerender.mjs",
    "preview": "npx serve dist"
  },
  "devDependencies": {
    "hono": "^4",
    "esbuild": "^0.21",
    "remark": "^15",
    "remark-html": "^16"
  }
}
```

Bestanden:
```
src/app.tsx           # Hono app; export handlers voor SSG prerender + CF Pages Functions
src/pages/*.tsx       # JSX pages (Home, Legal, Agency, POC, Pricing, ...)
src/components/*.tsx  # TierCard, KPIBadge, StatusWidget, Forms
src/data/*.json       # sla_tiers.json, status_fallback.json
functions/api/*.ts    # /api/status, /api/contact (Pages Functions)
scripts/prerender.mjs  # Render alle routes naar dist/ (SSG)
```

scripts/prerender.mjs (concept):
```js
import { renderToString } from 'hono/jsx/dom/server';
import routes from '../src/routes.js'; // eigen registry die elke route → component exporteert
import fs from 'node:fs/promises';

const out = 'dist';
await fs.mkdir(out, { recursive: true });
for (const { path, Component, props } of routes) {
  const html = '<!doctype html>' + renderToString(<Component {...props} />);
  const target = path === '/' ? `${out}/index.html` : `${out}${path}/index.html`;
  await fs.mkdir(target.replace(/\/index.html$/, ''), { recursive: true });
  await fs.writeFile(target, html);
}
```

Status fallback: `src/data/status_fallback.json` → gebruikt door StatusWidget SSR.

---

## 21. Cloudflare Pages setup (stappen)

- Repo koppelen → Project (Pages)
- Build command: `pnpm install && pnpm build`
- Build output: `dist`
- Pages Functions: map `functions/` (Node compat on), routes `/api/status`, `/api/contact`
- Env vars (Production/Preview): `TURNSTILE_SITE_KEY`, `TURNSTILE_SECRET_KEY`, `MAIL_PROVIDER=mailchannels`, `MAIL_FROM`, `MAIL_TO`
- Analytics: enable Cloudflare Web Analytics (cookieless)

Contact function (concept):
```ts
export const onRequestPost: PagesFunction = async (ctx) => {
  const body = await ctx.request.json();
  // verify Turnstile with TURNSTILE_SECRET_KEY
  // send email via MailChannels (or SendGrid)
  return new Response(JSON.stringify({ ok: true }), { headers: { 'content-type': 'application/json' } });
};
```

Status function (concept):
```ts
export const onRequestGet: PagesFunction = async () => {
  // optional: read KV/R2; else return static numbers
  return Response.json({ uptime_pct: 99.7, p50_latency_ms: 280, sla_clients_count: 6, updated_at: new Date().toISOString() });
};
```

---

## 22. QA Checklist (voor live‑gang)

- Performance: Lighthouse ≥ 95 (mobile/desktop); bundle size < 50kB JS
- A11y: headings/labels/contrast/keyboard nav OK
- SEO: titles/meta/OG/sitemap/robots present; canonical per route
- Forms: Turnstile werkt; e‑mail bezorging getest
- Legal: privacy/cookies/AV pagina’s aanwezig; links in footer
- Content: alle koppen en CTA’s consistent met ondernemersplan (USP’s/SLA)

---

## 23. Planning (korte sprint)

Dag 1: scaffold project, Home/Legal/Agency routes, styles, Tier/KPI components  
Dag 2: POC/Pricing/Checklists/SLA/About/Contact, footers/nav, content import uit repo  
Dag 3: Forms (Turnstile + mail), Status function + widget, SEO/meta/schema  
Dag 4: QA (Lighthouse/A11y), copy‑polish, deploy naar Pages (preview + prod)  
Dag 5: Screenshots/links opnemen in proof bundle; finetune CTA’s  
