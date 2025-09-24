# Orchyra Marketing Site — Build Checklist

Scope: marketing-only facade. Static pages that explain the offers per `consumers/.business/ondernemersplan_NL_nl/.002-draft/`.
No backend, no API endpoints, no purchase/credits flows, no provisioning.

Brand: Orchyra · Tagline: Private LLM Hosting in the Netherlands

---

## Phase 0 — Pre‑flight (Current State Validation)

- [ ] **Repository hygiene**
  - [ ] No Vue/Vite template placeholders left in `src/` and `public/` (logo, welcome components, etc.)
  - [ ] `server/index.ts` returns 404 for everything (neutral worker)
- [ ] **Package and worker names**
  - [ ] `package.json` → name: `orchyra-commercial-frontend-nl`
  - [ ] `wrangler.jsonc` → name: `orchyra-commercial-frontend-nl`
- [ ] **Workspace**
  - [ ] `pnpm-workspace.yaml` includes `consumers/.business/commercial-frontend_NL_nl`
- [ ] **Baseline commands**
  - [ ] `npm install`
  - [ ] `npm run type-check` passes
  - [ ] `npm run build` completes
  
- [ ] **HTML baseline**
  - [ ] `index.html` → `<html lang="nl">`, `<title>Orchyra</title>`
  - [ ] `public/favicon.ico` removed or replaced; `<link rel="icon">` matches

---

## Phase 1 — Layout Shell & Navigation

- [ ] **Create layout**
  - [ ] `src/layouts/DefaultLayout.vue` (header, footer, page slot)
  - [ ] `src/components/NavBar.vue` (Home, Public Tap, Private Tap, Pricing, Proof, FAQs, About, Contact/Legal)
  - [ ] `src/components/SiteFooter.vue` (contact links, legal microcopy)
- [ ] **Meta management**
  - [ ] `src/composables/useMeta.ts` (set title, description, keywords; optional JSON-LD injector)
- [ ] **Router**
  - [ ] Add static routes: `/`, `/public-tap`, `/private-tap`, `/pricing`, `/proof`, `/faqs`, `/about`, `/contact`
- [ ] **Acceptance**
  - [ ] Each route renders minimal content without placeholders or console errors

---

## Phase 2 — Home (Front Page) Skeleton

Source: `front-page.md`

- [ ] **Hero**
  - [ ] H1: “AI plumbing, done right.”
  - [ ] H2 (tagline): “Private LLM Hosting in the Netherlands”
  - [ ] Subline from front-page copy
  - [ ] Primary CTA: “Start with the Public Tap” (link to `/public-tap`)
  - [ ] Secondary CTA: “Get a Private Tap quote” (link to `/private-tap`)
  - [ ] Trust badges: “Open-source toolkit • EU data-friendly • Proof-first logs”
  - [ ] Hero image asset plan:
    - [ ] File: `hero_pipes.webp`
    - [ ] Alt: “flat vector blueprint diagram of clean data pipes feeding a dedicated tap, subtle cyan/teal accents, sturdy industrial lines, customer-facing, no text”
- [ ] **Sections**
  - [ ] Why businesses need AI plumbing (with bullets and citations references)
  - [ ] Three things, one toolbox (OSS, Public Tap, Private Tap)
  - [ ] Public Tap overview
  - [ ] Private Tap overview
  - [ ] Proof, not promises (describe artifacts)
  - [ ] Who it’s for (audience bullets)
  - [ ] FAQs preview (link to `/faqs`)
  - [ ] About preview (link to `/about`)
- [ ] **JSON-LD**
  - [ ] Inject Schema.org `ProfessionalService` JSON-LD in `<head>` for Home only (use example in `front-page.md`)
  - [ ] Fields to populate:
    - [ ] `name`: "Orchyra"
    - [ ] `alternateName`: "AI Plumbing by Vince"
    - [ ] `description`: "Open-source LLM orchestration, prepaid Public Tap, and dedicated Private Taps on EU GPUs."
    - [ ] `areaServed`: "NL, EU"
    - [ ] `url`: production URL (TODO)
    - [ ] `sameAs`: [GitHub/org URL] (TODO)
    - [ ] `offers`: 
      - [ ] Public Tap (prepaid credits) — `priceCurrency: EUR`, `price: "50"`, description
      - [ ] Private Tap (A100 80GB) — `priceCurrency: EUR`, `price: "1.80"`, `unitText: per GPU-hour`, `priceValidUntil`
- [ ] **SEO**
  - [ ] Title: “Orchyra — Private LLM Hosting in the Netherlands”
  - [ ] Meta description: “Independent AI plumber. Open-source toolkit, prepaid Public Tap, and dedicated Private Tap on EU GPUs. Proof-first, robust, and transparent.”
  - [ ] Keywords (primary): private LLM hosting; managed GPU inference; agentic API; open-source LLM orchestration; EU AI Act readiness; data residency EU; vLLM serving
  - [ ] Keywords (secondary): AI plumbing; OSS LLM toolkit; dedicated AI API; prepaid AI credits; Hugging Face models; GPU autoscaling; Prometheus metrics

---

## Phase 3 — Public Tap (Marketing Only)

Sources: `services.md`, `ADR-XXX-public-tap-prepaid-credits.md`, `ADR-XXX-public-tap-pricing.md`

- [ ] **Content**
  - [ ] What it is (shared API on curated OSS models)
  - [ ] For whom (developers, agencies, IT)
  - [ ] Why it matters (predictability, quick start)
  - [ ] Prepaid credits (non-refundable, 12-month validity)
  - [ ] Balance visibility note (dashboard/API — descriptive only)
- [ ] **Pricing note**
  - [ ] Baseline: €1.20 per 1M tokens (input+output combined) — “Draft”
  - [ ] Packs — “Draft”
    - [ ] Starter: €50 → ~41M tokens
    - [ ] Builder: €200 → ~166M tokens
    - [ ] Pro: €500 → ~416M tokens
  - [ ] Copy: clearly state 12-month validity and non-refundable terms
- [ ] **SEO**
  - [ ] Title: “Public Tap — Prepaid Credits | Orchyra”
  - [ ] Description: short summary; add keywords per `.002-draft`

---

## Phase 4 — Private Tap (Marketing Only)

Sources: `services.md`, `front-page.md`

- [ ] **Content**
  - [ ] Dedicated API, any OSS model, GPU scale options (1×/2×/4×/8×)
  - [ ] Optional OpenAI-compatible gateway
  - [ ] Value props (privacy, control, logs/metrics)
- [ ] **Pricing snapshot**
  - [ ] A100 80GB — €1.80 / GPU-hour + €250 / month base fee — “Draft”
  - [ ] H100 80GB — €3.80 / GPU-hour + €400 / month base fee — “Draft”
  - [ ] Note: Subject to provider rates; scaling 1×/2×/4×/8× GPUs
- [ ] **SEO**
  - [ ] Title: “Private LLM Hosting on Dedicated GPUs | Orchyra”
  - [ ] Description: enterprise-friendly, transparent, EU-first

---

## Phase 5 — Pricing (Static, Draft)

Sources: `ADR-XXX-public-tap-pricing.md`, `front-page.md`

- [ ] **Credit packs** (Starter/Builder/Pro, token estimates) marked “Draft”
- [ ] **GPU-hour snapshot** (A100/H100) marked “Draft”
- [ ] **SEO** Title: “Pricing — Credits & GPU-hour | Orchyra”
  
Details to include (from ADR + front-page):
- [ ] Baseline: €1.20 per 1M tokens (input+output combined) — “Draft”
- [ ] Packs — “Draft”: Starter €50 ≈ 41M; Builder €200 ≈ 166M; Pro €500 ≈ 416M
- [ ] GPU-hour — “Draft”: A100 80GB €1.80/h + €250/mo; H100 80GB €3.80/h + €400/mo
- [ ] Copy: Subject to provider rates; snapshots illustrative

---

## Phase 6 — Proof (Static Placeholders)

Source: `front-page.md`

- [ ] **Explain proof-first** (logs, SSE transcripts, Prometheus, version pinning)
- [ ] **Placeholders** for screenshots/diagrams (no live data, just descriptive text)
- [ ] **SEO** Title: “Proof — Logs, Metrics, SSE | Orchyra”
  
Artifacts to list (static description):
- [ ] Deployment report with SSE transcripts; throughput & latency metrics
- [ ] Prometheus dashboard snapshots and alert thresholds
- [ ] Version pinning and roll-back plan
- [ ] Documentation bundle aligned with EU AI Act transparency ethos
- [ ] Microcopy: “This is infrastructure—like water or electricity. You shouldn’t have to ‘trust’ it. You should see it.”

---

## Phase 7 — FAQs

Source: `front-page.md`

- [ ] **Common Q&A** (pricing vs OpenAI, OSS model choice, credits, serving efficiency, EU AI Act doc spirit)
- [ ] **SEO** Title: “FAQs | Orchyra”
  
Include answers to:
- [ ] Is this cheaper than OpenAI? → No; value = transparency, EU-friendly, dedicated GPUs
- [ ] Can I bring any OSS model? → Yes (Private Tap), validate VRAM
- [ ] How do prepaid credits work? → Packs, non-refundable, 12-month validity, balance visible
- [ ] What makes serving efficient? → High-throughput engines (e.g., vLLM), metrics exposed
- [ ] Are you EU AI Act ready? → We operate with documentation/transparency spirit; artifacts help governance

---

## Phase 8 — About

Sources: `identity.md`, `USP.md`, `brand.md`

- [ ] **Vince identity** (independent tradesman, OSS, proof-first)
- [ ] **USP pillars** (OSS transparency, proof-first ops, independent identity, prepaid simplicity)
- [ ] **SEO** Title: “About | Orchyra”
  - [ ] CTA: “Talk to Vince” → link to Contact

---

## Phase 9 — Contact & Legal (Static)

Sources: `front-page.md`, `ADR-XXX-public-tap-prepaid-credits.md`

- [ ] **Contact** (email, LinkedIn, GitHub)
- [ ] **Legal microcopy**
  - [ ] Public Tap Terms: prepaid, non-refundable, 12-month validity
  - [ ] Data & logs availability to customers
- [ ] **SEO** Title: “Contact & Legal | Orchyra”
  - [ ] Use footer microcopy: “Public Tap Terms: prepaid, non-refundable, 12-month validity”

---

## Phase 10 — Brand Styling Tokens & Assets

Sources: `brand.md`, `moodboard.md`, `moodboard-extended.md`

- [ ] **CSS tokens** in `src/assets/base.css`
  - [ ] Neutrals (background/text) and accents (cyan `#22d3ee`, teal `#2dd4bf`, purple `#7c5cff`)
  - [ ] Typography: system-ui/Inter/IBM Plex Sans/Source Sans (robust sans)
- [ ] **Favicon**
  - [ ] Replace/remove default `public/favicon.ico`
  - [ ] Ensure `<link rel="icon">` matches chosen asset or is removed
  
- [ ] **Imagery style (from moodboard)**
  - [ ] Pipes & plumbing as flat vector/line-art; blueprint grids
  - [ ] Workshop tools (wrenches, gauges); sturdy, functional design
  - [ ] Optional mascot: serious/minimal; avoid playful/glossy startup look
  - [ ] Avoid photoreal steel/copper textures; prefer clean diagrams

---

## Phase 11 — SEO Pass (All Pages)

- [ ] **Per-route meta** via `useMeta.ts`
  - [ ] Title (unique, brand-suffixed)
  - [ ] Meta description (≤155 chars)
  - [ ] Keywords (primary + secondary from `.002-draft`)
- [ ] **Home JSON-LD** present and valid (copy adapted from `front-page.md`)

---

## Images Checklist (from `front-page.md`)

- [ ] Hero — `hero_pipes.webp` (alt per spec)
- [ ] Problem→Solution panel — split image: messy vs fixed pipelines
- [ ] Public Tap — prepaid credits icon; curated models grid
- [ ] Private Tap — dedicated pipe into a building; GPU options row
- [ ] Proof — screenshots of Grafana/Prometheus and a green-stamped diagram
- [ ] About Vince — portrait or simplified icon/mascot

---

## Phase 2+ — Mesh Expansion (Future)

- [ ] Segmentation pages (Layer 1): Developers, IT Teams, Agencies, Compliance, Small Biz, Funders
- [ ] Service deep dives (Layer 2): OSS, Public Tap, Private Tap, Custom Toolkit Dev
- [ ] Proof & Trust (Layer 3): Proof Library, Case Studies, Competitors, USP
- [ ] Specialized endpoints (Layer 4): Docs portal link, SLA/Contract pack, Partner kits, Governance pack, Toolkit proposal, Financial model

---

## QA — Final Checks

- [ ] `npm run type-check` clean
- [ ] `npm run build` completes
- [ ] Manual link check (nav + in-page anchors)
- [ ] Grep for forbidden placeholders (`Welcome`, `logo.svg`, template components)
- [ ] No console warnings in dev
- [ ] Readability: headings, spacing, consistent copy

---

## (Optional) Preview & Deploy

- [ ] **Local preview**: `npm run preview` (wrangler dev after build)
- [ ] **Domain & env**: set URLs in JSON-LD and meta when production domain exists

---

## Content Checklist — Sources Mapping

- [ ] `front-page.md` → Home sections + JSON-LD example
- [ ] `services.md` → Public Tap & Private Tap pages
- [ ] `ADR-XXX-public-tap-prepaid-credits.md` → Legal notes for credits (Contact/Legal)
- [ ] `ADR-XXX-public-tap-pricing.md` → Pricing page (mark as Draft)
- [ ] `USP.md` → Home USP highlights + About page
- [ ] `competitors.md` → Background for Proof/FAQs (optional deep page later)
- [ ] `target-audiences.md` → “Who it’s for” section + tone guidance
- [ ] `page-layers.md`, `mesh-site-architecture.md` → IA cross-links (conceptual)
- [ ] `brand.md`, `moodboard.md`, `moodboard-extended.md` → tone, colors, typography, imagery

---

## Out of Scope (Explicit)

- [ ] No purchase/checkout/credits top-ups
- [ ] No dashboards, balance APIs, or GPU provisioning flows
- [ ] No live metrics or SSE streams (placeholders only)
- [ ] No user auth or forms (use `mailto:` links if needed)

---

## Notes

- Keep copy consistent with Orchyra and the tagline across headings and metadata.
- Use clear “Draft” labels wherever pricing is not finalized.
- Prefer simple, sturdy components; avoid design flourishes that imply functionality.
