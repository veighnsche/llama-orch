# Orchyra Marketing Site — Build Checklist

Scope: marketing-only facade. Static pages that explain the offers per `consumers/.business/ondernemersplan_NL_nl/.002-draft/`.
No backend, no API endpoints, no purchase/credits flows, no provisioning.

Brand: Orchyra · Tagline: Private LLM Hosting in the Netherlands

---

## Progress Snapshot

- [x] Phase 1 — Layout Shell & Navigation (layout, nav/footer, meta composable, routes)
- [x] Phase 2 — Home (Front Page) Skeleton (hero, sections, JSON-LD, SEO)
- [x] Pages scaffolded (static content): Public Tap, Private Tap, Pricing, Proof, FAQs, About, Contact/Legal
- [x] SEO per route using `useMeta` (locale-reactive; Home has JSON-LD)
- [x] i18n infrastructure (EN/NL) + Language Switcher; applied to NavBar, Footer, Home, Public Tap
- [x] i18n applied to remaining pages (Pricing, Proof, FAQs, About, Contact/Legal, Private Tap)
- [x] JSON-LD localized per language (reactive to locale switch on Home)
- [x] A11y labels localized (nav, footer, language switcher)
- [x] Localized SEO keywords per route (EN/NL)
- [x] Localized SEO descriptions per route (≤155 chars)
- [x] Canonical link and hreflang alternates per route
- [x] Favicon replacement/removal
- [ ] Images (hero and others) not added yet (placeholders only)
- [ ] QA (type-check/build pass, manual link pass)

## Phase 0 — Pre‑flight (Current State Validation)

- [x] **Repository hygiene**
    - [x] No Vue/Vite template placeholders left in `src/` and `public/` (logo, welcome components, etc.)
    - [x] `server/index.ts` returns 404 for everything (neutral worker)
- [x] **Package and worker names**
    - [x] `package.json` → name: `orchyra-commercial-frontend-nl`
    - [x] `wrangler.jsonc` → name: `orchyra-commercial-frontend-nl`
- [x] **Workspace**
    - [x] `pnpm-workspace.yaml` includes `consumers/.business/commercial-frontend_NL_nl`
- [x] **Baseline commands**
    - [x] `pnpm install`
    - [x] `pnpm run type-check` passes
    - [x] `pnpm run build` completes
- [x] **HTML baseline**
    - [x] `index.html` → `<html lang="nl">`, `<title>Orchyra</title>`
    - [x] `public/favicon.ico` removed or replaced; `<link rel="icon">` matches

---

## Phase 1 — Layout Shell & Navigation

- [x] **Create layout**
    - [x] `src/layouts/DefaultLayout.vue` (header, footer, page slot)
    - [x] `src/components/NavBar.vue` (Home, Public Tap, Private Tap, Pricing, Proof, FAQs, About, Contact/Legal)
    - [x] `src/components/SiteFooter.vue` (contact links, legal microcopy)
- [x] **Meta management**
    - [x] `src/composables/useMeta.ts` (set title, description, keywords; optional JSON-LD injector)
    - [x] Acceptance (i18n): Meta `title`/`description` update on locale change via `watchSources`
- [x] **Router**
    - [x] Add static routes: `/`, `/public-tap`, `/private-tap`, `/pricing`, `/proof`, `/faqs`, `/about`, `/contact`
- [x] **Acceptance**
    - [x] Each route renders minimal content without placeholders or console errors
    - [x] All visible nav/footer text localized via `$t(...)`
    - [x] Nav and footer `aria-label`s localized

---

## Phase 2 — Home (Front Page) Skeleton

Source: `front-page.md`

- [ ] **Hero**
    - [x] H1: “AI plumbing, done right.”
    - [x] H2 (tagline): “Private LLM Hosting in the Netherlands”
    - [x] Subline from front-page copy
    - [x] Primary CTA: “Start with the Public Tap” (link to `/public-tap`)
    - [x] Secondary CTA: “Get a Private Tap quote” (link to `/private-tap`)
    - [x] Trust badges: “Open-source toolkit • EU data-friendly • Proof-first logs”
    - [x] Hero image asset plan:
        - [x] File: `hero_pipes.png`
        - [x] Alt: “flat vector blueprint diagram of clean data pipes feeding a dedicated tap, subtle cyan/teal accents, sturdy industrial lines, customer-facing, no text”
- [x] **Sections**
    - [x] Why businesses need AI plumbing (with bullets and citations references)
    - [x] Three things, one toolbox (OSS, Public Tap, Private Tap)
    - [x] Public Tap overview
    - [x] Private Tap overview
    - [x] Proof, not promises (describe artifacts)
    - [x] Who it’s for (audience bullets)
    - [x] FAQs preview (link to `/faqs`)
    - [x] About preview (link to `/about`)
- [x] **JSON-LD**
    - [x] Inject Schema.org `ProfessionalService` JSON-LD in `<head>` for Home only (use example in `front-page.md`)
    - [x] Fields to populate:
        - [x] `name`: "Orchyra"
        - [x] `alternateName`: "Private LLM Hosting in the Netherlands" (localized)
        - [x] `description`: "Open-source LLM orchestration, prepaid Public Tap, and dedicated Private Taps on EU GPUs."
        - [x] `areaServed`: "NL, EU"
        - [x] `url`: production URL (env-configurable)
        - [x] `sameAs`: [GitHub/org URL] (env-configurable)
        - [x] `offers`:
            - [x] Public Tap (prepaid credits) — `priceCurrency: EUR`, `price: "50"`, description
            - [x] Private Tap (A100 80GB) — `priceCurrency: EUR`, `price: "1.80"`, `unitText` localized (e.g., "$t('a11y.perGpuHour')"), `priceValidUntil`
- [x] **SEO**
    - [x] Title: “Orchyra — Private LLM Hosting in the Netherlands”
    - [x] Meta description: “Independent AI plumber. Open-source toolkit, prepaid Public Tap, and dedicated Private Tap on EU GPUs. Proof-first, robust, and transparent.”
    - [x] Keywords (primary): private LLM hosting; managed GPU inference; agentic API; open-source LLM orchestration; EU AI Act readiness; data residency EU; vLLM serving
    - [x] Keywords (secondary): AI plumbing; OSS LLM toolkit; dedicated AI API; prepaid AI credits; Hugging Face models; GPU autoscaling; Prometheus metrics

---

## Phase 3 — Public Tap (Marketing Only)

Sources: `services.md`, `ADR-XXX-public-tap-prepaid-credits.md`, `ADR-XXX-public-tap-pricing.md`

- [x] **Content**
    - [x] What it is (shared API on curated OSS models)
    - [x] For whom (developers, agencies, IT)
    - [x] Why it matters (predictability, quick start)
    - [x] Prepaid credits (non-refundable, 12-month validity)
    - [x] Balance visibility note (dashboard/API — descriptive only)
- [x] **Pricing note**
    - [x] Baseline: €1.20 per 1M tokens (input+output combined) — “Draft”
    - [x] Packs — “Draft”
        - [x] Starter: €50 → ~41M tokens
        - [x] Builder: €200 → ~166M tokens
        - [x] Pro: €500 → ~416M tokens
    - [x] Copy: clearly state 12-month validity and non-refundable terms
- [x] **SEO**
    - [x] Title: “Public Tap — Prepaid Credits | Orchyra”
    - [x] Description: short summary; add keywords per `.002-draft`

---

## Phase 4 — Private Tap (Marketing Only)

Sources: `services.md`, `front-page.md`

- [x] **Content**
    - [x] Dedicated API, any OSS model, GPU scale options (1×/2×/4×/8×)
    - [x] Optional OpenAI-compatible gateway
    - [x] Value props (privacy, control, logs/metrics)
- [x] **Pricing snapshot**
    - [x] A100 80GB — €1.80 / GPU-hour + €250 / month base fee — “Draft”
    - [x] H100 80GB — €3.80 / GPU-hour + €400 / month base fee — “Draft”
    - [x] Note: Subject to provider rates; scaling 1×/2×/4×/8× GPUs
- [x] **SEO**
    - [x] Title: “Private LLM Hosting on Dedicated GPUs | Orchyra”
    - [x] Description: enterprise-friendly, transparent, EU-first

---

## Phase 5 — Pricing (Static, Draft)

Sources: `ADR-XXX-public-tap-pricing.md`, `front-page.md`

- [x] **Credit packs** (Starter/Builder/Pro, token estimates) marked “Draft”
- [x] **GPU-hour snapshot** (A100/H100) marked “Draft”
- [x] **SEO** Title: “Pricing — Credits & GPU-hour | Orchyra”

Details to include (from ADR + front-page):

- [x] Baseline: €1.20 per 1M tokens (input+output combined) — “Draft”
- [x] Packs — “Draft”: Starter €50 ≈ 41M; Builder €200 ≈ 166M; Pro €500 ≈ 416M
- [x] GPU-hour — “Draft”: A100 80GB €1.80/h + €250/mo; H100 80GB €3.80/h + €400/mo
- [x] Copy: Subject to provider rates; snapshots illustrative

---

## Phase 6 — Proof (Static Placeholders)

Source: `front-page.md`

- [x] **Explain proof-first** (logs, SSE transcripts, Prometheus, version pinning)
- [x] **Placeholders** for screenshots/diagrams (no live data, just descriptive text)
- [x] **SEO** Title: “Proof — Logs, Metrics, SSE | Orchyra”

Artifacts to list (static description):

- [x] Deployment report with SSE transcripts; throughput & latency metrics
- [x] Prometheus dashboard snapshots and alert thresholds
- [x] Version pinning and roll-back plan
- [x] Documentation bundle aligned with EU AI Act transparency ethos
- [x] Microcopy: “This is infrastructure—like water or electricity. You shouldn’t have to ‘trust’ it. You should see it.”

---

## Phase 7 — FAQs

Source: `front-page.md`

- [x] **Common Q&A** (pricing vs OpenAI, OSS model choice, credits, serving efficiency, EU AI Act doc spirit)
- [x] **SEO** Title: “FAQs | Orchyra”

Include answers to:

- [x] Is this cheaper than OpenAI? → No; value = transparency, EU-friendly, dedicated GPUs
- [x] Can I bring any OSS model? → Yes (Private Tap), validate VRAM
- [x] How do prepaid credits work? → Packs, non-refundable, 12-month validity, balance visible
- [x] What makes serving efficient? → High-throughput engines (e.g., vLLM), metrics exposed
- [x] Are you EU AI Act ready? → We operate with documentation/transparency spirit; artifacts help governance

---

## Phase 8 — About

Sources: `identity.md`, `USP.md`, `brand.md`

- [x] **Vince identity** (independent tradesman, OSS, proof-first)
- [x] **USP pillars** (OSS transparency, proof-first ops, independent identity, prepaid simplicity)
- [x] **SEO** Title: “About | Orchyra”
    - [x] CTA: “Talk to Vince” → link to Contact

---

## Phase 9 — Contact & Legal (Static)

Sources: `front-page.md`, `ADR-XXX-public-tap-prepaid-credits.md`

- [x] **Contact** (email, LinkedIn, GitHub)
- [x] **Legal microcopy**
    - [x] Public Tap Terms: prepaid, non-refundable, 12-month validity
    - [x] Data & logs availability to customers
- [x] **SEO** Title: “Contact & Legal | Orchyra”
    - [x] Use footer microcopy: “Public Tap Terms: prepaid, non-refundable, 12-month validity”

---

## Phase 10 — Brand Styling Tokens & Assets

Sources: `brand.md`, `moodboard.md`, `moodboard-extended.md`

- [x] **CSS tokens** in `src/assets/base.css`
    - [x] Neutrals (background/text) and accents (cyan `#22d3ee`, teal `#2dd4bf`, purple `#7c5cff`)
    - [x] Typography: system-ui/Inter/IBM Plex Sans/Source Sans (robust sans)
- [x] **Favicon**
    - [x] Replace/remove default `public/favicon.ico`
    - [x] Ensure `<link rel="icon">` matches chosen asset or is removed
- [ ] **Imagery style (from moodboard)**
    - [ ] Pipes & plumbing as flat vector/line-art; blueprint grids
    - [ ] Workshop tools (wrenches, gauges); sturdy, functional design
    - [ ] Optional mascot: serious/minimal; avoid playful/glossy startup look
    - [ ] Avoid photoreal steel/copper textures; prefer clean diagrams

---

## Phase 11 — SEO Pass (All Pages)

- [x] **Per-route meta** via `useMeta.ts`
    - [x] Title (unique, brand-suffixed)
    - [x] Meta description (≤155 chars) — localized per route (EN/NL)
    - [x] Keywords (primary + secondary from `.002-draft`) — localized per route (EN/NL)
    - [x] Locale reactivity: meta updates when language changes (use `watchSources` on `locale`)
    - [x] Canonical link tag present and correct per route
    - [x] hreflang alternates present for `en`, `nl`, and `x-default`
- [x] **Home JSON-LD** present and valid (copy adapted from `front-page.md`)
    - [x] Locale reactivity: JSON-LD updates when language changes

---

## Images Checklist (from `front-page.md`)

- [x] Hero — `hero_pipes.png` (alt per spec)
- [ ] Problem→Solution panel — split image: messy vs fixed pipelines
- [ ] Public Tap — prepaid credits icon; curated models grid
- [ ] Private Tap — dedicated pipe into a building; GPU options row
- [ ] Proof — screenshots of Grafana/Prometheus and a green-stamped diagram

---

## Phase 4 — Design Pass (Visual & UX)

Sources: `ondernemersplan_NL_nl/.002-draft/brand.md`, `moodboard.md`, `moodboard-extended.md`, `identity.md`, `front-page.md`, `page-layers.md`.

- [ ] **Color system (CSS variables)**
    - [ ] Base neutrals: `--bg-offwhite: #f5f5f5`, `--bg-gray: #e0e0e0`, `--ink-slate: #2c2f35`
    - [ ] Accents: `--acc-cyan: #22d3ee`, `--acc-teal: #2dd4bf`, `--acc-purple: #7c5cff`
    - [ ] Utility: `--ok-green: #22c55e`, `--err-red: #ef4444`
    - [ ] Acceptance: WCAG AA contrast on text and interactive states

- [ ] **Typography**
    - [ ] Headings: industrial sans (Inter / IBM Plex Sans / Source Sans), bold weights for H1/H2
    - [ ] Body: same family, regular/medium; `font-display: swap`; WOFF2
    - [ ] Acceptance: consistent rhythm and sizes across pages; no startup-stylized display fonts

- [ ] **Layout & Components**
    - [ ] Grid-based sections; sturdy, boxed cards (no glassmorphism, no large radii)
    - [ ] Blueprint-style background grid (subtle, low-contrast) on hero or section dividers
    - [ ] Cards for “Three things” on Home; pricing framed as a service menu (not SaaS tiers)
    - [ ] Trust badges row (Open-source toolkit • EU data-friendly • Proof-first logs)
    - [ ] Acceptance: sections read like technical documentation, not marketing gloss

- [ ] **Imagery Style**
    - [ ] Vector/line-art pipes, valves, gauges; workshop tools icons
    - [ ] No photoreal steel/copper textures; no abstract AI brains
    - [ ] Hero: `hero_pipes.png` per `front-page.md` (blueprint diagram; cyan/teal accents)
    - [ ] Acceptance: at least 1 blueprint diagram + 1 “proof” visual (e.g., dashboard snapshot)

- [ ] **Tone of Voice**
    - [ ] Direct, practical, proof-first; avoid hype language
    - [ ] Vince front-and-center as independent tradesman (About/Contact)
    - [ ] Acceptance: copy prefers “show, don’t market” with logs/metrics where relevant

- [ ] **Information Architecture (mesh alignment)**
    - [ ] Home links clearly to OSS → Public Tap → Private Tap
    - [ ] Cross-links: Proof ↔ Pricing ↔ FAQs; About ↔ Contact/Legal
    - [ ] Acceptance: self-routing CTAs present and consistent with `page-layers.md`

- [ ] **Accessibility & UX**
    - [ ] Localized aria-labels (done), visible focus rings, 44×44 touch targets
    - [ ] Keyboard navigable menus/links; skip-to-content link optional
    - [ ] Acceptance: axe/lighthouse a11y checks show no critical issues

- [ ] **Performance & Assets**
    - [ ] Compress images (PNG), responsive sizes; lazy-load below the fold
    - [ ] Preload primary font; avoid CLS; favicon replaced with brand-consistent icon
    - [ ] Acceptance: Lighthouse Performance ≥ 90 on Home (desktop) with images in place

- [ ] **Optional Mascot**
    - [ ] Minimal, serious vector (toolbox/pipe/gauge character) if used; not playful
    - [ ] Acceptance: only if it reinforces professionalism; otherwise omit
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
