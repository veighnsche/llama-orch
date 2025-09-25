# Orchyra — Commercial Frontend (NL)

Status: scaffold stub with all Vue/Vite placeholders removed. Ready for branded content and routes.

- Brand: Orchyra
- Tagline: Private LLM Hosting in the Netherlands
- Locale: nl (Dutch)
- Tech stack: Vite + Vue 3 + TypeScript + Vue Router + Cloudflare Workers (Wrangler)

## What’s in here

- Minimal Vue app stub at this folder root
    - Placeholders removed
    - `index.html` set to lang="nl" and title "Orchyra"
    - `src/App.vue` only renders `<RouterView />`
    - `src/router/index.ts` has only the home route
    - `server/index.ts` returns 404 for all routes (no demo JSON)
    - Neutral base styles in `src/assets/`

## Develop

From this folder:

```bash
# 1) install deps
npm install

# 2) run dev server
npm run dev

# 3) type-check and build
npm run type-check
npm run build

# 4) (optional) preview via Cloudflare
npm run preview    # wrangler dev after a build
```

Using pnpm (workspace note)

- This frontend lives under the monorepo; its `package.json` is at this folder root.
- If you prefer pnpm, run commands from this folder (or use `-C` to change cwd):

```bash
pnpm install
pnpm dev
```

If you want pnpm to discover this package via filters, include this path in `pnpm-workspace.yaml` (e.g. `consumers/.business/commercial-frontend_NL_nl`).

## Intended site structure (v1)

This frontend will implement the marketing website derived from the entrepreneurs plan (.002-draft). The initial route plan:

- Home (front page)
- Public Tap (prepaid credits)
- Private Tap (dedicated GPUs)
- Pricing
- Proof (logs, metrics, SSE transcripts)
- FAQs
- About (Vince)
- Contact / Legal

SEO/Schema

- Use Orchyra as the brand and the tagline “Private LLM Hosting in the Netherlands”.
- Add JSON-LD (ProfessionalService) to the `<head>` when the layout shell is built (see `.002-draft/front-page.md` for an example block).

## Content sources (business plan)

Key docs driving copy and structure are under:

- `consumers/.business/ondernemersplan_NL_nl/.002-draft/`
    - `front-page.md` — hero/sections/CTAs + JSON-LD example
    - `naming.md` — brand picked (Orchyra) and tagline
    - `USP.md` — four USP pillars
    - `services.md` — OSS / Public Tap / Private Tap (+ extra toolkit dev)
    - `ADR-XXX-public-tap-pricing.md` — draft pricing model/credit packs
    - `ADR-XXX-public-tap-prepaid-credits.md` — non-refundable credits policy
    - `competitors.md` — competitor map and lessons
    - `target-audiences.md`, `page-layers.md`, `mesh-site-architecture.md` — IA/routing plan
    - `brand.md`, `moodboard.md`, `moodboard-extended.md` — tone, visuals, palette

## Next steps

- Implement a layout shell with header/footer and JSON-LD injection
- Build Home route using `front-page.md` sections (H1/H2/CTAs, trust badges)
- Create Public Tap and Private Tap pages with copy from `services.md` and ADRs
- Add Pricing page (credit packs + GPU-hour snapshot) and FAQs
- Add Proof page (placeholder slots for logs/metrics/screenshots)
- Wire basic navigation and set meta tags (title/description/keywords)
- Replace favicon with a branded icon (optional)

## Notes

- No sample UI remains; the app is intentionally minimal and safe to commit.
- The Cloudflare worker is neutralized (returns 404). Add real endpoints only when needed.
- Keep branding consistent with Orchyra and the tagline across headings and metadata.
