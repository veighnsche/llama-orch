# Home (Front Page)

- Purpose: Introduce Orchyra, establish positioning, and route visitors deeper (Public Tap, Private Tap, Proof, Pricing, FAQs, About, Contact).
- Primary sources: `.002-draft/front-page.md`, `src/views/HomeView.vue`, `src/i18n/en.ts`, `src/i18n/nl.ts`.
- Components: `src/views/HomeView.vue`, layout shell `src/layouts/DefaultLayout.vue`.

## Sections

### SEO / Head

- Purpose: Localized title, meta description, keywords, canonical link, and hreflang alternates.
- Source: `front-page.md` (meta guidance), `i18n: seo.home`, `seoDesc.home`.
- Implementation: `useMeta` in `HomeView.vue`.

### JSON-LD (ProfessionalService)

- Purpose: Structured data for brand and offers.
- Fields: `name`, `alternateName`, `description`, `areaServed`, `url`, optional `sameAs`, `offers` (Public/Private Tap).
- Source: `front-page.md` JSON-LD example; `i18n: home.hero.*`, `publicTap.*`, `privateTap.*`.
- Implementation: `useMeta` in `HomeView.vue`.

### Hero

- Purpose: Convey brand promise and drive primary CTAs.
- Elements: H1, H2 (tagline), subline, CTAs (Public/Private), trust badges, hero image/alt.
- Source: `front-page.md → hero`; `i18n: home.hero.*`.

### Why companies need AI plumbing

- Purpose: Problem framing (governance, infra, compliance).
- Source: `front-page.md → why`; `i18n: home.why.*`.

### Three things, one toolbox

- Purpose: Describe offering composition (OSS toolkit, Public Tap, Private Tap).
- Source: `front-page.md → three`; `i18n: home.three.*`.

### Public Tap — test fast, pay upfront

- Purpose: Teaser for Public Tap details; bridge to route.
- Source: `front-page.md → public`; `i18n: home.public.*`.

### Private Tap — your own AI API, on dedicated GPUs

- Purpose: Teaser for Private Tap details; bridge to route.
- Source: `front-page.md → private`; `i18n: home.private.*`.

### We prove it works — before you rely on it

- Purpose: Proof-first ethos and artifacts overview.
- Source: `front-page.md → proof`; `i18n: home.proof.*`.

### Designed for IT teams and agencies — usable by anyone

- Purpose: Target audience clarity.
- Source: `front-page.md → audience`; `i18n: home.audience.*`.

### More links

- Purpose: Cross-links to FAQs and About.
- Source: `i18n: home.more.*`.

### Footer highlight

- Purpose: Reinforce brandline and Public Tap terms microcopy.
- Source: `front-page.md → footer`; `i18n: footer.*`.

### Imagery (planned)

- Purpose: Visual style per brand/moodboard.
- Assets: `hero_pipes.png` + future panels (problem→solution, curated models, dedicated pipe, proof visuals).
