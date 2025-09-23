# Navigatie & Interlinking — Ergonomieplan (Draft)

Doel: een site die snel te scannen is, duidelijke paden naar conversie biedt (call/POC/partner), en waar Qredits‑adviseurs moeiteloos door het ondernemers‑ en financieel plan kunnen klikken. Werkt met SSG (Hono JSX) en MDX‑content.

## 1) Informatiearchitectuur (IA)
- Primair (Top‑nav, desktop): Home • Legal • Agency • POC • Pricing • Checklists • SLA • About • Contact
- Secundair (Footer): Status • Plans • Bijlagen • Transparantie • Privacy • Cookies • AV • GitHub
- Docs/Plans hub: /plans (index) → Ondernemersplan • Financieel plan • Bijlagen
- Mobile: hamburger + large tap targets; dezelfde volgorde als desktop

## 2) Globale navigatie
- Top‑nav persistent; current route highlighted (aria‑current="page").
- Sticky CTA (right): “Book a call” + “POC aanvragen” (visible on all pages).
- Skip‑link (accessibility): “Ga naar hoofdinhoud”.

## 3) In‑page navigatie (lange pagina’s)
- Automatische ToC (ankerlinks op H2/H3 via rehype‑slug/autolink).
- “Terug naar boven” floating button (keyboard toegankelijk).
- Sectie‑permalinks (hover icon) voor deep‑links in gesprekken met adviseurs.

## 4) Breadcrumbs (docs/plans)
- Patroon: Home › Plans › [Document] › [Sectie]
- Genereer uit route + headings (frontmatter title als documentnaam).
- Altijd een “Terug naar Plans” link bovenaan docs.

## 5) Interlinking patronen
- Cross‑sell CTA’s per pagina:
  - Home → Pricing • POC • Legal/Agency varianten.
  - Legal → POC • Checklists (DPIA) • Plans (privacy/AVG secties).
  - Agency → Partner call • Case template • POC.
  - POC → Pricing • Plans (acceptatie‑KPI’s) • Contact.
  - Pricing → POC • SLA • FAQ → Plans voor diepgang.
  - Checklists → POC • Pricing • Plans.
  - SLA → Pricing • Status • Plans (SLA‑secties).
  - Plans (ondernemers/financieel) → POC • Pricing • Checklists • Bijlagen.
- “Lees ook” blok (RelatedLinks) onderaan elke sectie:
  - Bepaald door tags in frontmatter (bv. privacy, sla, pricing, poc).
  - Fallback: handmatige mapping in een JSON (related_routes.json).
- Inline links in MDX:
  - Headings krijgen deterministische ids (kebab‑case); gebruik ankerlinks `/plans/ondernemersplan#3-het-financieel-plan`.

## 6) URL‑ontwerp & consistentie
- Slugs in het NL, kort en semantisch: `/poc`, `/pricing`, `/checklists`, `/plans/ondernemersplan`.
- Canonicals per route; trailing slash één beleid (aan).
- UTM‑parameters ondersteunen tracking; strip bij SSR‑render (preserve in analytics).

## 7) CTA‑ergonomie
- Primaire CTA per pagina staat “boven de vouw” + herhaald na kernsecties.
- Sticky CTA (compact) op desktop; vaste “Contact” knop op mobiel onderaan.
- Microcopy: actiewoorden (“Plan een call”, “Start POC”).

## 8) Footer
- Kolommen: Product (SLA, Pricing, Status) • Resources (Checklists, Plans, Bijlagen) • Company (About, Contact) • Legal (Privacy, Cookies, AV) • Social (GitHub)
- Copyright + versienummer van docs (uit frontmatter) voor transparantie.

## 9) Zoek & vindbaarheid (MVP → +)
- MVP: ToC + ankers + duidelijke IA.
- +: client‑side zoek (Fuse.js) over headings/anchors met index JSON (~5–10kB).

## 10) Toegankelijkheid & toetsenbord
- Focus states zichtbaar; tabvolgorde logisch (logo → nav → hoofd → footer).
- Skip‑link, aria‑labels voor nav, breadcrumbs als nav[aria‑label="breadcrumbs"].

## 11) MDX‑docs interlinking specifics
- Frontmatter per bestand:
  ```
  ---
  title: Ondernemingsplan — Veighnsche
  version: 0.1
  tags: [privacy, sla, pricing]
  ---
  ```
- Tijdens build: maak een “docs index” (JSON) met { slug, title, headings[], tags[] }.
- Render ToC component + RelatedLinks gebaseerd op tags en headings nabijheid.

## 12) Patterns per route (concrete interlinks)
- / → linkkaarten naar: /legal, /agency, /poc, /pricing, /checklists, /plans
- /legal → cta: /poc + /checklists + /plans#privacy
- /agency → cta: /poc + /pricing + /marketing (later blog/cases)
- /poc → link: /pricing, /plans#acceptatie‑kpis, /contact
- /pricing → cta: /poc, link naar /sla en /plans#sla
- /checklists → cta: /poc, link naar /plans#bijlagen, /pricing
- /sla → link: /pricing, /status, /plans#sla
- /plans → indexkaarten naar “Ondernemingsplan”, “Financieel plan”, “Bijlagen”; met download knoppen; link naar /transparency
- /plans/ondernemersplan → CTA side rail: /poc, /pricing, /checklists; RelatedLinks op tags
- /plans/financieel‑plan → CTA: download PDF; link naar /pricing, /poc; link naar /transparency
- /transparency → linkkaarten naar roadmap/changelog/incidenten; knoppen naar /plans/* en downloads; link naar /status

## 13) Werking in code (schets)
- Components: `TopNav`, `FooterNav`, `Breadcrumbs`, `ToC`, `RelatedLinks`, `StickyCTA`.
- Data: `routes.ts` (sitemap), `related_routes.json`, `docs_index.json` (build‑generated).
- MDX → HTML: headings krijgen id’s; ToC leest uit `docs_index.json` voor ankers.

## 14) Meetpunten (funnel ergonomie)
- Click‑throughs TopNav → key routes.
- Scroll depth op /plans/* en ToC click ratio.
- CTA click rate op /poc en /pricing.
- “Lees ook” (RelatedLinks) click rate.

## 15) QA‑checklist (navigatie)
- Breadcrumbs correct op /plans/*.
- ToC werkt, ankers uniek.
- RelatedLinks toont max 3–5 relevante links, geen loops.
- Sticky CTA zichtbaar maar niet overlappend met footer.
