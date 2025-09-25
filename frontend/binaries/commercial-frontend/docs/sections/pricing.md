# Pricing (Draft)

- Purpose: Present initial draft pricing for Public Tap credits and Private Tap GPU-hour snapshots.
- Primary sources: `.002-draft/ADR-XXX-public-tap-pricing.md`, `.002-draft/front-page.md`, `src/views/PricingView.vue`, `src/i18n/*`.

## Sections

### SEO / Head

- Purpose: Localized meta and canonical + hreflang.
- Source: `i18n: seo.pricing`, `seoDesc.pricing`.

### Public Tap — Credits (Draft)

- Purpose: Baseline per 1M tokens and estimated token counts for packs.
- Source: `ADR-XXX-public-tap-pricing.md`; `i18n: publicTap.bPrice1..bPrice4`, `pricing.note1`.

### Private Tap — GPU-hour (Draft)

- Purpose: A100/H100 snapshots and caveats.
- Source: `front-page.md`; `i18n: privateTap.bPrice1..bPrice3`, `pricing.note2`.

### Notes

- Purpose: Mark everything as draft and subject to benchmarking/provider changes.
- Source: ADRs and `front-page.md`.
