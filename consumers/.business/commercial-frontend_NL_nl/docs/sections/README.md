# Orchyra Marketing Site — Sections Index

This folder contains per-page section maps to structure investigation and iteration.

## Files

- home.md — Home (front page)
- public-tap.md — Public Tap (prepaid credits)
- private-tap.md — Private Tap (dedicated GPUs)
- pricing.md — Pricing (draft)
- proof.md — Proof (logs, metrics, SSE)
- faqs.md — FAQs
- about.md — About
- contact-legal.md — Contact & Legal
- layout.md — Global layout & navigation (optional)

## Investigation Checklist Template

Copy this template to the top of each page doc as you work through it:

- [ ] Content validated against `.002-draft` sources
- [ ] SEO: title, description (≤155 chars), keywords, canonical, hreflang
- [ ] JSON-LD (if applicable): fields populated and localized
- [ ] Cross-links present and correct
- [ ] Imagery: assets identified or placeholders defined
- [ ] Accessibility: headings order, link text, focus states
- [ ] Notes / Open questions

## Source Pointers

- Business plan and drafts: `consumers/.business/ondernemersplan_NL_nl/.002-draft/`
- i18n: `src/i18n/en.ts`, `src/i18n/nl.ts`
- Views: `src/views/`
- Meta helper: `src/composables/useMeta.ts`
- Styles/tokens: `src/styles/tokens.css`, `src/assets/base.css`, `src/styles/blueprint.css`
