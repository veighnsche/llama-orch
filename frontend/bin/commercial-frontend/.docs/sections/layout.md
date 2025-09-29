# Global Layout & Navigation

- Purpose: Ensure consistent shell with header, footer, and language switcher; wire SEO plumbing.
- Primary sources: `src/layouts/DefaultLayout.vue`, `src/components/NavBar.vue`, `src/components/SiteFooter.vue`, `src/components/LanguageSwitcher.vue`, `src/composables/useMeta.ts`.

## Sections

### Header / Navigation

- Purpose: Primary IA with localized labels; top-level routes; language switcher inclusion.
- Source: `NavBar.vue`; `i18n: nav.*`.

### Footer

- Purpose: Brandline, footer nav, and legal microcopy.
- Source: `SiteFooter.vue`; `i18n: footer.*`.

### Language Switcher

- Purpose: Toggle `en`/`nl`, persist in local storage, set `html[lang]`.
- Source: `LanguageSwitcher.vue`.

### SEO / Metadata Composable

- Purpose: Centralize dynamic title/description/keywords, canonical, hreflang, and JSON-LD.
- Source: `src/composables/useMeta.ts`.

### Styles & Tokens

- Purpose: Base styles and brand tokens; blueprint background utility.
- Source: `src/assets/base.css`, `src/styles/tokens.css`, `src/styles/blueprint.css`.
