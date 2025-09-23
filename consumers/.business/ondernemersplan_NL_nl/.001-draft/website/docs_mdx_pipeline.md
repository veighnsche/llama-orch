# MDX Pipeline — Business/Financial Plans op de Site

Doel: de ondernemers- en financiële plannen als MDX bronbestanden publiceren, renderen naar HTML voor de site (SSG), en exact dezelfde bron gebruiken voor PDF export.

## Bronnen
- Canonical: `.001-draft/` map (NL) — gebruik `.md` → rename/transform naar `.mdx` met frontmatter (titel, datum, versie).
- Inhoud: ondernemersplan secties + financieel plan + bijlagenindex.

## Parsing/Rendering (build-time)
- Gebruik remark/rehype:
  - `remark-parse`, `remark-gfm`, `remark-frontmatter`, `remark-mdx`
  - `remark-rehype`, `rehype-slug`, `rehype-autolink-headings`, `rehype-stringify`
- Output: veilige HTML string → in Hono JSX via `dangerouslySetInnerHTML` binnen een Layout component (SSG/prerender).

## Structuur
- `website/content/` (build cache): gekopieerde `.mdx` + gegenereerde HTML (tussenstap)
- Routes:
  - `/plans/ondernemersplan` (gecombineerde index)
  - `/plans/financieel-plan` (samenvatting + links)
  - `/plans/bijlagen` (navigatie)

## Frontmatter (voorbeeld)
```
---
 title: Ondernemingsplan — Veighnsche
 version: 0.1
 language: nl
 updated: 2025-01-05
---
```

## MDX componenten (optioneel)
- Kaart/Callout, Tabellen (GFM), ToC generator (op basis van headings).

## Validatie
- Buildstep faalt bij ontbrekende frontmatter/titel.
