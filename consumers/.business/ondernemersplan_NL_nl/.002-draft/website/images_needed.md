# Images Needed — Draft 002 Website (v2)

Goal: minimal images with maximum impact. Prefer SVG for decorative shapes/icons and WEBP/AVIF for photos. Keep total payload small (<150 kB images on first paint).

## Global
- Logo (SVG)
  - File: `img/logo_llama_orch.svg`
  - Notes: monochrome variant for dark backgrounds.
  - Alt: "llama‑orch"
- Favicon + App icons
  - Files: `img/favicon.svg` (or `.ico` fallback), `img/icon-192.png`, `img/icon-512.png`
  - Sizes: 192×192, 512×512
- Open Graph cover
  - File: `img/og_llama_orch_v2.webp`
  - Size: 1200×630 (landscape)
  - Alt: "llama‑orch — Open‑source AI op eigen servers"

## HERO
- Background decorative shape (SVG)
  - File: `img/hero_wave_v2.svg`
  - Usage: subtle radial/wave shape behind hero copy (non‑blocking)
  - Alt: decorative (empty alt: "")
- Optional proof strip logos (SVG/WEBP) — only if available
  - Files: `img/proof/logo_client_*.svg`
  - Alt: client name (e.g., "Client X logo")

## VALUE (Waarom wij)
- Small line icons (SVG) — 3–6
  - Files: `img/icons/value_eu.svg`, `img/icons/value_lockin.svg`, `img/icons/value_metrics.svg`, `img/icons/value_speed.svg`, `img/icons/value_security.svg`, `img/icons/value_sla.svg`
  - Alt: succinct feature, e.g., "EU/AVG‑first pictogram"

## STEPS (Van intake naar go‑live)
- Optional step icons (SVG) — 3–6
  - Files: `img/icons/step_discovery.svg`, `img/icons/step_templates.svg`, `img/icons/step_config.svg`, `img/icons/step_test.svg`, `img/icons/step_accept.svg`, `img/icons/step_golive.svg`
  - Alt: step name

## LEGAL
- Sector illustration/photo (WEBP)
  - File: `img/hero_legal_v2.webp`
  - Size: 1600×900 (desktop), 800×450 (mobile)
  - Alt: "Privacy‑gerichte juridische sector illustratie"

## AGENCY
- Partnership illustration/photo (WEBP)
  - File: `img/hero_agency_v2.webp`
  - Size: 1600×900 (desktop), 800×450 (mobile)
  - Alt: "Samenwerking/partnerschap illustratie"

## POC
- Timeline/checklist illustration (SVG/WEBP)
  - File: `img/illus_poc_timeline_v2.svg` (preferred) or `img/illus_poc_timeline_v2.webp`
  - Alt: "POC in 10 werkdagen — tijdlijn"

## PRICING & SLA
- Subtle background accent (SVG) — optional
  - File: `img/bg_pricing_stripes_v2.svg`
  - Alt: decorative (empty alt)
- Most‑popular badge icon (SVG) — optional
  - File: `img/icons/badge_popular.svg`
  - Alt: "Populair"

## CHECKLISTS
- Two PDF thumbnails (WEBP)
  - Files: `img/thumb_checklist_dpia_v2.webp`, `img/thumb_checklist_onprem_v2.webp`
  - Size: 480×640 (portrait) or 640×480 (landscape)
  - Alt: "DPIA checklist omslag", "On‑Prem Readiness omslag"

## CONTACT
- Optional team/headshot or abstract (WEBP/SVG)
  - File: `img/illus_contact_v2.webp` (or `img/illus_contact_v2.svg`)
  - Size: 1200×800 (desktop), 800×533 (mobile)
  - Alt: "Contact illustratie"

---

## Export & performance guidelines
- Prefer SVG for logos, icons, and decorative backgrounds; set `role="img"` only when informative.
- Use WEBP (or AVIF if available) for photos; quality ~70–80; aim <60 kB per hero/section image.
- Provide mobile renditions (srcset) where big images are used: 800w and 1600w.
- Use empty alt ("") for purely decorative images; meaningful alt for informative ones.
- Filenames: lowercase, hyphenated, with `v2` suffix where relevant.
- Place assets under `/.002-draft/website/img/` with subfolders `icons/`, `proof/` as needed.

## Mapping to sections in code
- `index_v2.html`: targets per section have IDs `#home`, `#value`, `#steps`, `#legal`, `#agency`, `#poc`, `#pricing`, `#checklists`, `#contact`.
- CSS personalities are in `styles_v2.css` via `.sec.<section>` rules; decorative images can be added as `<img>` or background images if needed.
