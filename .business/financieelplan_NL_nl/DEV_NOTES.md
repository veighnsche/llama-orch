# DEV_NOTES — One-time learning from .002-draft (no runtime coupling)

Scope: This document captures developer-facing notes after reading the marketing/business draft under `.business/ondernemersplan_NL_nl/.002-draft`. It informs template tone and defaults only. The runtime generator will NOT read `.002-draft`.

## Tone of voice (from .002-draft)

- Direct, practical, proof-first; avoid hype.
- Independent tradesman metaphor (AI plumber); local, approachable, accountable.
- Transparency and evidence: logs, metrics, documentation; align with EU AI Act spirit.
- Clear, business-credible language; no “revolution/disruption” phrasing.

Implication for templates: concise, audit-like prose; add brief, neutral explanations of assumptions; keep disclaimers explicit: “Indicatief; geen fiscale advisering.”

## Recurring concepts & signals (marketing → template stance)

- Open-source, transparent tooling; reproducibility and determinism.
- EU-friendly posture: documentation and transparency valued.
- Pricing/commercial narratives (prepaid credits, GPU packs) exist, but are out-of-scope for the finance calculator’s logic except for tone/disclaimer.
- No direct mentions of DSO/DPO/KOR/seasonality in the marketing draft; treat these as normal finance parameters with sensible defaults (see below).

## Proposed domain defaults (for schema defaults and template wording)

These defaults exist to keep example behavior predictable and match common SME assumptions in NL. They are overridable by input.

- DSO (debiteuren, dagen): 30 dagen → shift omzet cash +1 maand (rounding by 30-day months).
- DPO (crediteuren, dagen): 14 dagen → shift COGS cash typically stays in same month or +1 maand depending on day rounding.
- BTW-percentage: 21% standaard.
- BTW-model: `omzet_enkel` (eenvoudig en consistent).
- BTW-periode: `monthly` (CLI default), maar `quarterly` is supported.
- KOR: `false` by default; if `true` or `btw_vrij=true`, BTW-afdracht = 0 met uitleg.
- Seizoen: none by default; if provided, allow month-specific multipliers (e.g., `"2025-12": {omzet_pm_pct: 30}`).
- Afschrijving: lineair per maand; start per item `start_maand` (prorata via start-datum inclusie) met vaste levensduur (mnd) > 0.
- Lening: annuïteit met nominale maandrente r = jaar%/12/100; steun voor `grace_mnd` en `alleen_rente_in_grace`.

Rationale: Predictability and transparency (themes in the draft) favor simple, explainable defaults.

## Terminology map (NL labels for outputs)

Use these labels consistently in headings, column names, and prose:

- Overzicht → `Overzicht`
- Investeringen & Financiering → `Investeringen & Financiering`
- Liquiditeit → `Liquiditeit`
- Exploitatie → `Exploitatie`
- Qredits / Maandlasten → `Qredits / Maandlasten`
- Belastingen → `Belastingen (indicatief)`
- Schema (mapping) → `Schema (mapping input → output)`

Key line items:

- Omzet → `Omzet`
- COGS (kostprijs omzet) → `COGS / Inkoop`
- Brutomarge → `Marge (Bruto)`
- OPEX / Overhead → `OPEX / Overhead`
- Afschrijving → `Afschrijving`
- Rente → `Rente`
- Resultaat voor belasting → `Resultaat (v/bel)`
- Begin kas → `Begin kas`
- Eind kas → `Eind kas`
- BTW-afdracht → `BTW-afdracht`
- Inkomsten/Uitgaven (liquiditeit) → `Inkomsten` / `Uitgaven`
- Lening maandlasten → `Rente p/m`, `Aflossing p/m`, `Restschuld`
- DSO/DPO → `DSO (dagen)`, `DPO (dagen)` (explanatory fields, not table columns)
- Seizoen → `Seizoensaanpassing`

Columns (CSV expectations):

- Exploitatie (maandelijks): `maand, omzet, cogs, marge, opex_<categorie>..., opex_totaal, afschrijving, rente, resultaat_vb`
- Liquiditeit (maandelijks): `maand, begin_kas, in_omzet, in_overig, uit_cogs, uit_opex, uit_btw, uit_rente, uit_aflossing, eind_kas`
- Investeringen: `omschrijving, levensduur_mnd, start_maand, afschrijving_pm, bedrag`
- Financiering: `bron, bedrag, rente_nominaal_jr_pct, looptijd_mnd, grace_mnd`
- Amortisatie: `maand, verstrekker, rente_pm, aflossing_pm, restschuld`
- Belastingen (indicatief): `regime, btw_pct, btw_model, kor, btw_vrij`

## Template tone examples (short, neutral)

- Overzicht: “Start kas (eigen inbreng + leningen maand 1): …” / “Laagste kasstand: … (maand …)”
- Liquiditeit: “Kasstroomtabel met DSO/DPO verschuivingen en BTW-afdracht per periode.”
- Exploitatie: “Omzet → COGS → marge → OPEX → afschrijving → rente → resultaat (v/bel).”
- Maandlasten: “Volledig aflosschema; beoordeling van maandlast-dekking in Overzicht.”
- Belastingen: “Regime: IB/VPB. BTW-model: … KOR/vrijstelling gedocumenteerd indien van toepassing. Indicatief; geen fiscale advisering.”

## Runtime separation (lock-in for design)

- The generator will NOT read `.002-draft` (or any repo scans) at runtime.
- All prose is fixed in `templates/*.md.tpl`, populated via placeholders only.
- Changing YAML changes numbers and deterministic short verdicts only; prose stays identical.

## Open items to carry into templates

- Ensure disclaimers are present and consistent.
- Keep brand voice: practical, audit-like phrasing.
- Avoid marketing superlatives; prefer factual, short sentences.
- Headings and labels strictly in NL as above.
