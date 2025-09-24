# Contradictions and Tensions (Draft)

This report lists subtle contradictions or cross-document tensions found across the Markdown files in `consumers/.business/ondernemersplan_NL_nl/.002-draft/`. Each item includes citations and a suggested resolution.

Scope of scan (15 files):
- `ADR-XXX-public-tap-prepaid-credits.md`
- `ADR-XXX-public-tap-pricing.md`
- `USP.md`
- `brand.md`
- `competitors.md`
- `front-page.md`
- `identity.md`
- `index.md`
- `mesh-site-architecture.md`
- `moodboard-extended.md`
- `moodboard.md`
- `naming.md`
- `page-layers.md`
- `services.md`
- `target-audiences.md`

---

## 1) Public Tap billing model: prepaid vs “pay‑as‑you‑go metered”

- Evidence for prepaid, non‑refundable credits (12‑month shelf life):
  - `ADR-XXX-public-tap-prepaid-credits.md` — Decision bullets (lines ~21–26, 34–41)
  - `ADR-XXX-public-tap-pricing.md` — Overview + Shelf Life (lines ~9–11, 33–37)
  - `front-page.md` — Section 3 “Public Tap” (lines ~83–99)
  - `target-audiences.md` — Hobbyists/devs “Public Tap prepaid credits” (lines ~18–23)
- Conflicting phrasing that implies metered/postpaid:
  - `services.md` — “Shared, metered API … Pay-as-you-go pricing.” (lines ~38–41)
  - `identity.md` — Level 1: “Pay-as-you-go metered API.” (lines ~65–70)

Suggested resolution:
- Unify to “prepaid, non‑refundable credits (12‑month validity)” for Public Tap everywhere.
- Edit `services.md` and `identity.md` to remove “metered/postpaid” language and add a short one-liner like: “Prepaid credits; usage decrements balance; service pauses at 0.”

---

## 2) Brand naming: “Orchyra” vs “AI Plumbing by Vince” (JSON‑LD)

- Picked brand name:
  - `naming.md` — Picked → “Brand: Orchyra; Tagline: Private LLM Hosting in the Netherlands.” (lines ~57–62)
- JSON‑LD still uses a different service name:
  - `front-page.md` — Schema.org block → `"name": "AI Plumbing by Vince"` (lines ~199–226, specifically ~206)

Suggested resolution:
- Decide presentation hierarchy: e.g., “Orchyra — by Vince (independent AI plumber).”
- Update JSON‑LD `name` to “Orchyra” (or “Orchyra — AI Plumbing by Vince”) to match the chosen brand, and keep “Vince” prominent in page content per `brand.md`.

---

## 3) SEO anchor vs page headings

- SEO guidance says to consistently use “Private LLM Hosting” across headings/metadata:
  - `naming.md` — Notes (lines ~64–68)
- Current H1 on the front page is “AI plumbing, done right.”
  - `front-page.md` — Hero H1 (lines ~19–23)

Suggested resolution:
- Either adjust the hero to include the anchor (e.g., “Private LLM Hosting — AI plumbing, done right.”) or add an immediately visible H2/subline containing the exact phrase “Private LLM Hosting in the Netherlands,” to satisfy the “consistent across headings” intent while keeping the brand voice.

---

## 4) Site redesign timing: “no redesign in Draft 2” vs detailed mesh plan documents

- Draft 2 explicitly says “zonder herontwerp site” (no site redesign in this draft):
  - `index.md` — Section C (lines ~110–116)
- Separate docs propose a layered/mesh IA with page counts and categories:
  - `page-layers.md` — full layered plan
  - `mesh-site-architecture.md` — mesh diagram and routing principles

Suggested resolution:
- Clarify status at the top of `page-layers.md` and `mesh-site-architecture.md` (e.g., “Proposal for post‑Draft‑2 IA. Not implemented yet.”), or update `index.md` to call out these as future‑phase proposals to avoid perceived contradiction of scope.

---

## 5) Visual guidance: “cartoon plumber optional” vs “don’t show literal human plumbers”

- Allows an optional cartoon plumber mascot:
  - `moodboard.md` — Mascot (optional) (lines ~56–60)
- Says “Don’t show literal human plumbers”:
  - `moodboard-extended.md` — Do’s and Don’ts (line ~144)

Interpretation and suggestion:
- These can co‑exist if the prohibition targets photorealistic human imagery. Clarify wording in `moodboard-extended.md` to “Don’t use photos of real human plumbers; vector/cartoons are acceptable if kept serious.” Add a cross‑reference to the mascot section in `moodboard.md`.

---

## 6) Minor phrasing alignment: “Public Tap — shared, metered API”

- Even if “metered” intends “usage decrements prepaid balance,” it reads like postpaid. See items in §1.

Suggested resolution:
- Standardize phrasing to “shared API with prepaid credits” or “prepaid credit‑metered API” to avoid misinterpretation.

---

## 7) Naming examples still show “Vinch” after “Orchyra” was picked

- Recommended pairing shows “Brand: Vinch” while the final selection is “Brand: Orchyra.”
  - `naming.md` — Recommended Pairing (lines ~45–53) vs Picked (lines ~57–62)
- Example hero still uses “Vinch.”
  - `naming.md` — Example front‑page hero block (lines ~50–53)

Suggested resolution:
- Update the example hero to “Orchyra,” or annotate the “Recommended Pairing” section as historical/superseded by “Picked.”
- Ensure any other references to “Vinch” as the brand are aligned with the final choice “Orchyra.”

## Notes (non‑contradictions verified)

- GPL posture appears consistent across docs (`USP.md`, `brand.md`, `identity.md`) and matches the repository plan to migrate to GPL‑3.0‑or‑later.
- “Docs on separate subdomain” is consistently stated (`page-layers.md`, `mesh-site-architecture.md`, `target-audiences.md`).

---

## Next actions (edit checklist)

- [x] Update `services.md` (§1)
- [x] Update `identity.md` (§1)
- [x] Update `front-page.md` JSON‑LD `name` (§2)
- [x] Adjust front-page H1/H2 per SEO anchor (§3)
- [x] Add “proposal status” note to `page-layers.md` and `mesh-site-architecture.md` (§4)
- [x] Clarify photo vs. cartoon guidance in `moodboard-extended.md` (§5)
