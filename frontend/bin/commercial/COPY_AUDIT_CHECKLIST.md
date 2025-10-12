# Copy Audit Checklist — Frontend Commercial Components

This checklist captures remaining copy and link fixes to align components with stakeholder docs and policies.

## Completed in this pass

- **navigation.tsx**: “Docs” links updated to GitHub docs URL.
- **hero-section.tsx**: License badge → GPL-3.0-or-later; removed hard-coded star count.
- **features/cross-node-orchestration.tsx**: Terminology aligned to “pool/machines”; kept CLI literals.
- **features/real-time-progress.tsx**: Explicit cancel → POST /v1/cancel (idempotent).
- **what-is-rbee.tsx**: “open-source” → “open source”.
- **features-section.tsx**: “OpenAI-Compatible” label; avoid automatic fallback wording; “CPU Fallback” → “CPU Backend”.
- **features/core-features-tabs.tsx**: Same as above.
- **how-it-works-section.tsx**: Installer → git/cargo; OS claim softened.

## Remaining tasks

- **Marketplace commission alignment**
  - [ ] `frontend/bin/commercial/components/providers/providers-marketplace.tsx`: Update “15% commission / 85% you keep” to stakeholder range (30–40% platform fee; provider 60–70%). Adjust example math accordingly.

- **Marketplace timing**
  - [ ] `frontend/bin/commercial/components/use-cases/use-cases-primary.tsx`: Change “Joins rbee marketplace (M3)” → “(M5)” or “(future)”.

- **“Real‑time” hyphenation sweep**
  - [ ] Standardize to “real‑time” across:
    - `frontend/bin/commercial/components/enterprise/enterprise-how-it-works.tsx`
    - `frontend/bin/commercial/components/enterprise/enterprise-use-cases.tsx`
    - `frontend/bin/commercial/components/enterprise/enterprise-testimonials.tsx`
    - `frontend/bin/commercial/components/audience-selector.tsx`
    - `frontend/bin/commercial/components/developers/developers-pricing.tsx`
    - `frontend/bin/commercial/components/enterprise/enterprise-comparison.tsx`
    - `frontend/bin/commercial/components/pricing/pricing-tiers.tsx`
    - `frontend/bin/commercial/components/social-proof-section.tsx`
    - `frontend/bin/commercial/components/use-cases-section.tsx`
    - `frontend/bin/commercial/components/use-cases/use-cases-primary.tsx`

- **OpenAI-compatible casing**
  - [ ] Ensure “OpenAI‑compatible” in body text and “OpenAI‑Compatible” in headings/badges. Review:
    - `frontend/bin/commercial/components/developers/developers-hero.tsx`
    - `frontend/bin/commercial/components/pricing/pricing-comparison.tsx`
    - `frontend/bin/commercial/components/enterprise/*`
    - Any other references flagged by repo search.

- **No automatic fallback language**
  - [ ] Remove/adjust any remaining phrasing that implies automatic CPU fallback; prefer “orchestrate across the backends you configure.”

- **Pricing plan naming & trials**
  - [ ] `frontend/bin/commercial/components/pricing/pricing-tiers.tsx`: Consider renaming tiers to match stakeholders (Starter / Professional / Enterprise), or reconcile docs to UI.
  - [ ] `frontend/bin/commercial/components/pricing/pricing-faq.tsx`: Verify 30‑day trial claim for Team/Professional; soften if not available.

- **Footer placeholders**
  - [ ] `frontend/bin/commercial/components/footer.tsx`: Replace `href="#"` for Discord, Twitter/X, Blog, About, Contact Sales, Privacy Policy, Terms of Service with real URLs or remove until ready.

- **EU‑only claims**
  - [ ] Where copy claims “EU‑only routing” (e.g., `use-cases/use-cases-primary.tsx`), qualify with “when compliance mode is enabled” if not always enforced.

- **Navigation CTAs**
  - [ ] `frontend/bin/commercial/components/navigation.tsx`: Wire “Join Waitlist” to a working route or form if available.

## Optional enhancements

- **Greps to validate**
  - Real‑time: `rg -n "real time" frontend/bin/commercial/components`
  - OpenAI compatible variants: `rg -n "OpenAI[- ]Compatible|OpenAI compatible" frontend/bin/commercial/components`
  - Placeholder links: `rg -n "href=\"#\"" frontend/bin/commercial/components`

- **Dynamic GitHub stars**
  - Replace fixed star counts with a dynamically fetched value or non-numeric label.
