# PricingSection Redesign Summary

**Date:** 2025-01-13  
**Components Updated:** `PricingSection` (organism), `PricingTier` (molecule)

## Overview

Redesigned the pricing section for stronger value framing, improved conversion, and enhanced mobile clarity. All changes use Tailwind utilities with `motion-safe:` prefixes for animations (no Framer Motion). Maintains atomic design structure and reuses existing atoms.

---

## ✅ Implementation Checklist

### 1. Section Header (Clarity + Trust)
- [x] Added subtitle: "Run rbee free at home. Add collaboration and governance when your team grows."
- [x] Added micro-trust badges row with icons: Open source, OpenAI-compatible, Multi-GPU, No feature gates
- [x] Applied `motion-safe:animate-in motion-safe:fade-in-50 motion-safe:duration-500` to header wrapper
- [x] Icons from Lucide: `Unlock`, `Zap`, `Layers`, `Shield`

### 2. Pricing Grid Composition
- [x] Switched to 12-column grid: `grid grid-cols-12 gap-6 lg:gap-8 max-w-6xl mx-auto`
- [x] Card spans: `col-span-12 md:col-span-4` for all three tiers
- [x] Team plan uses `order-first md:order-none` for mobile prominence
- [x] Equal height cards via `flex flex-col h-full` on PricingTier root
- [x] CTA in `mt-auto pt-6` container for alignment
- [x] Staggered entrance animations:
  - Home/Lab: `duration-500`
  - Team: `duration-500 delay-100`
  - Enterprise: `duration-500 delay-200`

### 3. Featured Plan Enhancement (Team)
- [x] Glow ring: `shadow-[0_0_0_1px_var(--primary)] shadow-primary/20 hover:shadow-primary/30`
- [x] Top accent bar: `absolute top-0` with `h-1 bg-primary rounded-full`
- [x] Enhanced ribbon: `text-[11px] bg-primary/15 text-primary font-semibold`
- [x] Additional ring: `ring-1 ring-primary/30`

### 4. Billing Toggle (Monthly/Yearly)
- [x] Segmented control above grid: `bg-muted p-1 rounded-lg`
- [x] Active state: `bg-background text-foreground shadow-sm`
- [x] "Save 2 months" badge on Yearly button
- [x] State managed via `useState` hook
- [x] Extended PricingTier props:
  - `priceYearly?: string | number`
  - `isYearly?: boolean`
  - `saveBadge?: string`
- [x] Team plan shows €990/year when toggled with "2 months free" badge

### 5. Feature Lists (Benefit-First Copy)
- [x] **Home/Lab:**
  - "Unlimited GPUs on your hardware"
  - "OpenAI-compatible API"
  - "Multi-modal models"
  - "Active community support"
  - "Open source core"
- [x] **Team:**
  - "Everything in Home/Lab"
  - "Web UI for cluster & models"
  - "Shared workspaces & quotas"
  - "Priority support (business hours)"
  - "Rhai policy templates (rate/data)"
- [x] **Enterprise:**
  - "Everything in Team"
  - "Dedicated, isolated instances"
  - "Custom SLAs & onboarding"
  - "White-label & SSO options"
  - "Enterprise security & support"

### 6. CTA Design & Routes
- [x] **Home/Lab:** "Download rbee" (outline) → `/download`
- [x] **Team:** "Start 30-Day Trial" (primary) → `/signup?plan=team`
- [x] **Enterprise:** "Contact Sales" (outline) → `/contact?type=enterprise`
- [x] Micro-notes under each CTA:
  - Home/Lab: "Local use. No feature gates."
  - Team: "Cancel anytime during trial."
  - Enterprise: "We'll reply within 1 business day."
- [x] CTAs use Next.js `Link` with `asChild` pattern

### 7. Visual Polish (PricingTier)
- [x] Card root classes:
  - Base: `bg-card/90 backdrop-blur supports-[backdrop-filter]:bg-card/75 rounded-2xl p-7 md:p-8 border-2`
  - Hover: `motion-safe:hover:translate-y-[-2px] motion-safe:hover:shadow-lg motion-safe:transition-all`
- [x] Title hierarchy:
  - Title: `text-xl font-semibold tracking-tight`
  - Price: `text-4xl font-extrabold`
  - Period: `text-sm text-muted-foreground ml-2`
- [x] Feature list: `mt-5 space-y-2 text-sm`
- [x] Check icon: `text-chart-3` with `aria-hidden`
- [x] Footer micro-note: `text-[12px] text-muted-foreground/90 mt-2 text-center`

### 8. PricingTier API Changes
- [x] Added optional props (backward-compatible):
  - `priceYearly?: string | number`
  - `currency?: 'USD' | 'EUR' | 'GBP' | 'CUSTOM'`
  - `ctaHref?: string`
  - `footnote?: string`
  - `isYearly?: boolean`
  - `saveBadge?: string`
- [x] Root element changed from `div` to `section` with `aria-labelledby`
- [x] Feature list has `role="list"` and `aria-label`
- [x] Button has descriptive `aria-label`
- [x] External `className` merged via `cn()`

### 9. Editorial Visual
- [x] Added Next.js `Image` component (desktop-only, `hidden lg:block`)
- [x] Path: `/images/pricing-hero.webp`
- [x] Dimensions: 1100×620
- [x] Alt text: "isometric dark UI of a small homelab rig effortlessly scaling into a multi-node GPU cluster; neon teal and amber accents, clean editorial style, cinematic contrast"
- [x] Priority loading enabled
- [x] Styling: `rounded-2xl ring-1 ring-border/60 shadow-sm mx-auto mt-10`

### 10. Accessibility & Semantics
- [x] Each card is a `<section>` with `aria-labelledby` linking to title
- [x] Buttons have descriptive `aria-label`: e.g., "Start 30-Day Trial for Team plan"
- [x] Feature lists have `role="list"` and `aria-label`
- [x] Icons have `aria-hidden="true"`
- [x] Toggle buttons have `aria-pressed` state
- [x] All animations prefixed with `motion-safe:` utilities
- [x] Color contrast maintained on `bg-card` vs foreground

### 11. Footer Reassurance
- [x] Strengthened copy: "Every plan includes the full rbee orchestrator. No feature gates. No artificial limits."
- [x] Added compliance note: "Prices exclude VAT. OSS license applies to Home/Lab."
- [x] Styling: `text-[12px] text-muted-foreground/80 mt-2`

---

## 🎨 Design Tokens Used

- **Colors:** `primary`, `muted`, `muted-foreground`, `chart-3`, `border`, `card`, `foreground`
- **Spacing:** Tailwind scale (1-12, px, auto)
- **Typography:** `text-xs`, `text-sm`, `text-lg`, `text-xl`, `text-4xl`
- **Borders:** `rounded-md`, `rounded-lg`, `rounded-2xl`, `rounded-full`
- **Shadows:** `shadow-sm`, `shadow-lg`, custom `shadow-[0_0_0_1px_var(--primary)]`
- **Transitions:** `transition-all` with `motion-safe:` prefix

---

## 📦 Dependencies

- **React:** `useState` hook for billing toggle
- **Next.js:** `Image` for editorial visual, `Link` for routing
- **Lucide React:** `Shield`, `Zap`, `Layers`, `Unlock`, `Check`
- **Radix UI:** `Slot` (via Button component)
- **Tailwind CSS:** All styling via utility classes

---

## 🧪 Build Verification

```bash
cd frontend/bin/commercial && npm run build
```

**Result:** ✅ Build successful (compiled with warnings unrelated to pricing components)

---

## 📱 Responsive Behavior

- **Mobile (< 768px):**
  - Cards stack vertically (`col-span-12`)
  - Team plan appears first (`order-first`)
  - Editorial image hidden
  - Trust badges wrap naturally
  
- **Tablet (768px - 1024px):**
  - Three-column grid (`md:col-span-4`)
  - Normal order restored (`md:order-none`)
  - Editorial image still hidden
  
- **Desktop (≥ 1024px):**
  - Three-column grid with larger gaps (`lg:gap-8`)
  - Editorial image visible (`lg:block`)
  - All animations active (if motion not reduced)

---

## 🔗 Related Files

- **Organism:** `/components/organisms/PricingSection/PricingSection.tsx`
- **Molecule:** `/components/molecules/PricingTier/PricingTier.tsx`
- **Atom:** `/components/atoms/Button/Button.tsx` (reused)
- **Utility:** `/lib/utils.ts` (`cn` helper)

---

## 🚀 Next Steps (Optional Enhancements)

1. **Tooltip Component:** Add tooltips for "Rhai", "White-label", "SSO" terms
2. **Image Asset:** Create or source the `/images/pricing-hero.webp` editorial visual
3. **A/B Testing:** Test Monthly vs. Yearly default state
4. **Analytics:** Track toggle interactions and CTA clicks
5. **i18n:** Externalize copy to translation files
6. **Dark Mode:** Verify contrast ratios in both themes

---

## 📝 Notes

- All animations respect `prefers-reduced-motion` via `motion-safe:` prefix
- Component remains backward-compatible (all new props are optional)
- Maintains atomic design structure (organism → molecule → atom)
- No external animation libraries required (pure Tailwind)
- Build output shows no errors related to pricing components
