# Hero Section Redesign Summary

## Overview

Complete redesign of the hero section following a 12-column grid layout with enhanced messaging hierarchy, improved accessibility, and compound visual storytelling.

## Key Changes

### Layout Architecture
- **12-column CSS Grid:** Cols 1–6 for messaging, cols 7–12 for visuals
- **Min-height:** Changed from `min-h-screen` to `min-h-[88svh]` for better viewport handling
- **Semantic HTML:** Added `aria-labelledby` landmark and proper heading structure

### Messaging Stack (Cols 1–6)

#### Top Utility Row
- Open-source badge (existing PulseBadge)
- Mini anchor links to Docs and GitHub with subtle separators
- Responsive: hidden on small screens

#### Kicker (New)
```
Self-Hosted • OpenAI-Compatible • Multi-Backend
```
- Eyebrow text, semibold, uppercase, tracking-wide
- Uses `text-muted-foreground` for hierarchy

#### Headline
- **Improved:** Reduced from `text-7xl` → `text-6xl` on desktop for better balance
- **Responsive:** `text-5xl md:text-6xl lg:text-6xl`
- **Motion:** Subtle opacity fade-in (respects `prefers-reduced-motion`)

#### Subcopy (Rewritten)
```
Run LLMs on your hardware—across any GPUs and machines. 
Build with AI, keep control, and avoid vendor lock-in.
```
- **Max-width:** `max-w-[58ch]` for optimal readability
- **Line-height:** `leading-8` for better rhythm
- **Length:** ~130 characters for scanability

#### Micro-proof Bullets (New)
Three concise value props with checkmarks:
- Your GPUs, your network
- Zero API fees
- Drop-in OpenAI API

#### CTA Group
- **Primary:** "Get Started Free" with `aria-label` and `data-umami-event="cta:get-started"`
- **Secondary:** "View Docs" as proper link with `href="/docs"` using `asChild` pattern
- **Tertiary:** GitHub star link with hover animation (arrow translates on hover)
- All CTAs have visible focus rings (`ring-2 ring-primary/40 offset-2`)

#### Trust Badges (Bottom Support Row)
Unified 4-item list with consistent styling:
1. Open Source (GPL-3.0) — Github icon
2. On GitHub — Star icon
3. OpenAI-Compatible — API badge
4. $0 • No Cloud Required — DollarSign icon

### Visual Stack (Cols 7–12)

#### Terminal Window
- **Shortened script:** 3–4 lines for clarity
- **GPU labels:** Title case (Workstation, Mac Studio, Gaming PC)
- **Cost label:** Changed to "Local Inference" for clarity
- **Accessibility:** Added `aria-live="polite"` to animated "Generating code..." line
- **Responsive:** `max-w-[520px] lg:max-w-none mx-auto`

#### Floating KPI Card (New Component)
- **Position:** Absolute, bottom-left of terminal
- **Styling:** `rounded-2xl shadow-lg/40 backdrop-blur-md bg-secondary/60 dark:bg-secondary/30`
- **Content:**
  - GPU Pool → 3 nodes / 7 GPUs
  - Cost → $0.00 / hr (uses `text-chart-3`)
  - Latency → ~34 ms
- **Animation:** Fade + rise entrance (y-2 → y-0, opacity 0→1, 300ms, delay 150ms)
- **Accessibility:** Respects `prefers-reduced-motion`

#### Network Diagram (New)
- **Visibility:** `hidden lg:block` (large screens only)
- **Container:** `aspect-[16/9] rounded-2xl ring-1 ring-border/60 bg-card`
- **Image:** Next.js `<Image>` with `priority` flag
- **Path:** `/images/homelab-network.png` (1280×720)
- **Alt text:** Descriptive for accessibility

## New Components Created

### FloatingKPICard
**Path:** `components/molecules/FloatingKPICard/FloatingKPICard.tsx`

- Client component with motion support
- Respects `prefers-reduced-motion`
- Uses semantic design tokens
- Exported from `components/molecules/index.ts`

## Accessibility Improvements

1. **Landmarks:** `<section aria-labelledby="hero-title">` with `id="hero-title"` on h1
2. **Focus Management:** All interactive elements have visible focus rings
3. **ARIA Live:** Animated terminal output uses `aria-live="polite"` to prevent verbosity
4. **Icon Labels:** Decorative icons marked with `aria-hidden="true"`
5. **Semantic Lists:** Trust badges and micro-proof bullets use proper `<ul>` and `<li>`
6. **Color Contrast:** All text meets WCAG AA (≥4.5:1) using semantic tokens

## Responsive Behavior

### Large (lg+)
- Two-column 12-grid layout
- Network diagram visible
- Floating KPI card visible
- Utility row with Docs/GitHub links

### Medium (md)
- Stacks vertically
- Terminal follows CTAs
- Floating KPI card visible
- Network diagram hidden

### Small (sm)
- Full stack layout
- Terminal constrained to `max-w-[520px] mx-auto`
- Utility links hidden
- Floating KPI card hidden
- Headline `text-5xl`

## Motion & Performance

### Animation Strategy
- **Headline:** Opacity fade (250ms)
- **CTAs:** Staggered entrance (50ms each) — *implemented via component state*
- **Terminal:** Scale from 98% + opacity (300ms) — *handled by existing component*
- **KPI Card:** Delayed entrance (150ms)
- **GitHub Arrow:** Hover translate (group-hover)

### Reduced Motion
All animations check `prefers-reduced-motion: reduce` and fall back to static display.

### Image Optimization
- Next.js `<Image>` with `priority` for hero image
- Proper width/height to prevent layout shift
- `object-cover` for aspect ratio preservation

## Copy Updates

| Element | Old | New |
|---------|-----|-----|
| Kicker | *(none)* | Self-Hosted • OpenAI-Compatible • Multi-Backend |
| Subcopy | "Orchestrate AI inference across any hardware—your GPUs, your network, your rules. Build with AI, monetize idle hardware, or ensure compliance. Zero vendor lock-in." | "Run LLMs on your hardware—across any GPUs and machines. Build with AI, keep control, and avoid vendor lock-in." |
| GPU Labels | workstation, mac-studio, gaming-pc | Workstation, Mac Studio, Gaming PC |
| Cost Label | Cost: | Local Inference |
| Trust Badge 1 | Open Source | Open Source (GPL-3.0) |
| Trust Badge 4 | No Cloud Required | $0 • No Cloud Required |

## QA Checklist

- [x] Text wraps cleanly at 320px
- [x] No overflow from badges or buttons
- [x] Focus order: badge → kicker → h1 → subcopy → primary CTA → secondary → tertiary → terminal
- [x] All colors use semantic tokens from `globals.css`
- [x] Dark mode support via design tokens
- [x] Visible focus rings on all interactive elements
- [x] ARIA labels and landmarks
- [x] Reduced motion support
- [x] Responsive breakpoints (sm, md, lg)
- [x] Semantic HTML (section, h1, ul, li)
- [x] Next.js Image optimization

## Files Modified

1. **HeroSection.tsx** — Complete redesign
2. **molecules/index.ts** — Added FloatingKPICard export

## Files Created

1. **FloatingKPICard/FloatingKPICard.tsx** — New molecule component
2. **public/images/README.md** — Image requirements documentation
3. **REDESIGN_SUMMARY.md** — This file

## Required Assets

**Image:** `/public/images/homelab-network.png` (1280×720)

See `/public/images/README.md` for detailed image specifications.

## Analytics Integration

- Primary CTA includes `data-umami-event="cta:get-started"` for tracking
- All external links have `rel="noopener noreferrer"` for security

## Performance Notes

- Hero section is client component (`'use client'`) due to motion hooks
- Terminal window remains as-is (already optimized)
- Image uses `priority` flag for above-fold loading
- Minimal JavaScript for motion detection

## Browser Support

- Modern browsers with CSS Grid support
- Fallback for `prefers-reduced-motion`
- SVH units for viewport height (with fallback to vh in older browsers)

---

**Outcome:** A confident, story-driven hero that communicates self-hosted control, demonstrates product capability visually, and funnels users into "Get Started" or "Docs" without clutter.
