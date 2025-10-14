# TechnicalSection Redesign Summary

**Date:** 2025-10-13  
**Status:** ✅ Complete

## Overview

Redesigned `TechnicalSection` organism for sharper engineering credibility and faster scanning. The new design emphasizes hierarchy, interactivity, and accessibility while maintaining dark theme consistency.

## Changes Implemented

### 1. Layout Restructure (12-Column Grid)

**Before:** Simple 2-column grid (`md:grid-cols-2`)  
**After:** Responsive 12-column grid with sticky right column

```tsx
<div className="grid grid-cols-12 gap-6 lg:gap-10 max-w-6xl mx-auto">
  {/* Left: col-span-12 lg:col-span-6 */}
  {/* Right: col-span-12 lg:col-span-6 lg:sticky lg:top-20 */}
</div>
```

- **Left column:** Architecture Highlights + diagram
- **Right column:** Technology Stack (sticky on `lg+` for cross-reference)
- **Max width:** Increased to `6xl` for better content distribution

### 2. Architecture Highlights Improvements

**Tightened Copy:**
- ✅ BDD-Driven Development → "42/62 scenarios passing (68% complete)" + meta: "Live CI coverage"
- ✅ Cascading Shutdown Guarantee → "No orphaned processes. Clean VRAM lifecycle."
- ✅ Process Isolation → "Worker-level sandboxes. Zero cross-leak."
- ✅ Protocol-Aware Orchestration → "SSE, JSON, binary protocols."
- ✅ Smart/Dumb Separation → "Central brain, distributed execution."

**Progress Bar Added:**
```tsx
<div className="relative h-2 rounded bg-muted">
  <div className="absolute inset-y-0 left-0 w-[68%] bg-chart-3 rounded" />
</div>
```
- Visual indicator: 68% BDD coverage (42/62 scenarios)
- Label + meta text for context

**Variant Change:** All bullets now use `variant="check"` for consistency

### 3. Architecture Diagram (Desktop-Only)

Added Next.js `<Image>` component with:
- **Path:** `/images/rbee-arch.svg`
- **Dimensions:** 920×560
- **Visibility:** `hidden md:block` (desktop only)
- **Styling:** `rounded-2xl ring-1 ring-border/60 shadow-sm`
- **Alt text:** Comprehensive description for accessibility and AI image generation

```tsx
<Image
  src="/images/rbee-arch.svg"
  width={920}
  height={560}
  className="hidden md:block rounded-2xl ring-1 ring-border/60 shadow-sm"
  alt="clean systems diagram of rbee: control plane (Rhai rules, scheduler) orchestrating workers across GPUs; arrows for SSE/JSON/binary protocols; isolated processes with graceful shutdown; dark UI, teal and amber accent lines"
  priority
/>
```

### 4. Technology Stack → Interactive Spec Cards

**Before:** Plain boxes with `bg-muted`  
**After:** Interactive cards with hover states and staggered animations

**Card Structure:**
```tsx
<article
  role="group"
  aria-label="Tech: Rust"
  className="bg-muted/60 border border-border rounded-xl p-4 hover:border-primary/40 transition-colors motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-400"
>
  <div className="font-semibold text-foreground">Rust</div>
  <div className="text-sm text-muted-foreground">Performance + memory safety.</div>
</article>
```

**Technologies:**
1. **Rust** — Performance + memory safety.
2. **Candle ML** — Rust-native inference.
3. **Rhai Scripting** — Embedded, sandboxed policies.
4. **SQLite** — Embedded, zero-ops DB.
5. **Axum + Vue.js** — Async backend + modern UI.

**Animation Delays:** `delay-100`, `delay-200`, `delay-300`, `delay-400`, `delay-500`

### 5. Open Source CTA Card

**Enhanced styling:**
- Background: `bg-primary/10` with `border-primary/30`
- Padding: Increased to `p-5`
- Button: Wrapped in `<a>` tag with proper `target` and `rel` attributes
- Aria-label: "View rbee source on GitHub"

**Architecture Docs Link:**
```tsx
<Link
  href="/docs/architecture"
  className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground transition-colors"
>
  Read Architecture →
</Link>
```

### 6. Micro-Kickers & Subtitle

**Section Subtitle:**
> "Rust-native orchestrator with process isolation, protocol awareness, and policy routing via Rhai."

**Kickers:**
- Left column: "CORE PRINCIPLES" (`text-xs tracking-wide uppercase`)
- Right column: "STACK"

### 7. Motion & Responsiveness

**Section-level animation:**
```tsx
className="motion-safe:animate-in motion-safe:fade-in-50 motion-safe:duration-500"
```

**Card animations:**
- Staggered entrance: `slide-in-from-bottom-2` with delays
- All prefixed with `motion-safe:` to respect user preferences

**Responsive behavior:**
- Mobile (`< lg`): Stacked layout, no sticky positioning
- Desktop (`lg+`): Side-by-side with sticky right column

### 8. Accessibility & Semantics

**Section-level:**
```tsx
<SectionContainer
  headingId="tech-title"
  // Renders: <section aria-labelledby="tech-title">
/>
```

**Spec cards:**
- Each card is an `<article role="group" aria-label="Tech: [name]">`
- Interactive elements have descriptive `aria-label` attributes

**Contrast:**
- `bg-muted/60` maintains WCAG AA compliance
- Text uses `text-foreground` for primary content

### 9. Component API Extensions

#### BulletListItem
**New prop:** `meta?: string`
```tsx
export interface BulletListItemProps {
  // ... existing props
  meta?: string // Right-aligned muted text
}
```

**Implementation:**
```tsx
<div className="flex items-start justify-between gap-2">
  <div className="font-medium text-foreground">{title}</div>
  {meta && <div className="text-xs text-muted-foreground whitespace-nowrap">{meta}</div>}
</div>
```

#### SectionContainer
**New props:**
- `subtitle?: string | ReactNode` (already existed, now typed as ReactNode)
- `headingId?: string` (for `aria-labelledby`)

**Changes:**
- Subtitle now renders as `<div>` instead of `<p>` to support ReactNode
- Section gets `aria-labelledby={headingId}`
- Heading gets `id={headingId}`

## Implementation Checklist

- [x] Swap container grid to 12-col; apply sticky right on lg+
- [x] Replace highlight bullets with tightened copy + add progress bar
- [x] Insert desktop-only architecture `<Image>` with prompt-rich alt
- [x] Convert stack items to animated spec cards; keep OSS CTA as accent card
- [x] Add subtitle under section title; add "Core Principles / Stack" kickers
- [x] Wire View Source to GitHub repo and "Read Architecture" to docs
- [x] Add a11y attributes and motion-safe classes
- [x] Extend BulletListItem with `meta` prop
- [x] Extend SectionContainer with `headingId` prop

## Files Modified

1. **`TechnicalSection.tsx`** — Complete redesign
2. **`BulletListItem.tsx`** — Added `meta` prop with right-aligned display
3. **`SectionContainer.tsx`** — Added `headingId` prop and `aria-labelledby`

## Next Steps

### Required Asset
Create the architecture diagram SVG at:
```
/public/images/rbee-arch.svg
```

**Suggested content (based on alt text):**
- Control plane with Rhai rules engine and scheduler
- Worker nodes distributed across GPUs
- Protocol arrows (SSE, JSON, binary)
- Process isolation boundaries
- Graceful shutdown flow indicators
- Dark theme with teal/amber accent lines

### Optional Enhancements
1. **GitHub URL:** Update placeholder in line 144:
   ```tsx
   href="https://github.com/yourusername/rbee"
   ```
2. **Architecture Docs:** Create `/docs/architecture` page
3. **Responsive Testing:** QA on sm/md/lg/xl breakpoints
4. **Motion Testing:** Verify with `prefers-reduced-motion` enabled

## Design Rationale

### Why 12-Column Grid?
- More granular control over responsive layouts
- Industry standard (Bootstrap, Tailwind, Material)
- Easier to create asymmetric layouts in future

### Why Sticky Right Column?
- Keeps tech stack visible while scrolling through architecture details
- Improves cross-referencing between principles and implementation
- Common pattern in documentation sites

### Why Staggered Animations?
- Guides eye flow from top to bottom
- Creates sense of depth and polish
- Respects `prefers-reduced-motion` via `motion-safe:` prefix

### Why Check Variant for Bullets?
- Visual consistency (all items are "completed" features)
- Stronger credibility signal than dots
- Aligns with "Built by Engineers" positioning

## Performance Notes

- **Image:** Uses Next.js `<Image>` with `priority` flag for LCP optimization
- **Animations:** CSS-based (no JS), hardware-accelerated
- **Sticky positioning:** Native CSS, no scroll listeners
- **Bundle size:** No new dependencies added

## Browser Support

- **Grid:** All modern browsers (IE11 not supported)
- **Sticky positioning:** All modern browsers
- **CSS animations:** All modern browsers with graceful degradation
- **motion-safe:** Supported in all browsers with `prefers-reduced-motion` media query

---

**Status:** Ready for review and asset creation
