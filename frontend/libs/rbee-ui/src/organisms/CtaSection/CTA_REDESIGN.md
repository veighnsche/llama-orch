# CTASection Redesign - Reusable Props-Driven Component

**Date:** 2025-10-13  
**Status:** ✅ Complete

## Summary

Converted `DevelopersCTA` into a reusable, props-driven `CTASection` organism that replaces the complex legacy implementation. The component now supports flexible CTA layouts with optional eyebrows, gradients, and configurable alignment.

## Key Changes

### 1. **Props-Driven API**
Created clean interface covering 90% of CTA variants:

```typescript
type CTAAction = {
  label: string
  href: string
  iconLeft?: React.ComponentType<{ className?: string }>
  iconRight?: React.ComponentType<{ className?: string }>
  variant?: 'default' | 'outline' | 'ghost' | 'secondary' | 'link' | 'destructive'
}

type CTASectionProps = {
  eyebrow?: string
  title: string
  subtitle?: string
  primary: CTAAction
  secondary?: CTAAction
  note?: string                     // small trust line under buttons
  align?: 'center' | 'left'         // default 'center'
  emphasis?: 'none' | 'gradient'    // toggles subtle bg flourish
  id?: string
  className?: string
}
```

### 2. **Layout & Composition**

**Section wrapper:**
- Base: `py-24 border-b border-border bg-background`
- Optional gradient: `relative isolate before:absolute before:inset-0 before:bg-[radial-gradient(70%_50%_at_50%_0%,theme(colors.primary/10),transparent_60%)]`

**Content structure:**
```
Section
├─ Container (max-w-7xl, px-6 lg:px-8)
└─ Content block (max-w-3xl, text-center or text-left)
   ├─ Eyebrow (optional, rounded-full badge)
   ├─ Title (text-4xl sm:text-5xl, font-bold)
   ├─ Subtitle (text-lg, text-muted-foreground)
   ├─ Actions row (flex-col sm:flex-row, gap-3)
   │  ├─ Primary button (size="lg", bg-primary)
   │  └─ Secondary button (size="lg", variant="outline")
   └─ Trust note (text-sm, text-muted-foreground)
```

### 3. **Visual Improvements**

**Typography:**
- Bumped content container to `max-w-3xl` (was 2xl) for better line breaks
- Title uses `tracking-tight` for premium feel
- Subtitle uses `text-balance` for natural wrapping

**Button spacing:**
- Consistent `gap-3` between buttons
- Icons properly spaced with `mr-2` (left) or `ml-2` (right)
- Primary button icon has hover animation: `group-hover:translate-x-1`

**Alignment options:**
- `align="center"`: text-center, justify-center (default)
- `align="left"`: text-left, justify-start

### 4. **Motion Hierarchy (tw-animate-css)**

All animations use `motion-safe:` prefix:

- **Eyebrow:** `fade-in-50 duration-300`
- **Title:** `fade-in slide-in-from-bottom-2 duration-400 delay-100`
- **Subtitle:** `fade-in-50 duration-400 delay-150`
- **Buttons:** `zoom-in-50 duration-300 delay-200`
- **Note:** `fade-in-50 duration-300 delay-300`

### 5. **Accessibility**

- Single `<h2>` for section title
- Actions are semantic `<Link>` components via `Button asChild`
- Icons marked `aria-hidden="true"`
- Optional `id` prop for deep-linking
- Respects `prefers-reduced-motion` automatically

### 6. **Implementation Details**

**Button rendering:**
```tsx
<Button asChild size="lg" variant={variant}>
  <Link href={action.href}>
    {IconLeft && <IconLeft className="mr-2 h-4 w-4" aria-hidden="true" />}
    {action.label}
    {IconRight && <IconRight className="ml-2 h-4 w-4" aria-hidden="true" />}
  </Link>
</Button>
```

**Gradient emphasis:**
```tsx
emphasis === 'gradient' &&
  'relative isolate before:absolute before:inset-0 before:bg-[radial-gradient(...)] before:pointer-events-none'
```

## Migration

### Files Changed
- ✅ **Replaced:** `components/organisms/CtaSection/CtaSection.tsx` (131 → 153 lines)
- ✅ **Updated:** `components/organisms/index.ts` (exports CTASection + types)
- ✅ **Updated:** `app/page.tsx` (uses new CTASection with props)
- ✅ **Updated:** `app/developers/page.tsx` (uses shared CTASection)
- ✅ **Deleted:** `components/organisms/Developers/developers-cta.tsx`

### Homepage Usage
```tsx
<CTASection
  title="Stop depending on AI providers. Start building today."
  subtitle="Join 500+ developers who've taken control of their AI infrastructure."
  primary={{ label: 'Get started free', href: '/getting-started', iconRight: ArrowRight }}
  secondary={{ label: 'View documentation', href: '/docs', iconLeft: BookOpen, variant: 'outline' }}
  note="100% open source. No credit card required. Install in 15 minutes."
  emphasis="gradient"
/>
```

### Developers Page Usage
```tsx
<CTASection
  title="Stop Depending on AI Providers. Start Building Today."
  subtitle="Join 500+ developers who've taken control of their AI infrastructure."
  primary={{ label: 'Get Started Free', href: '/getting-started', iconRight: ArrowRight }}
  secondary={{ label: 'View Documentation', href: '/docs', iconLeft: GitHubIcon, variant: 'outline' }}
  note="100% open source. No credit card required. Install in 15 minutes."
/>
```

## QA Checklist

- [x] Single shared CTASection organism; no duplicates
- [x] Props-driven; no hardcoded copy
- [x] Title/subtitle readable at all breakpoints
- [x] Buttons have correct contrast and focus rings
- [x] Icons marked decorative (`aria-hidden`)
- [x] Optional gradient flourish is subtle
- [x] Motion uses tw-animate-css only
- [x] Respects `prefers-reduced-motion`
- [x] TypeScript compilation passes
- [x] Semantic HTML (`<h2>`, `<Link>`)
- [x] Alignment options work (center/left)
- [x] Button hover animations smooth

## Design Tokens

All spacing, colors, and radii use theme tokens:

- **Borders:** `border-border`
- **Backgrounds:** `bg-background`, `bg-primary`, `bg-transparent`
- **Text:** `text-foreground`, `text-muted-foreground`, `text-primary-foreground`
- **Radii:** `rounded-full`, `rounded-md`
- **Spacing:** `gap-3`, `py-24`, `px-6`, `mt-3`, `mt-6`, `mt-8`

## Comparison: Before vs After

### Before (DevelopersCTA)
```tsx
export function DevelopersCTA() {
  return (
    <section className="border-b border-border bg-background py-24">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            Stop Depending on AI Providers. Start Building Today.
          </h2>
          {/* Hardcoded content... */}
        </div>
      </div>
    </section>
  )
}
```

**Issues:**
- Hardcoded copy
- No props
- Not reusable
- No animations
- No alignment options
- No gradient option

### After (CTASection)
```tsx
export function CTASection({
  eyebrow,
  title,
  subtitle,
  primary,
  secondary,
  note,
  align = 'center',
  emphasis = 'none',
  id,
  className,
}: CTASectionProps) {
  // Props-driven, reusable, animated
}
```

**Benefits:**
- Fully props-driven
- Reusable across all pages
- Staggered animations
- Alignment options
- Optional gradient emphasis
- Type-safe API
- Semantic HTML with `asChild` Links

## Performance

- No client-side JavaScript required (static rendering)
- CSS animations use `motion-safe:` prefix
- Buttons use Next.js `<Link>` for optimal routing
- Staggered animations via inline `style={{ animationDelay }}`
- Gradient uses CSS pseudo-element (no extra DOM nodes)

## Future Enhancements

Possible additions without breaking changes:

1. **Tertiary action:** Add `tertiary?: CTAAction` for 3-button layouts
2. **Image variant:** Add `image?: string` for visual CTAs
3. **Badge variant:** Add `badge?: { text: string; color: string }` for eyebrow badges with icons
4. **Size variants:** Add `size?: 'default' | 'compact' | 'hero'` for different scales

---

**Result:** A clean, configurable CTA section with stronger hierarchy, optional background emphasis, and reusable actions—ready to drop anywhere without one-off tweaks. Single source of truth for all CTA layouts.
