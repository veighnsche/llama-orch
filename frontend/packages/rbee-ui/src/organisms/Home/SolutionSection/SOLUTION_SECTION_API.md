# SolutionSection API Reference

**Component:** `SolutionSection.tsx`  
**Type:** Organism (Reusable)  
**Version:** 2.0 (Extended API)

---

## Overview

`SolutionSection` is a flexible organism for presenting "How It Works" sections with features, steps, and optional compliance metrics or earnings data. Used by Enterprise, Developers, and Providers audiences.

---

## Props API

### Core Props

```typescript
interface SolutionSectionProps {
  // Header
  kicker?: string                  // Eyebrow text (e.g., "How rbee Works")
  eyebrowIcon?: ReactNode          // NEW: Icon next to kicker (e.g., <Shield />)
  title: string                    // Required: H2 heading
  subtitle?: string                // Optional: Support copy below title
  
  // Content
  features: Feature[]              // Required: 4 feature tiles
  steps: Step[]                    // Required: How It Works steps
  earnings?: Earnings              // Optional: Metrics/earnings sidebar
  aside?: ReactNode                // NEW: Custom sidebar (overrides earnings)
  illustration?: ReactNode         // NEW: Decorative background image
  
  // CTAs
  ctaPrimary?: CTA                 // Primary button
  ctaSecondary?: CTA               // Secondary button
  ctaCaption?: string              // NEW: Helper text below CTAs
  
  // Meta
  id?: string                      // Section ID (used for aria-labelledby)
  className?: string               // Additional CSS classes
}
```

### Type Definitions

```typescript
type Feature = {
  icon: ReactNode                  // Lucide icon (h-6 w-6)
  title: string                    // Feature title
  body: string                     // Feature description
  badge?: string | ReactNode       // NEW: Policy badge (e.g., "GDPR Art. 30")
}

type Step = {
  title: string                    // Step title
  body: string                     // Step description
}

type EarningRow = {
  model: string                    // Left label (e.g., "Data Sovereignty")
  meta: string                     // Left meta (e.g., "GDPR Art. 44")
  value: string                    // Right value (e.g., "100%")
  note?: string                    // Right note (e.g., "EU-only")
}

type Earnings = {
  title?: string                   // Sidebar title (default: "Compliance Metrics")
  rows: EarningRow[]               // Metrics rows
  disclaimer?: string              // Footer disclaimer
  imageSrc?: string                // (Unused in current implementation)
}

type CTA = {
  label: string                    // Button text
  href: string                     // Link destination
  ariaLabel?: string               // Optional ARIA label
}
```

---

## Usage Examples

### Basic Usage (Minimum Required)

```tsx
<SolutionSection
  title="How It Works"
  features={[
    {
      icon: <Shield className="h-6 w-6" />,
      title: "Feature Title",
      body: "Feature description."
    },
    // ... 3 more features
  ]}
  steps={[
    { title: "Step 1", body: "Step description." },
    // ... more steps
  ]}
/>
```

### Enterprise Usage (Full Features)

```tsx
<SolutionSection
  id="how-it-works"
  kicker="How rbee Works"
  eyebrowIcon={<Shield className="h-4 w-4" />}
  title="EU-Native AI Infrastructure"
  subtitle="Enterprise-grade, self-hosted AI..."
  features={[
    {
      icon: <Shield className="h-6 w-6" />,
      title: "100% Data Sovereignty",
      body: "Data stays on your infrastructure.",
      badge: "GDPR Art. 44"
    },
    // ... more features
  ]}
  steps={[
    { title: "Deploy On-Premises", body: "Install rbee..." },
    // ... more steps
  ]}
  earnings={{
    title: "Compliance Metrics",
    rows: [
      { model: "Data Sovereignty", meta: "GDPR Art. 44", value: "100%", note: "EU-only" },
      // ... more rows
    ],
    disclaimer: "Consult your legal team..."
  }}
  illustration={
    <Image src="/decor/bg.webp" width={1200} height={640} ... />
  }
  ctaPrimary={{ label: "Request Demo", href: "/demo" }}
  ctaSecondary={{ label: "View Docs", href: "/docs" }}
  ctaCaption="EU data residency guaranteed."
/>
```

### Custom Sidebar (Advanced)

```tsx
<SolutionSection
  // ... other props
  aside={
    <div className="lg:sticky lg:top-24 lg:self-start">
      <CustomMetricsPanel />
    </div>
  }
/>
```

---

## Layout Structure

```
<section id={id} aria-labelledby="{id}-h2">
  {illustration}                    <!-- Decorative background -->
  
  <div>                             <!-- Container -->
    <!-- Header -->
    <div>
      <p>{eyebrowIcon} {kicker}</p>
      <h2 id="{id}-h2">{title}</h2>
      <p>{subtitle}</p>
    </div>
    
    <!-- Feature Tiles (4-up grid) -->
    <div className="grid lg:grid-cols-4">
      {features.map(f => (
        <div>
          {f.icon}
          <h3>{f.title} {f.badge}</h3>
          <p>{f.body}</p>
        </div>
      ))}
    </div>
    
    <!-- Steps + Sidebar -->
    <div className="grid lg:grid-cols-[1.2fr_0.8fr]">
      <!-- Steps Card -->
      <div>
        <h3>How It Works</h3>
        <ol>
          {steps.map((s, i) => (
            <li>
              <div>{i + 1}</div>
              <div>{s.title}</div>
              <div>{s.body}</div>
            </li>
          ))}
        </ol>
      </div>
      
      <!-- Aside (custom or earnings) -->
      {aside || earnings}
    </div>
    
    <!-- CTAs -->
    <div>
      <Button>{ctaPrimary.label}</Button>
      <Button>{ctaSecondary.label}</Button>
      <p>{ctaCaption}</p>
    </div>
  </div>
</section>
```

---

## Styling Guidelines

### Feature Tiles

- **Layout:** `flex items-start gap-4` (icon beside text)
- **Icon container:** `rounded-xl bg-primary/10 p-3`
- **Badge:** `rounded-full border border-primary/20 bg-primary/10 px-2 py-0.5 text-[10px]`

### Steps Card

- **Wrapper:** `rounded-2xl border border-border bg-card/40 p-6 md:p-8`
- **List:** `<ol className="space-y-6">`
- **Number badge:** `h-8 w-8 rounded-full bg-primary/10 text-primary`

### Metrics Sidebar

- **Wrapper:** `rounded-2xl border border-border bg-card p-6`
- **Sticky:** `lg:sticky lg:top-24 lg:self-start`
- **Row:** `flex items-start justify-between gap-4`

### Illustration

- **Position:** `absolute left-1/2 -translate-x-1/2 -z-10`
- **Styling:** `opacity-15 blur-[0.5px]`
- **Responsive:** `hidden md:block`

---

## Motion System

All animations use `tw-animate-css`:

1. **Header:** `animate-in fade-in-50 slide-in-from-bottom-2 duration-500`
2. **Tiles:** `animate-in fade-in-50 [animation-delay:100ms]`
3. **Steps:** `animate-in fade-in-50 [animation-delay:150ms]`
4. **Sidebar:** `animate-in fade-in-50 slide-in-from-right-2 [animation-delay:200ms]`

**Respects:** `prefers-reduced-motion` via `motion-reduce:animate-none`

---

## Accessibility

### ARIA

- Section: `aria-labelledby="{id}-h2"`
- H2: `id="{id}-h2"`
- Steps: `<ol role="list">`
- Each step: `<li aria-label="Step {n}: {title}">`

### Icons

- All decorative icons: `aria-hidden="true"`
- Illustration: `aria-hidden="true"`

### Contrast

- Subtitle: `text-foreground/85` (≥4.5:1)
- Body text: `text-muted-foreground` (≥4.5:1)
- Metrics title: `text-foreground` (≥4.5:1)

---

## Responsive Behavior

### Mobile (<768px)

- Single column layout
- Tiles stack vertically (2 columns on sm)
- Steps and sidebar stack
- Illustration hidden
- CTAs stack vertically

### Tablet (768-1023px)

- Tiles: 2 columns
- Steps and sidebar stack
- Illustration visible

### Desktop (≥1024px)

- Tiles: 4 columns
- Steps + sidebar: 2 columns (1.2fr / 0.8fr)
- Sidebar sticky
- Illustration visible

---

## Migration Guide (v1 → v2)

### Breaking Changes

**None.** All new props are optional.

### New Features

1. **`eyebrowIcon`** — Add icon next to kicker
2. **`badge`** on features — Add policy references
3. **`illustration`** — Add decorative background
4. **`aside`** — Custom sidebar (overrides earnings)
5. **`ctaCaption`** — Helper text below CTAs

### Updated Styles

- Feature tiles: Left-aligned (was center)
- Steps: Semantic `<ol>` (was `<div>`)
- Sidebar: Sticky on lg+ (was static)
- Background: Centered radial (was left)

---

## Best Practices

### Content

- **Features:** 4 items (optimal for lg:grid-cols-4)
- **Steps:** 3-5 items (more than 5 gets cluttered)
- **Metrics:** 3-4 rows (fits sticky sidebar height)

### Copy

- **Title:** 8-12 words, outcome-focused
- **Subtitle:** 20-30 words, benefit-driven
- **Feature bodies:** 8-12 words, action-oriented
- **Step bodies:** 12-18 words, clear instructions

### Icons

- **Feature icons:** `h-6 w-6` (Lucide)
- **Eyebrow icon:** `h-4 w-4` (Lucide)
- Always include `aria-hidden="true"`

### Badges

- Use for policy references: "GDPR Art. 44"
- Keep short: 2-4 words max
- Don't overuse: 1-2 per section

---

## Related Components

- **EnterpriseHero** — Hero section above this
- **DevelopersSolution** — Developer-focused variant
- **ProvidersSolution** — Provider-focused variant

---

**Last Updated:** 2025-10-13  
**Version:** 2.0  
**Status:** ✅ Production Ready
