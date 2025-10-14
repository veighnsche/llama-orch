# ProblemSection Migration Guide

**Status**: ✅ **COMPLETE** - All problem sections now use the unified `ProblemSection` component.

The problem components have been unified into a single **reusable, prop-driven organism** at `@/components/organisms/ProblemSection/ProblemSection.tsx` that is used across all verticals (Providers, Developers, Enterprise, Home).

## ✨ What Changed

### Before
- Hardcoded content
- No customization
- Single-use component

### After
- Fully prop-driven API
- Reusable across verticals
- Enhanced visual hierarchy
- Built-in animations (tw-animate-css)
- CTA banner with dual buttons
- Loss tags on cards
- Kicker text support
- Full TypeScript types

## 🎨 New Features

1. **Tighter Visual Rhythm**: py-20 lg:py-28, updated spacing
2. **Kicker Text**: Small, muted label above headline
3. **Enhanced Typography**: font-extrabold tracking-tight
4. **Loss Tags**: Optional monetary impact tags on cards
5. **CTA Banner**: Dual-button conversion section
6. **Staggered Animations**: Cards fade in with delays (75ms, 150ms, 200ms)
7. **Reduced Motion Support**: `motion-reduce:animate-none`
8. **Tabular Numbers**: Consistent number rendering

## 📦 Component API

```tsx
import { ProvidersProblem } from '@/components/organisms/Providers/providers-problem'
import type { ProblemItem } from '@/components/organisms/Providers/providers-problem'

<ProvidersProblem
  kicker="Optional kicker text"
  title="Problem headline"
  subtitle="Optional subtitle with context"
  items={[
    {
      icon: <Icon className="h-6 w-6 text-destructive" />,
      title: "Problem Title",
      body: "Problem description",
      tag: "Optional loss tag (e.g., €50-200/mo)"
    }
  ]}
  ctaPrimary={{ label: "Primary CTA", href: "/path" }}
  ctaSecondary={{ label: "Secondary CTA", href: "#anchor" }}
  ctaCopy="CTA banner copy"
  className="optional-classes"
/>
```

## 🔄 Migration Examples

### Example 1: Developers Vertical

```tsx
import { ProvidersProblem } from '@/components/organisms/Providers/providers-problem'
import { Clock, CreditCard, Server } from 'lucide-react'

<ProvidersProblem
  kicker="The Developer's Dilemma"
  title="Stop Wasting Time on Infrastructure"
  subtitle="Building AI apps shouldn't require a PhD in DevOps."
  items={[
    {
      icon: <Clock className="h-6 w-6 text-destructive" />,
      title: "Setup Overhead",
      body: "Days spent configuring GPU environments instead of building features.",
      tag: "Lost productivity: 20+ hrs/mo"
    },
    {
      icon: <CreditCard className="h-6 w-6 text-destructive" />,
      title: "Cloud Lock-In",
      body: "Pay 3-5× markup for managed GPU instances with zero flexibility.",
      tag: "Overpaying: €500-2000/mo"
    },
    {
      icon: <Server className="h-6 w-6 text-destructive" />,
      title: "Vendor Risk",
      body: "Your app depends on a single cloud provider's uptime and pricing whims.",
    }
  ]}
  ctaPrimary={{ label: "Start Building", href: "/signup" }}
  ctaSecondary={{ label: "View API Docs", href: "/docs" }}
  ctaCopy="Ship faster with rbee's zero-config GPU marketplace."
/>
```

### Example 2: Enterprise Vertical

```tsx
import { ProvidersProblem } from '@/components/organisms/Providers/providers-problem'
import { TrendingDown, Shield, Zap } from 'lucide-react'

<ProvidersProblem
  kicker="The Enterprise Challenge"
  title="Your AI Budget Is Out of Control"
  subtitle="Cloud GPU costs are unpredictable and compliance is a nightmare."
  items={[
    {
      icon: <TrendingDown className="h-6 w-6 text-destructive" />,
      title: "Runaway Costs",
      body: "Cloud GPU bills double every quarter with no visibility into usage.",
      tag: "Budget overrun: 200%+"
    },
    {
      icon: <Shield className="h-6 w-6 text-destructive" />,
      title: "Compliance Risk",
      body: "Data sovereignty requirements make US-based clouds a non-starter.",
    },
    {
      icon: <Zap className="h-6 w-6 text-destructive" />,
      title: "Vendor Lock-In",
      body: "Migrating between cloud providers costs 6+ months of engineering time.",
    }
  ]}
  ctaPrimary={{ label: "Request Demo", href: "/enterprise/demo" }}
  ctaSecondary={{ label: "Download Whitepaper", href: "/enterprise/whitepaper" }}
  ctaCopy="Take back control with rbee's EU-hosted, cost-predictable GPU platform."
/>
```

## 🎯 Default Props (Providers)

The component ships with Providers-specific defaults:

- **Kicker**: "The Cost of Idle GPUs"
- **Title**: "Stop Letting Your Hardware Bleed Money"
- **Subtitle**: "Most GPUs sit idle ~90% of the time..."
- **Items**: 3 cards (Wasted Investment, Electricity Costs, Missed Opportunity)
- **CTA Primary**: "Start Earning" → /signup
- **CTA Secondary**: "Estimate My Payout" → #earnings-calculator

## ✅ Accessibility

- Semantic headings (h2, h3)
- Icons marked `aria-hidden="true"`
- Reduced motion support
- AA contrast compliance on destructive/red
- Tabular numbers for monetary values

## 🎨 Design Tokens

All styling uses design tokens:
- `border-border`
- `bg-background` / `bg-card`
- `text-foreground` / `text-muted-foreground`
- `text-destructive` / `bg-destructive`

## 📝 Migration Checklist

- [ ] Import `ProvidersProblem` and types
- [ ] Replace old component with new one
- [ ] Pass custom `items` array with icons
- [ ] Update `kicker`, `title`, `subtitle`
- [ ] Configure CTA buttons and copy
- [ ] Test animations (check reduced motion)
- [ ] Verify contrast on tags
- [ ] Remove old component file

## 🚀 Next Steps

1. Identify all existing "problem" organisms in the codebase
2. Port each to `ProvidersProblem` with custom props
3. Delete old organism files
4. Update design system exports

## 🔗 Related

- [Button Component](/components/atoms/Button/Button.tsx)
- [Design Tokens](/styles/tokens.css)
- [Animation Utilities](https://tailwindcss.com/docs/animation)
