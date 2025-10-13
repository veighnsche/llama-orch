# SolutionSection Component

## Overview

The `SolutionSection` component presents rbee's value proposition through a tightly structured narrative focused on **cost**, **privacy**, **stability**, and **hardware utilization**. The redesign adds proof badges, compatibility information, privacy pledges, and clear CTAs while preserving the architecture diagram in the layout flow.

## Design Principles

1. **Hierarchy & Motion**: Staggered fade-in animations (respects `prefers-reduced-motion`)
2. **Brand Expression**: Uses semantic design tokens from `globals.css`
3. **Reusability**: Leverages existing atoms (Badge, Button) and molecules (FeatureCard, SectionContainer)
4. **Accessibility**: Proper heading order (H2 → H3), ARIA labels, descriptive alt text

## Structure

```
SolutionSection
├── Header (H2 + subtitle via SectionContainer)
├── Proof Badges (4 outline badges)
├── Architecture Diagram (preserved from original)
├── Key Benefits Grid (4 FeatureCards with micro-metrics)
├── Privacy & Control Pledge (callout with shield icon)
└── CTA Row (primary + secondary buttons)
```

## Typography

- **Headline**: `text-4xl md:text-5xl font-semibold tracking-tight`
- **Subtitle**: `text-lg md:text-xl text-muted-foreground max-w-3xl`
- **Card Titles**: `text-base font-semibold`
- **Card Descriptions**: `text-sm text-muted-foreground leading-6`
- **Micro-metrics**: `text-xs text-muted-foreground`

## Layout & Spacing

- **Container**: `max-w-7xl mx-auto px-4 sm:px-6 lg:px-8`
- **Vertical Rhythm**: `space-y-12 md:space-y-16`
- **Grid**: `grid md:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6`

## Motion System

- **Animation**: `fade-in-up` (0.4s ease-out)
- **Stagger**: 60ms between elements
- **Reduced Motion**: Animations disabled via `prefers-reduced-motion` media query
- **Hover**: Cards translate up 0.5px with shadow on hover

## Components Used

### Atoms
- **Badge**: Proof badges with `variant="outline"`
- **Button**: Primary (solid) and secondary (ghost) CTAs

### Molecules
- **SectionContainer**: Wraps section with title/subtitle
- **FeatureCard**: Enhanced with `children` prop for micro-metrics
- **PledgeCallout**: Security pledge with shield icon
- **ArchitectureDiagram**: System topology visualization

## Content

### Proof Badges
1. OpenAI-compatible API
2. Runs on CUDA · Metal · CPU
3. Zero API fees (electricity only)
4. Code stays in your network

### Benefits
1. **Zero Ongoing Costs** — Pay only for electricity. No API bills, no per-token surprises.
   - Metric: _Typical: €0.08–€0.15/kWh_
2. **Complete Privacy** — Code and data never leave your network. Audit-ready by design.
   - Metric: _Audit-ready logs, EU-friendly_
3. **Locked to Your Rules** — Models update only when you approve. No breaking changes.
   - Metric: _No forced updates_
4. **Use All Your Hardware** — CUDA, Metal, and CPU orchestrated as one pool.
   - Metric: _Multi-node orchestration_

### Privacy Pledge
> Your models. Your rules. rbee enforces zero-trust auth, immutable audit trails, and strict bind policies—so your code stays yours.

### CTAs
1. **Primary**: "Run on my GPUs" → `#quickstart`
2. **Secondary**: "See scheduler policy" → `/docs/scheduler-policy`

## Accessibility

- ✅ Heading hierarchy: H2 (section) → H3 (cards)
- ✅ ARIA labels on compatibility strip
- ✅ `aria-hidden="true"` on decorative icons
- ✅ Descriptive alt text on logo images (doubles as image generation prompts)
- ✅ Reduced motion support
- ✅ WCAG AA contrast in light/dark modes

## Mobile Responsiveness

- Benefits grid stacks 1-per-row on mobile (`md:grid-cols-2 lg:grid-cols-4`)
- Compatibility strip wraps gracefully (`flex-wrap`)
- CTA buttons stack vertically on mobile (`flex-col sm:flex-row`)
- Typography scales down on smaller screens

## Dark Mode

All colors use semantic tokens that adapt automatically:
- `bg-secondary` / `text-secondary-foreground`
- `bg-card` / `text-card-foreground`
- `text-muted-foreground`
- `text-primary`
- Chart colors (`chart-1` through `chart-5`)

## Future Enhancements

- [ ] Add Intersection Observer for scroll-triggered animations
- [ ] Consider adding a "Learn More" link to each benefit card
- [ ] A/B test CTA copy variations
- [ ] Add telemetry to track CTA click-through rates
