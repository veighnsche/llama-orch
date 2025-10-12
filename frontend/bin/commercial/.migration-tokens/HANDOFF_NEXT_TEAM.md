# Handoff: Shared Components Migration

**Date:** 2025-10-12  
**From:** Analysis Team  
**To:** Implementation Team  
**Status:** READY FOR IMPLEMENTATION  
**Estimated Effort:** 7-8 weeks (1 developer)

---

## What We Completed

✅ **Analyzed** 60+ component files across `/frontend/bin/commercial/components/`  
✅ **Identified** 25 duplicated component patterns with 150+ usage instances  
✅ **Documented** complete migration plan with priorities and estimates  
✅ **Calculated** impact: ~4,800 lines of code reduction (~50% reduction)  
✅ **Defined** folder structure: `/components/primitives/`

---

## Your Mission

**Extract 25 duplicated component patterns into reusable primitives.**

This will:
- Reduce code duplication by ~50%
- Improve maintainability (single source of truth)
- Ensure design consistency across all pages
- Speed up future development

---

## Critical Decisions Made

### 1. Folder Structure: `/components/primitives/`

**Why "primitives"?**
- Aligns with Radix UI terminology (already in use)
- Not tied to Atomic Design but conveys same concept
- Clear hierarchy: `ui/` (library) → `primitives/` (custom) → `*-section.tsx` (composed)

**Structure:**
```
/components/
├── ui/                          # shadcn/ui library components (DON'T TOUCH)
├── primitives/                  # YOUR WORK HERE
│   ├── layout/
│   │   └── SectionContainer.tsx
│   ├── badges/
│   │   └── PulseBadge.tsx
│   ├── cards/
│   │   ├── FeatureCard.tsx
│   │   ├── TestimonialCard.tsx
│   │   ├── AudienceCard.tsx
│   │   └── SecurityCrateCard.tsx
│   ├── code/
│   │   ├── TerminalWindow.tsx
│   │   └── CodeBlock.tsx
│   ├── icons/
│   │   └── IconBox.tsx
│   ├── lists/
│   │   ├── CheckListItem.tsx
│   │   └── BulletListItem.tsx
│   ├── progress/
│   │   └── ProgressBar.tsx
│   ├── stats/
│   │   └── StatCard.tsx
│   │   └── StepNumber.tsx
│   ├── callouts/
│   │   └── BenefitCallout.tsx
│   ├── indicators/
│   │   └── TrustIndicator.tsx
│   ├── diagrams/
│   │   ├── ArchitectureDiagram.tsx
│   │   └── ArchitectureDiagram.stories.tsx
│   ├── navigation/
│   │   ├── NavLink.tsx
│   │   └── NavLink.stories.tsx
│   ├── footer/
│   │   ├── FooterColumn.tsx
│   │   └── FooterColumn.stories.tsx
│   ├── pricing/
│   │   ├── PricingTier.tsx
│   │   ├── PricingTier.stories.tsx
│   │   ├── ComparisonTable.tsx
│   │   └── ComparisonTable.stories.tsx
│   ├── earnings/
│   │   ├── EarningsCard.tsx
│   │   ├── EarningsCard.stories.tsx
│   │   ├── GPUListItem.tsx
│   │   └── GPUListItem.stories.tsx
│   ├── solutions/
│   │   ├── SolutionCard.tsx
│   │   └── SolutionCard.stories.tsx
│   └── index.ts                 # Barrel export
├── hero-section.tsx             # Existing sections (migrate these)
├── features-section.tsx
└── ...

### 2. No Breaking Changes

**CRITICAL:** All existing pages must maintain **identical visual appearance and behavior**.

This is a pure refactor. No design changes. No new features.

### 3. Testing Requirements

Each primitive component **MUST** have:

1. **Storybook story** (`.stories.tsx`)
   - Default state
   - All variants
   - Interactive controls
   - Props documentation

2. **Unit tests** (Vitest - `.spec.tsx`)
   - Props validation
   - Variant rendering
   - Event handlers (if applicable)
   - Accessibility checks

3. **TypeScript types**
   - Strict prop types
   - Exported interfaces
   - JSDoc comments

---

## Implementation Order (FOLLOW THIS)

### Phase 1: High-Impact Primitives (Week 1) ⭐⭐⭐

**Start here. These give maximum ROI.**

1. **SectionContainer** (15+ usages, ~600 lines saved)
   - See: `SHARED_COMPONENTS_MIGRATION_PLAN.md` → Component #1
   - Pattern in: `technical-section.tsx`, `hero-section.tsx`, `pricing-section.tsx`, etc.

2. **FeatureCard** (12+ usages, ~480 lines saved)
   - See: Migration Plan → Component #5
   - Pattern in: `solution-section.tsx`, `problem-section.tsx`, `use-cases-section.tsx`

3. **IconBox** (10+ usages, ~200 lines saved)
   - See: Migration Plan → Component #11
   - Pattern in: Most card components

4. **CheckListItem** (8+ usages, ~160 lines saved)
   - See: Migration Plan → Component #7
   - Pattern in: `pricing-section.tsx`, `comparison-section.tsx`, `enterprise-features.tsx`

5. **PulseBadge** (6+ usages, ~180 lines saved)
   - See: Migration Plan → Component #2
   - Pattern in: `hero-section.tsx`, `developers-hero.tsx`, `email-capture.tsx`

**Deliverable:** 5 primitive components with stories + tests

---

### Phase 2: Visual Primitives (Week 2) ⭐⭐

6. **TerminalWindow** (5+ usages, ~250 lines saved)
7. **CodeBlock** (6+ usages, ~180 lines saved)
8. **ProgressBar** (4+ usages, ~80 lines saved)
9. **BenefitCallout** (6+ usages, ~120 lines saved)

**Deliverable:** 4 primitive components with stories + tests

---

### Phase 3: Specialized Primitives (Week 3) ⭐

10. **TestimonialCard** (3+ usages)
11. **StatCard** (4+ usages)
12. **StepNumber** (4 usages)
13. **TrustIndicator** (5+ usages)
14. **BulletListItem** (5+ usages)
15. **TabButton** (3+ usages)

**Deliverable:** 6 primitive components with stories + tests

---

### Phase 4: Complex Primitives (Week 4) ⭐

16. **AudienceCard** (3 usages, ~300 lines saved)
17. **SecurityCrateCard** (5 usages, ~250 lines saved)
18. **ArchitectureDiagram** (2+ usages, ~150 lines saved)

**Deliverable:** 3 primitive components with stories + tests

---

### Phase 5: New Primitives from Extended Analysis (Week 5) ⭐⭐

19. **NavLink** (20+ usages in navigation.tsx, footer.tsx)
   - Pattern: Reusable navigation link with hover states
   - Found in: `navigation.tsx`, `footer.tsx`

20. **FooterColumn** (4 usages in footer.tsx)
   - Pattern: Footer column with title and links
   - Found in: `footer.tsx`

21. **PricingTier** (3 usages in pricing-tiers.tsx, pricing-section.tsx)
   - Pattern: Pricing card with features list, price, CTA
   - Found in: `pricing-tiers.tsx`, `pricing-section.tsx`

22. **ComparisonTableRow** (10+ usages in pricing-comparison.tsx, comparison-section.tsx)
   - Pattern: Table row with feature name and check/x icons
   - Found in: `pricing-comparison.tsx`, `comparison-section.tsx`

23. **EarningsCard** (2 usages in providers-earnings.tsx, providers-hero.tsx)
   - Pattern: Earnings display with stats and breakdown
   - Found in: `providers-earnings.tsx`, `providers-hero.tsx`

24. **GPUListItem** (6+ usages in providers-earnings.tsx, providers-hero.tsx)
   - Pattern: GPU item with status indicator, name, and earnings
   - Found in: `providers-earnings.tsx`, `providers-hero.tsx`

25. **UseCaseCard** (4 usages in enterprise-use-cases.tsx)
   - Pattern: Use case card with icon, challenge, and solution sections
   - Found in: `enterprise-use-cases.tsx`

**Deliverable:** 7 primitive components with stories + tests

---

### Phase 6: Migration (Week 6-7)

**Now replace all usage across the codebase.**

**Order:**
1. Main sections: `hero-section.tsx`, `features-section.tsx`, `pricing-section.tsx`
2. Developer pages: `components/developers/*.tsx`
3. Enterprise pages: `components/enterprise/*.tsx`
4. Feature pages: `components/features/*.tsx`
5. Provider pages: `components/providers/*.tsx`
6. Pricing pages: `components/pricing/*.tsx`
7. Navigation and footer: `navigation.tsx`, `footer.tsx`
8. Remaining sections

**Process per file:**
1. Import primitives from `@/components/primitives`
2. Replace duplicated code with primitive components
3. Verify visual appearance (compare before/after screenshots)
4. Run tests
5. Delete old code
6. Commit with message: `refactor(commercial): migrate [component-name] to primitives`

---

### Phase 7: Validation (Week 8)

1. ✅ Visual regression testing (compare screenshots)
2. ✅ Accessibility audit (run axe-core)
3. ✅ Performance benchmarks (Lighthouse)
4. ✅ Code review
5. ✅ Update component library docs

---

## How to Implement a Primitive Component

### Example: `FeatureCard`

**Step 1: Create the component**

```tsx
// /components/primitives/cards/FeatureCard.tsx
import type { LucideIcon } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface FeatureCardProps {
  /** Lucide icon component */
  icon: LucideIcon
  /** Card title */
  title: string
  /** Card description */
  description: string
  /** Icon background color (Tailwind class) */
  iconColor?: string
  /** Enable hover effect */
  hover?: boolean
  /** Card size variant */
  size?: 'sm' | 'md' | 'lg'
  /** Additional CSS classes */
  className?: string
}

export function FeatureCard({
  icon: Icon,
  title,
  description,
  iconColor = 'primary',
  hover = false,
  size = 'md',
  className,
}: FeatureCardProps) {
  const sizeClasses = {
    sm: 'p-4 space-y-2',
    md: 'p-6 space-y-3',
    lg: 'p-8 space-y-4',
  }

  return (
    <div
      className={cn(
        'bg-card border border-border rounded-lg',
        sizeClasses[size],
        hover && 'transition-all hover:border-primary/50 hover:bg-card/80',
        className
      )}
    >
      <div className={cn(
        'rounded-lg flex items-center justify-center',
        size === 'sm' ? 'h-8 w-8' : size === 'md' ? 'h-10 w-10' : 'h-12 w-12',
        `bg-${iconColor}/10`
      )}>
        <Icon className={cn(
          size === 'sm' ? 'h-4 w-4' : size === 'md' ? 'h-5 w-5' : 'h-6 w-6',
          `text-${iconColor}`
        )} />
      </div>
      <h3 className={cn(
        'font-bold text-card-foreground',
        size === 'sm' ? 'text-base' : size === 'md' ? 'text-lg' : 'text-xl'
      )}>
        {title}
      </h3>
      <p className={cn(
        'text-muted-foreground leading-relaxed',
        size === 'sm' ? 'text-xs' : 'text-sm'
      )}>
        {description}
      </p>
    </div>
  )
}
```

**Step 2: Create Storybook story**

```tsx
// /components/primitives/cards/FeatureCard.stories.tsx
import type { Meta, StoryObj } from '@storybook/react'
import { DollarSign, Shield, Zap } from 'lucide-react'
import { FeatureCard } from './FeatureCard'

const meta: Meta<typeof FeatureCard> = {
  title: 'Primitives/Cards/FeatureCard',
  component: FeatureCard,
  tags: ['autodocs'],
  argTypes: {
    icon: {
      control: false,
      description: 'Lucide icon component',
    },
    iconColor: {
      control: 'select',
      options: ['primary', 'chart-2', 'chart-3', 'chart-4'],
    },
    size: {
      control: 'select',
      options: ['sm', 'md', 'lg'],
    },
  },
}

export default meta
type Story = StoryObj<typeof FeatureCard>

export const Default: Story = {
  args: {
    icon: DollarSign,
    title: 'Zero Ongoing Costs',
    description: 'Pay only for electricity. No subscriptions. No per-token fees.',
  },
}

export const WithHover: Story = {
  args: {
    ...Default.args,
    hover: true,
  },
}

export const Small: Story = {
  args: {
    ...Default.args,
    size: 'sm',
  },
}

export const Large: Story = {
  args: {
    icon: Shield,
    title: 'Complete Privacy',
    description: 'Code never leaves your network. GDPR-compliant by default.',
    size: 'lg',
    iconColor: 'chart-3',
  },
}

export const AllVariants: Story = {
  render: () => (
    <div className="grid grid-cols-3 gap-4">
      <FeatureCard
        icon={DollarSign}
        title="Zero Costs"
        description="No subscriptions."
        size="sm"
      />
      <FeatureCard
        icon={Shield}
        title="Complete Privacy"
        description="GDPR-compliant by default."
        iconColor="chart-3"
      />
      <FeatureCard
        icon={Zap}
        title="Never Changes"
        description="Models update only when YOU decide."
        size="lg"
        iconColor="chart-2"
        hover
      />
    </div>
  ),
}
```

**Step 3: Create unit tests**

```tsx
// /components/primitives/cards/FeatureCard.spec.tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { DollarSign } from 'lucide-react'
import { FeatureCard } from './FeatureCard'

describe('FeatureCard', () => {
  it('renders title and description', () => {
    render(
      <FeatureCard
        icon={DollarSign}
        title="Test Title"
        description="Test description"
      />
    )
    
    expect(screen.getByText('Test Title')).toBeInTheDocument()
    expect(screen.getByText('Test description')).toBeInTheDocument()
  })

  it('renders icon', () => {
    const { container } = render(
      <FeatureCard
        icon={DollarSign}
        title="Test"
        description="Test"
      />
    )
    
    expect(container.querySelector('svg')).toBeInTheDocument()
  })

  it('applies size classes correctly', () => {
    const { container, rerender } = render(
      <FeatureCard
        icon={DollarSign}
        title="Test"
        description="Test"
        size="sm"
      />
    )
    
    expect(container.firstChild).toHaveClass('p-4')
    
    rerender(
      <FeatureCard
        icon={DollarSign}
        title="Test"
        description="Test"
        size="lg"
      />
    )
    
    expect(container.firstChild).toHaveClass('p-8')
  })

  it('applies hover class when hover prop is true', () => {
    const { container } = render(
      <FeatureCard
        icon={DollarSign}
        title="Test"
        description="Test"
        hover
      />
    )
    
    expect(container.firstChild).toHaveClass('hover:border-primary/50')
  })
})
```

**Step 4: Export from barrel**

```tsx
// /components/primitives/index.ts
export { FeatureCard, type FeatureCardProps } from './cards/FeatureCard'
// ... other exports
```

**Step 5: Migrate usage**

```tsx
// Before (in solution-section.tsx):
<div className="bg-card border border-border rounded-lg p-6 space-y-3">
  <div className="h-10 w-10 rounded-lg bg-chart-3/10 flex items-center justify-center">
    <DollarSign className="h-5 w-5 text-chart-3" />
  </div>
  <h3 className="text-lg font-bold text-card-foreground">Zero Ongoing Costs</h3>
  <p className="text-muted-foreground text-sm leading-relaxed">
    Pay only for electricity. No subscriptions. No per-token fees.
  </p>
</div>

// After:
import { FeatureCard } from '@/components/primitives'

<FeatureCard
  icon={DollarSign}
  title="Zero Ongoing Costs"
  description="Pay only for electricity. No subscriptions. No per-token fees."
  iconColor="chart-3"
/>
```

---

## Files You'll Touch

### Create (25 components × 3 files each = 75 new files)

```
/components/primitives/
├── layout/
│   ├── SectionContainer.tsx
│   ├── SectionContainer.stories.tsx
│   └── SectionContainer.spec.tsx
├── badges/
│   ├── PulseBadge.tsx
│   ├── PulseBadge.stories.tsx
│   └── PulseBadge.spec.tsx
├── cards/
│   ├── FeatureCard.tsx
│   ├── FeatureCard.stories.tsx
│   ├── FeatureCard.spec.tsx
│   ├── TestimonialCard.tsx
│   ├── TestimonialCard.stories.tsx
│   ├── TestimonialCard.spec.tsx
│   ├── AudienceCard.tsx
│   ├── AudienceCard.stories.tsx
│   ├── AudienceCard.spec.tsx
│   ├── SecurityCrateCard.tsx
│   ├── SecurityCrateCard.stories.tsx
│   └── SecurityCrateCard.spec.tsx
... (continue for all 18 components)
└── index.ts
```

### Modify (60+ existing component files)

All files in:
- `/components/*.tsx` (main sections)
- `/components/developers/*.tsx` (10 files)
- `/components/enterprise/*.tsx` (11 files)
- `/components/features/*.tsx` (9 files)
- `/components/providers/*.tsx` (11 files)
- `/components/pricing/*.tsx` (4 files)
- `navigation.tsx`, `footer.tsx`

**See:** `SHARED_COMPONENTS_MIGRATION_PLAN.md` → Appendix for full list

---

## Testing Checklist (Per Component)

Before marking a component "done":

- [ ] Component file created with TypeScript types
- [ ] JSDoc comments on all props
- [ ] Storybook story with all variants
- [ ] Unit tests with 90%+ coverage
- [ ] Accessibility: proper ARIA labels, keyboard navigation
- [ ] Responsive: works on mobile, tablet, desktop
- [ ] Dark mode: uses theme tokens correctly
- [ ] Exported from barrel (`index.ts`)
- [ ] Used in at least 3 places (migrate usage)
- [ ] Old duplicated code deleted
- [ ] Visual regression: screenshots match before/after

---

## Quality Gates

**DO NOT PROCEED to next phase until:**

✅ All components in current phase have:
- Storybook stories (viewable at `http://localhost:6006`)
- Unit tests passing (`pnpm test`)
- 90%+ test coverage
- No TypeScript errors
- No ESLint warnings

✅ Code review approved  
✅ Visual regression tests pass

---

## Common Pitfalls (AVOID THESE)

### ❌ DON'T: Change designs while refactoring
```tsx
// BAD: Adding new features
<FeatureCard 
  icon={DollarSign}
  title="Zero Costs"
  description="..."
  newFeature={true}  // ❌ NO! This is a refactor, not a redesign
/>
```

### ❌ DON'T: Skip tests
```tsx
// BAD: No tests
// "I'll add tests later" = Technical debt
```

### ❌ DON'T: Use hardcoded colors
```tsx
// BAD: Hardcoded colors
<div className="bg-blue-500">  // ❌ Use theme tokens

// GOOD: Theme tokens
<div className="bg-primary">  // ✅
```

### ❌ DON'T: Make components too specific
```tsx
// BAD: Too specific
<DeveloperFeatureCard />  // ❌ Only works for developers page

// GOOD: Generic and reusable
<FeatureCard />  // ✅ Works everywhere
```

### ❌ DON'T: Skip Storybook stories
```tsx
// BAD: No story
// "I'll document it later" = Nobody knows how to use it
```

### ✅ DO: Keep it simple
```tsx
// GOOD: Simple, reusable, well-typed
<FeatureCard
  icon={DollarSign}
  title="Zero Costs"
  description="No subscriptions."
  iconColor="primary"
  size="md"
  hover
/>
```

---

## Success Criteria

**You're done when:**

1. ✅ All 25 primitive components created with stories + tests
2. ✅ All 60+ section files migrated to use primitives
3. ✅ Zero visual regressions (screenshots match)
4. ✅ Test coverage ≥90% for primitives
5. ✅ No TypeScript errors
6. ✅ No ESLint warnings
7. ✅ Storybook builds successfully
8. ✅ Bundle size unchanged or smaller
9. ✅ All old duplicated code deleted
10. ✅ Code review approved

---

## Resources

### Documentation
- **Migration Plan:** `.migration-tokens/SHARED_COMPONENTS_MIGRATION_PLAN.md`
- **Component Patterns:** See migration plan → each component has full pattern + props
- **Usage Matrix:** See migration plan → Appendix

### Tools
- **Storybook:** `pnpm run storybook` (port 6006)
- **Tests:** `pnpm test` (Vitest)
- **Coverage:** `pnpm test:coverage`
- **Type Check:** `pnpm tsc --noEmit`
- **Lint:** `pnpm lint`

### Code Style
- Use `cn()` from `@/lib/utils` for className merging
- Use `cva` (class-variance-authority) for complex variants
- Follow existing shadcn/ui patterns in `/components/ui/`
- Use Tailwind design tokens (no hardcoded colors)

---

## Questions?

**Before starting:**
1. Read `SHARED_COMPONENTS_MIGRATION_PLAN.md` in full
2. Review existing `/components/ui/` components for patterns
3. Set up Storybook and Vitest
4. Create a test branch: `git checkout -b feat/primitives-migration`

**During implementation:**
- If a component pattern is unclear, check the migration plan
- If you find more duplicated patterns, add them to the plan
- If you need to change a design, STOP and discuss first

**When stuck:**
- Check existing shadcn/ui components for reference
- Look at Radix UI docs for accessibility patterns
- Review Tailwind CSS docs for design tokens

---

## Commit Strategy

**Branch:** `feat/primitives-migration`

**Commit messages:**
```bash
# Phase 1-4: Creating primitives
feat(primitives): add SectionContainer component
feat(primitives): add FeatureCard component
test(primitives): add tests for PulseBadge

# Phase 5: Migration
refactor(commercial): migrate hero-section to primitives
refactor(commercial): migrate features-section to primitives
refactor(commercial): migrate developers pages to primitives

# Phase 6: Cleanup
chore(commercial): remove duplicated component code
docs(primitives): update component library documentation
```

**PR Title:** `feat(commercial): extract shared components into primitives`

**PR Description Template:**
```markdown
## Summary
Extracted 25 duplicated component patterns into reusable primitives.

## Impact
- 🎯 Code reduction: ~4,800 lines (~50%)
- 🧪 Test coverage: 90%+
- 📚 Storybook stories: 25 components documented
- 🎨 Visual changes: None (pure refactor)

## Checklist
- [x] All 25 primitives created
- [x] All 60+ files migrated
- [x] Storybook stories added
- [x] Unit tests added (90%+ coverage)
- [x] Visual regression tests pass
- [x] No TypeScript errors
- [x] No ESLint warnings
- [x] Old code deleted

## Screenshots
[Before/After comparison screenshots]
```

---

## Timeline

**Total:** 7-8 weeks (1 developer, full-time)

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Phase 1 | 5 high-priority primitives |
| 2 | Phase 2 | 4 visual primitives |
| 3 | Phase 3 | 6 specialized primitives |
| 4 | Phase 4 | 3 complex primitives |
| 5 | Phase 5 | 7 new primitives (extended analysis) |
| 6-7 | Phase 6 | All files migrated |
| 8 | Phase 7 | Validation & docs |

---

## Final Notes

**This is a high-impact refactor.** Take your time. Quality over speed.

**Follow the phases in order.** Don't skip ahead. Each phase builds on the previous.

**Test everything.** Visual regressions are easy to introduce. Compare screenshots before/after.

**Ask questions early.** If something is unclear, ask before implementing.

**Document as you go.** Future developers will thank you.

---

**Good luck! 🚀**

---

**Handoff Complete.**  
**Next Team: Start with Phase 1, Component #1 (SectionContainer).**

---

## Appendix: New Components from Extended Analysis

### 19. NavLink
**Pattern:**
```tsx
<Link
  href="/features"
  className="text-muted-foreground hover:text-foreground transition-colors"
>
  Features
</Link>
```

**Props:**
- `href: string`
- `children: ReactNode`
- `variant?: 'default' | 'mobile'`
- `onClick?: () => void`

**Found in:** `navigation.tsx` (20+ instances), `footer.tsx` (16+ instances)

---

### 20. FooterColumn
**Pattern:**
```tsx
<div>
  <h3 className="text-foreground font-bold mb-4">Product</h3>
  <ul className="space-y-2 text-sm">
    {links.map(link => (
      <li><a href={link.href}>{link.text}</a></li>
    ))}
  </ul>
</div>
```

**Props:**
- `title: string`
- `links: Array<{ href: string; text: string; external?: boolean }>`

**Found in:** `footer.tsx` (4 instances)

---

### 21. PricingTier
**Pattern:**
```tsx
<div className="bg-card border-2 border-border rounded-lg p-8 space-y-6">
  <div>
    <h3 className="text-2xl font-bold text-foreground">{title}</h3>
    <div className="mt-4">
      <span className="text-4xl font-bold text-foreground">{price}</span>
      <span className="text-muted-foreground ml-2">{period}</span>
    </div>
  </div>
  <ul className="space-y-3">
    {features.map(feature => (
      <li className="flex items-start gap-2">
        <Check className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
        <span className="text-muted-foreground">{feature}</span>
      </li>
    ))}
  </ul>
  <Button>{ctaText}</Button>
</div>
```

**Props:**
- `title: string`
- `price: string | number`
- `period?: string`
- `features: string[]`
- `ctaText: string`
- `ctaVariant?: 'default' | 'outline'`
- `highlighted?: boolean`
- `badge?: string`

**Found in:** `pricing-tiers.tsx` (3 instances), `pricing-section.tsx` (3 instances)

---

### 22. ComparisonTableRow
**Pattern:**
```tsx
<tr className="border-b border-border">
  <td className="p-4 text-muted-foreground">{feature}</td>
  <td className="text-center p-4">
    <Check className="h-5 w-5 text-chart-3 mx-auto" />
  </td>
  <td className="text-center p-4">
    <X className="h-5 w-5 text-muted-foreground/30 mx-auto" />
  </td>
</tr>
```

**Props:**
- `feature: string`
- `values: Array<boolean | string | ReactNode>`
- `highlightColumn?: number`

**Found in:** `pricing-comparison.tsx` (10+ instances), `comparison-section.tsx` (6 instances)

---

### 23. EarningsCard
**Pattern:**
```tsx
<div className="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
  <h3 className="mb-6 text-xl font-bold text-foreground">Your Potential Earnings</h3>
  <div className="rounded-xl border border-primary/20 bg-primary/10 p-6">
    <div className="mb-2 text-sm text-primary">Monthly Earnings</div>
    <div className="text-5xl font-bold text-foreground">€{amount}</div>
    <div className="mt-2 text-sm text-muted-foreground">{subtitle}</div>
  </div>
  <div className="grid gap-4 sm:grid-cols-2">
    {stats.map(stat => (
      <div className="rounded-lg border border-border bg-background/50 p-4">
        <div className="mb-1 text-sm text-muted-foreground">{stat.label}</div>
        <div className="text-2xl font-bold text-foreground">{stat.value}</div>
      </div>
    ))}
  </div>
</div>
```

**Props:**
- `title: string`
- `amount: number | string`
- `subtitle?: string`
- `stats: Array<{ label: string; value: string | number }>`
- `breakdown?: Array<{ label: string; value: string | number }>`

**Found in:** `providers-earnings.tsx`, `providers-hero.tsx`

---

### 24. GPUListItem
**Pattern:**
```tsx
<div className="flex items-center justify-between rounded-lg border border-border bg-background/50 p-3">
  <div className="flex items-center gap-3">
    <div className="h-2 w-2 rounded-full bg-chart-3" />
    <div>
      <div className="text-sm font-medium text-foreground">{name}</div>
      <div className="text-xs text-muted-foreground">{subtitle}</div>
    </div>
  </div>
  <div className="text-right">
    <div className="text-sm font-medium text-foreground">{value}</div>
    <div className="text-xs text-muted-foreground">{label}</div>
  </div>
</div>
```

**Props:**
- `name: string`
- `subtitle?: string`
- `value: string | number`
- `label?: string`
- `status?: 'active' | 'idle' | 'offline'`
- `statusColor?: string`

**Found in:** `providers-earnings.tsx` (6 instances), `providers-hero.tsx` (2 instances)

---

### 25. UseCaseCard
**Pattern:**
```tsx
<div className="rounded-lg border border-border bg-card p-8">
  <div className="mb-4 flex items-center gap-3">
    <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
      <Icon className="h-6 w-6 text-primary" />
    </div>
    <div>
      <h3 className="text-xl font-bold text-foreground">{title}</h3>
      <p className="text-sm text-muted-foreground">{subtitle}</p>
    </div>
  </div>
  
  <p className="mb-4 leading-relaxed text-muted-foreground">{description}</p>
  
  <div className="mb-4 rounded-lg border border-border bg-background p-4">
    <div className="mb-2 font-semibold text-foreground">Challenge:</div>
    <ul className="space-y-1 text-sm text-muted-foreground">
      {challenges.map(challenge => <li>• {challenge}</li>)}
    </ul>
  </div>
  
  <div className="rounded-lg border border-chart-3/50 bg-chart-3/10 p-4">
    <div className="mb-2 font-semibold text-chart-3">Solution with rbee:</div>
    <ul className="space-y-1 text-sm text-muted-foreground">
      {solutions.map(solution => <li>• {solution}</li>)}
    </ul>
  </div>
</div>
```

**Props:**
- `icon: LucideIcon`
- `title: string`
- `subtitle: string`
- `description: string`
- `challenges: string[]`
- `solutions: string[]`

**Found in:** `enterprise-use-cases.tsx` (4 instances)

---

**End of Appendix**
