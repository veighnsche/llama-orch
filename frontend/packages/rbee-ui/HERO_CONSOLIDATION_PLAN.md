# Hero Template Consolidation Plan

## Executive Summary

After analyzing all 7 Hero components, there's significant structural inconsistency in the **left-side messaging content**. The right-side "aside" components are appropriately unique per page, but the left side needs standardization.

---

## Current State Analysis

### Left Side (Messaging) Components

| Hero | Badge | Headline Structure | Subcopy | Bullets/Stats | CTAs | Trust Elements |
|------|-------|-------------------|---------|---------------|------|----------------|
| **HomeHero** | PulseBadge | Prefix + Highlight (2 lines) | ✅ | BulletListItem (3x) | 2 buttons | Trust badges (3x) |
| **EnterpriseHero** | Badge + Icon | Single string | ✅ | StatsGrid (tiles, 3 cols) | 2 buttons | ComplianceChips |
| **DevelopersHero** | Badge + Pulse | Two-line lockup | ✅ ReactNode | Trust badges (inline) | 2 buttons + tertiary link | Trust badges |
| **PricingHero** | Badge | ReactNode (flexible) | ✅ | Assurance items (2 cols) | 2 buttons | N/A |
| **FeaturesHero** | None | Prefix + Gradient span | ✅ | Micro-badges (4x) | 2 buttons | Stat strip (bottom) |
| **ProvidersHero** | Badge + Icon | Single string | ✅ | StatsGrid (pills, 3 cols) | 2 buttons | Trust line text |
| **UseCasesHero** | Badge (custom) | Heading + Highlight | ✅ | Proof indicators | 2 buttons | Proof indicators |

### Key Inconsistencies

1. **Badge Variants**: PulseBadge, Badge + Icon, Badge + Pulse, plain Badge, none
2. **Headline Structure**: Some use `prefix + highlight`, others use single strings, some use ReactNode
3. **Proof Elements**: BulletListItem, StatsGrid (tiles), StatsGrid (pills), micro-badges, trust badges, proof indicators, compliance chips
4. **Layout Grid**: Some use `lg:grid-cols-12`, others `lg:grid-cols-2`, others `lg:grid-cols-[1.1fr_0.9fr]`
5. **Animation Classes**: Inconsistent use of `animate-in`, `fade-in`, `slide-in-from-*`
6. **Spacing**: Different `space-y` values, different padding strategies

---

## Proposed Solution: Unified HeroTemplate

### Architecture

Create a new **`HeroTemplate`** component that handles all left-side messaging with a consistent structure:

```
┌─────────────────────────────────────────────────────────────┐
│ HeroTemplate                                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LEFT (Messaging - Standardized)    RIGHT (Aside - Unique) │
│  ┌──────────────────────────────┐   ┌──────────────────┐   │
│  │ 1. Badge (optional)          │   │                  │   │
│  │ 2. Headline (flexible)       │   │  Page-specific   │   │
│  │ 3. Subcopy                   │   │  visual/aside    │   │
│  │ 4. Proof Elements (variant)  │   │  component       │   │
│  │ 5. CTA Buttons (1-3)         │   │  (slot)          │   │
│  │ 6. Trust Elements (optional) │   │                  │   │
│  └──────────────────────────────┘   └──────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Specification

### 1. Badge Component (Top)

**Variants to support:**
- `pulse` - Animated pulse dot + text (HomeHero)
- `icon` - Icon + text (EnterpriseHero, ProvidersHero)
- `simple` - Plain badge (PricingHero, UseCasesHero)
- `none` - No badge (FeaturesHero)

**Props:**
```typescript
badge?: {
  variant: 'pulse' | 'icon' | 'simple' | 'none'
  text: string
  icon?: ReactNode
  showPulse?: boolean
}
```

---

### 2. Headline Component

**Variants to support:**
- `two-line-highlight` - Prefix on line 1, highlight on line 2 (HomeHero, FeaturesHero)
- `inline-highlight` - Single line with inline gradient span (UseCasesHero)
- `custom` - Full ReactNode control (PricingHero, DevelopersHero)
- `simple` - Plain string (EnterpriseHero, ProvidersHero)

**Props:**
```typescript
headline: {
  variant: 'two-line-highlight' | 'inline-highlight' | 'custom' | 'simple'
  content: string | ReactNode
  highlight?: string // For two-line or inline variants
}
```

---

### 3. Subcopy

**Standard:**
- Always a `<p>` tag
- Always `text-xl` or `text-lg`
- Always `text-muted-foreground`
- Always `leading-relaxed` or `leading-8`
- Max width constraint (e.g., `max-w-[58ch]` or `max-w-2xl`)

**Props:**
```typescript
subcopy: string | ReactNode
subcopyMaxWidth?: 'narrow' | 'medium' | 'wide' // Maps to max-w-[58ch], max-w-2xl, max-w-prose
```

---

### 4. Proof Elements

**Variants to support:**
- `bullets` - BulletListItem components (HomeHero)
- `stats-tiles` - StatsGrid with tiles variant (EnterpriseHero)
- `stats-pills` - StatsGrid with pills variant (ProvidersHero)
- `badges` - Inline Badge components (FeaturesHero, DevelopersHero)
- `indicators` - Simple text with optional dots (UseCasesHero)
- `none` - No proof elements (PricingHero)

**Props:**
```typescript
proofElements?: {
  variant: 'bullets' | 'stats-tiles' | 'stats-pills' | 'badges' | 'indicators' | 'none'
  items: Array<{
    // For bullets
    title?: string
    variant?: 'check' | 'dot' | 'arrow'
    color?: string
    
    // For stats
    icon?: ReactNode
    value?: string
    label?: string
    
    // For badges/indicators
    text?: string
    hasDot?: boolean
  }>
  columns?: 2 | 3 // For stats grids
}
```

---

### 5. CTA Buttons

**Standard:**
- Always 1-3 buttons
- Primary button always first
- Secondary button always second
- Optional tertiary link (DevelopersHero)
- Consistent sizing: `size="lg"`, `h-14` or `h-12`
- Consistent gap: `gap-4` or `gap-3`

**Props:**
```typescript
ctas: {
  primary: {
    label: string
    href?: string
    onClick?: () => void
    showIcon?: boolean
    dataUmamiEvent?: string
  }
  secondary: {
    label: string
    href?: string
    variant?: 'outline' | 'secondary'
  }
  tertiary?: {
    label: string
    href: string
    mobileOnly?: boolean
  }
}
```

---

### 6. Trust Elements (Bottom)

**Variants to support:**
- `badges` - Trust badges with icons/links (HomeHero)
- `chips` - ComplianceChip components (EnterpriseHero)
- `text` - Simple text line (ProvidersHero)
- `stat-strip` - Bottom stat strip (FeaturesHero)
- `none` - No trust elements

**Props:**
```typescript
trustElements?: {
  variant: 'badges' | 'chips' | 'text' | 'stat-strip' | 'none'
  items?: Array<{
    type?: 'github' | 'api' | 'cost' | 'compliance'
    label: string
    href?: string
    icon?: ReactNode
    ariaLabel?: string
  }>
  text?: string
  statStrip?: {
    items: Array<{ label: string; value: string }>
  }
}
```

---

### 7. Layout & Grid

**Standardize to:**
- Container: `container mx-auto px-4`
- Grid: `grid lg:grid-cols-12 gap-12 items-center`
- Left column: `lg:col-span-6` or `lg:col-span-7` (configurable)
- Right column: `lg:col-span-6` or `lg:col-span-5` (configurable)
- Vertical spacing: `space-y-8` (consistent)

**Props:**
```typescript
layout?: {
  leftCols?: 6 | 7
  rightCols?: 5 | 6
  gap?: 8 | 10 | 12
  verticalSpacing?: 6 | 8 | 10
}
```

---

### 8. Background & Styling

**Standardize to:**
- Section wrapper: `relative isolate overflow-hidden`
- Background: `bg-gradient-to-b from-background to-card` (default)
- Optional: HoneycombPattern, radial gradients
- Padding: `py-24 lg:py-28` or `py-20 lg:py-24`

**Props:**
```typescript
background?: {
  variant: 'gradient' | 'radial' | 'honeycomb' | 'custom'
  customClassName?: string
}
padding?: 'default' | 'compact' | 'spacious'
```

---

### 9. Animations

**Standardize to:**
- Consistent animation classes: `animate-in fade-in duration-500`
- Staggered delays: `delay-100`, `delay-150`, `delay-200`, etc.
- Slide directions: `slide-in-from-bottom-2`, `slide-in-from-left-4`, etc.

**Props:**
```typescript
animations?: {
  enabled?: boolean
  stagger?: boolean
  direction?: 'bottom' | 'left' | 'right'
}
```

---

## Migration Strategy

### Phase 1: Create HeroTemplate Component

1. Create `/frontend/packages/rbee-ui/src/templates/HeroTemplate/HeroTemplate.tsx`
2. Implement all variants and props from spec above
3. Create comprehensive Storybook stories for all variants
4. Write unit tests for prop combinations

### Phase 2: Migrate Existing Heroes (One at a Time)

**Order of migration (easiest to hardest):**

1. **PricingHero** (simplest structure)
   - Badge: simple
   - Headline: custom
   - Proof: assurance items (can map to indicators)
   - Trust: none

2. **UseCasesHero** (similar to Pricing)
   - Badge: simple
   - Headline: inline-highlight
   - Proof: indicators
   - Trust: none

3. **FeaturesHero** (no badge, stat strip)
   - Badge: none
   - Headline: two-line-highlight
   - Proof: badges
   - Trust: stat-strip

4. **ProvidersHero** (stats pills)
   - Badge: icon
   - Headline: simple
   - Proof: stats-pills
   - Trust: text

5. **DevelopersHero** (tertiary link)
   - Badge: pulse
   - Headline: custom
   - Proof: badges
   - Trust: badges
   - Special: tertiary link

6. **EnterpriseHero** (compliance chips)
   - Badge: icon
   - Headline: simple
   - Proof: stats-tiles
   - Trust: chips

7. **HomeHero** (most complex)
   - Badge: pulse
   - Headline: two-line-highlight
   - Proof: bullets
   - Trust: badges

### Phase 3: Cleanup

1. Delete old Hero template files
2. Update all page imports to use new HeroTemplate
3. Update Storybook stories
4. Update documentation

---

## File Structure

```
frontend/packages/rbee-ui/src/templates/
├── HeroTemplate/
│   ├── HeroTemplate.tsx          # Main component
│   ├── HeroTemplate.stories.tsx  # Storybook stories
│   ├── HeroTemplate.spec.tsx     # Unit tests
│   ├── HeroTemplateProps.tsx     # Type definitions
│   ├── index.ts                  # Exports
│   └── README.md                 # Component documentation
└── [OLD HEROES TO DELETE AFTER MIGRATION]
    ├── HomeHero/
    ├── EnterpriseHero/
    ├── DevelopersHero/
    ├── PricingHero/
    ├── FeaturesHero/
    ├── ProvidersHero/
    └── UseCasesHero/
```

---

## Benefits

1. **Consistency**: All heroes have the same left-side structure
2. **Maintainability**: Single source of truth for hero messaging patterns
3. **Flexibility**: Right-side "aside" remains unique per page
4. **DRY**: Eliminates ~1000 lines of duplicated code
5. **Type Safety**: Centralized TypeScript types
6. **Testing**: Single test suite covers all variants
7. **Documentation**: One Storybook story shows all patterns

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing pages | Migrate one hero at a time, test thoroughly |
| Over-abstraction | Keep right-side as flexible slot, don't force unification |
| Type complexity | Use discriminated unions for variant-specific props |
| Performance | Use React.memo for HeroTemplate, lazy load heavy components |

---

## Acceptance Criteria

- [ ] HeroTemplate component created with all variants
- [ ] All 7 heroes migrated to use HeroTemplate
- [ ] Storybook stories show all variants
- [ ] Unit tests cover all prop combinations
- [ ] Visual regression tests pass
- [ ] No breaking changes to page-level APIs
- [ ] Documentation updated
- [ ] Old hero files deleted

---

## Estimated Effort

- **Phase 1** (Create HeroTemplate): 8-12 hours
- **Phase 2** (Migrate 7 heroes): 10-14 hours (1.5-2h per hero)
- **Phase 3** (Cleanup): 2-3 hours
- **Total**: 20-29 hours (~3-4 days)

---

## Next Steps

1. **Review this plan** with team
2. **Approve** the HeroTemplate API design
3. **Create** feature branch: `feat/hero-template-consolidation`
4. **Implement** Phase 1
5. **Migrate** heroes one by one (separate commits)
6. **Cleanup** and merge

---

## Questions to Resolve

1. Should we support custom grid layouts per hero, or enforce standard 6/6 or 7/5 split?
2. Should animations be configurable or always enabled?
3. Should we extract badge variants into separate Badge components?
4. Should we support multiple proof element types in a single hero?
5. Should trust elements always be at the bottom, or allow positioning?

---

**Status**: ✅ COMPLETE - All 6 heroes migrated  
**Owner**: Cascade AI  
**Created**: 2025-10-17  
**Last Updated**: 2025-10-17  
**Completion Date**: 2025-10-17
