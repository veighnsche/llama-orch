# Molecule Usage Guide

Quick reference for using the new consolidated molecules in the commercial frontend.

---

## StatsGrid

**Purpose**: Display statistics in various layouts (pills, tiles, cards, inline)

**Import**:
```tsx
import { StatsGrid } from '@/components/molecules/StatsGrid/StatsGrid'
```

**Usage**:
```tsx
<StatsGrid
  variant="pills"  // or 'tiles' | 'cards' | 'inline'
  columns={3}      // 2 | 3 | 4
  stats={[
    {
      icon: <DollarSign className="h-4 w-4 text-primary" />,
      value: '€50–200',
      label: 'per GPU / month',
      helpText: 'Optional accessibility text'
    },
    // ... more stats
  ]}
/>
```

**Variants**:
- `pills` - Horizontal cards with icon + value + label (used in ProvidersHero)
- `tiles` - Vertical tiles with larger text (similar to StatTile)
- `cards` - Simple centered cards
- `inline` - Compact inline cards with centered icon (used in ProvidersCTA)

---

## IconPlate

**Purpose**: Consistent icon containers with customizable size, tone, and shape

**Import**:
```tsx
import { IconPlate } from '@/components/molecules/IconPlate/IconPlate'
```

**Usage**:
```tsx
<IconPlate
  icon={<Shield className="h-6 w-6" />}
  size="lg"        // 'sm' | 'md' | 'lg'
  tone="primary"   // 'primary' | 'muted' | 'success' | 'warning'
  shape="square"   // 'square' | 'circle'
  className="mb-4" // Optional additional classes
/>
```

**Sizes**:
- `sm` - 8x8 (h-8 w-8) with 3.5x3.5 icon
- `md` - 9x9 (h-9 w-9) with 4x4 icon
- `lg` - 12x12 (h-12 w-12) with 5x5 icon

**Tones**:
- `primary` - bg-primary/10 text-primary
- `muted` - bg-muted text-muted-foreground
- `success` - bg-emerald-500/10 text-emerald-500
- `warning` - bg-amber-500/10 text-amber-500

**When to use**:
- Replace inline `<div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10">`
- Standardize icon containers across molecules
- Ensure consistent sizing and spacing

---

## StatInfoCard

**Purpose**: Stat card with centered icon, value, and label (extracted from ProvidersCTA)

**Import**:
```tsx
import { StatInfoCard } from '@/components/molecules/StatInfoCard/StatInfoCard'
```

**Usage**:
```tsx
<StatInfoCard
  icon={<Clock className="h-4 w-4 text-primary" />}
  value="< 15 minutes"
  label="Setup time"
/>
```

**Note**: This is now redundant with `StatsGrid` variant="inline". Use `StatsGrid` for new code.

---

## Gradient Utilities

**Purpose**: Consistent background gradients across sections

**Available classes**:
```css
.bg-radial-glow              /* Radial gradient from top */
.bg-section-gradient         /* Vertical gradient background → card */
.bg-section-gradient-primary /* Vertical gradient with primary accent */
```

**Usage**:
```tsx
// Before
<section className="bg-[radial-gradient(60rem_40rem_at_50%_-10%,theme(colors.primary/7),transparent)]">

// After
<section className="bg-radial-glow">
```

**Migrated sections**:
- SolutionSection
- EnterpriseCompliance
- EnterpriseSecurity
- EnterpriseFeatures
- EnterpriseCTA

**Remaining** (can be migrated):
- ProvidersHero
- DevelopersHero
- EnterpriseHero
- CtaSection

---

## Migration Patterns

### Icon Container → IconPlate

**Before**:
```tsx
<div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
  <Icon className="h-6 w-6 text-primary" />
</div>
```

**After**:
```tsx
<IconPlate icon={<Icon className="h-6 w-6" />} size="lg" tone="primary" />
```

---

### Stat Pills → StatsGrid

**Before**:
```tsx
<div className="grid gap-3 sm:grid-cols-3">
  <div className="rounded-lg border border-border/70 bg-background/60 p-4">
    <div className="flex items-center gap-2.5">
      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10">
        <DollarSign className="h-4 w-4 text-primary" />
      </div>
      <div>
        <div className="text-xl font-bold">€50–200</div>
        <div className="text-xs text-muted-foreground">per GPU / month</div>
      </div>
    </div>
  </div>
  {/* ... more pills */}
</div>
```

**After**:
```tsx
<StatsGrid
  variant="pills"
  columns={3}
  stats={[
    {
      icon: <DollarSign className="h-4 w-4 text-primary" />,
      value: '€50–200',
      label: 'per GPU / month',
    },
    // ... more stats
  ]}
/>
```

---

### Inline Radial Gradient → Utility Class

**Before**:
```tsx
<section className="bg-[radial-gradient(60rem_40rem_at_50%_-10%,theme(colors.primary/7),transparent)]">
```

**After**:
```tsx
<section className="bg-radial-glow">
```

---

## Best Practices

1. **Always use IconPlate** for icon containers instead of inline divs
2. **Use StatsGrid** for any stat display with 2+ items
3. **Use gradient utilities** instead of inline gradient definitions
4. **Keep molecules small** - don't add too many props
5. **Document variants** when adding new molecule variants

---

## Examples in Codebase

### IconPlate Examples
- `UseCasesSection.tsx` - md size, primary tone
- `HomeSolutionSection.tsx` - lg size, primary tone
- `PledgeCallout.tsx` - md size, circle shape, custom colors
- `SecurityCrate.tsx` - lg size, conditional tone (primary/muted)

### StatsGrid Examples
- `providers-hero.tsx` - pills variant, 3 columns
- `providers-cta.tsx` - inline variant, 3 columns

### Gradient Utility Examples
- `SolutionSection.tsx` - bg-radial-glow
- `enterprise-compliance.tsx` - bg-radial-glow
- `enterprise-security.tsx` - bg-radial-glow

---

## Quick Decision Tree

**Need to display an icon in a container?**
→ Use `IconPlate`

**Need to display 2+ statistics?**
→ Use `StatsGrid`

**Need a radial gradient background?**
→ Use `.bg-radial-glow`

**Need a vertical gradient background?**
→ Use `.bg-section-gradient` or `.bg-section-gradient-primary`

**Building a new molecule with an icon?**
→ Use `IconPlate` internally (see `StatsGrid`, `StatInfoCard` for examples)

---

**Last Updated**: 2025-10-13  
**Status**: Active - use these molecules for all new code
