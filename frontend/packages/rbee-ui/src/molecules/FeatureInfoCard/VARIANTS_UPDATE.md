# FeatureInfoCard Variants Update

## Changes Made

Added two new capabilities to `FeatureInfoCard`:

### 1. Neutral Tone Variant
Added `neutral` tone for clean, minimal backgrounds in solution sections.

**Styling:**
- Background: `bg-background` (neutral, no gradient)
- Border: `border-border`
- Hover: `hover:bg-muted/30`

**Usage:**
```tsx
<FeatureInfoCard
  icon={DollarSign}
  title="Zero ongoing costs"
  body="Pay only for electricity."
  tone="neutral"
  size="sm"
/>
```

### 2. Size Variant for Body Text
Added `size` prop to control body text size.

**Options:**
- `sm` (default) - `text-sm` - Compact, for dense information
- `base` - `text-base` - Larger, more readable for emphasis

**Usage:**
```tsx
// Small text (solutions section)
<FeatureInfoCard size="sm" />

// Larger text (problems section)
<FeatureInfoCard size="base" />
```

## Updated Components

### HomeSolutionSection
Benefits now use:
- `tone="neutral"` - Clean background
- `size="sm"` - Compact text

```tsx
<FeatureInfoCard
  icon={benefit.icon}
  title={benefit.title}
  body={benefit.body}
  tone="neutral"
  size="sm"
/>
```

### ProblemSection
Problems now use:
- `tone="destructive"` (unchanged)
- `size="base"` - **Larger, more readable text**

```tsx
<FeatureInfoCard
  icon={item.icon}
  title={item.title}
  body={item.body}
  tone={item.tone || 'destructive'}
  size="base"
  tag={item.tag}
/>
```

## All Tone Variants

| Tone | Background | Use Case |
|------|------------|----------|
| `default` | Card background | General features |
| `neutral` | **Neutral background** | **Solution sections** |
| `primary` | Primary gradient | Positive features |
| `destructive` | Destructive gradient | Problems/warnings |
| `muted` | Muted gradient | Secondary info |

## All Size Variants

| Size | Text Size | Use Case |
|------|-----------|----------|
| `sm` | `text-sm` | Solutions, benefits, compact info |
| `base` | `text-base` | Problems, emphasis, readability |

## Visual Comparison

### Solutions Section (neutral + sm)
```
┌─────────────────────────────────┐
│ 💰                              │  ← Neutral background
│ Zero ongoing costs              │  ← text-sm body
│ Pay only for electricity.       │
└─────────────────────────────────┘
```

### Problems Section (destructive + base)
```
┌─────────────────────────────────┐
│ 🔒                              │  ← Destructive gradient
│ The provider shuts down         │  ← text-base body (larger)
│ APIs get deprecated. Your       │
│ AI-built code becomes...        │
│ [Loss €2,400/mo]                │
└─────────────────────────────────┘
```

## Storybook Stories

Updated stories to showcase new variants:
- ✅ `Neutral` - New neutral tone example
- ✅ `LargeText` - New size="base" example
- ✅ `SolutionsGrid` - Renamed from Grid, uses neutral + sm
- ✅ `ProblemsGrid` - Updated to use size="base"

## CVA Implementation

```tsx
// New neutral tone in featureInfoCardVariants
const featureInfoCardVariants = cva('...', {
  variants: {
    tone: {
      neutral: 'border-border bg-background hover:bg-muted/30 hover:border-primary/50',
      // ... other tones
    },
  },
})

// New bodyVariants for text sizing
const bodyVariants = cva('text-balance leading-relaxed text-muted-foreground', {
  variants: {
    size: {
      sm: 'text-sm',
      base: 'text-base',
    },
  },
  defaultVariants: {
    size: 'sm',
  },
})
```

## Files Updated

- ✅ `FeatureInfoCard.tsx` - Added neutral tone + size variants
- ✅ `FeatureInfoCard.stories.tsx` - Updated all stories
- ✅ `HomeSolutionSection.tsx` - Uses neutral + sm
- ✅ `ProblemSection.tsx` - Uses size="base"

## Result

**Solutions section:** Clean neutral background with compact text  
**Problems section:** Destructive gradient with larger, more readable text

Both sections now have distinct visual hierarchy and appropriate emphasis.
