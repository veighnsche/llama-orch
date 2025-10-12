# Design Tokens Guide

<!-- Created by: TEAM-FE-010 -->

## Overview

All design tokens are defined in `tokens-base.css` and automatically adapt between light and dark modes.

## Token Categories

### Background & Foreground
- `bg-background` / `text-foreground` - Base page colors
- `bg-card` / `text-card-foreground` - Card backgrounds
- `bg-popover` / `text-popover-foreground` - Popover/dropdown backgrounds

### Interactive Elements
- `bg-primary` / `text-primary-foreground` - Primary actions (buttons, links)
- `bg-secondary` / `text-secondary-foreground` - Secondary elements
- `bg-accent` / `text-accent-foreground` - Accent highlights

### Emphasis & States
- `bg-muted` / `text-muted-foreground` - Muted/subtle elements
- `bg-destructive` / `text-destructive-foreground` - Errors/warnings

### Borders & Inputs
- `border-border` - Border colors
- `bg-input` / `border-input` - Input field borders
- `ring-ring` - Focus rings

### ⭐ NEW: Highlight Tokens (TEAM-FE-010)

**Purpose:** For emphasized backgrounds like pricing cards, comparison tables, and featured sections.

**Problem Solved:** The bright orange (`#f59e0b`) used for primary/accent is too intense as a background in dark mode. It creates visual overwhelm and poor contrast.

#### Light Mode
```css
--highlight: #f59e0b;           /* Bright orange */
--highlight-foreground: #ffffff; /* White text */
```

#### Dark Mode
```css
--highlight: #78350f;           /* Dark brown/amber (muted) */
--highlight-foreground: #fbbf24; /* Lighter amber text */
```

**Usage:**
```vue
<template>
  <!-- Pricing card with highlighted background -->
  <div class="bg-highlight text-highlight-foreground">
    <h3>Most Popular</h3>
    <p>€99/month</p>
  </div>
  
  <!-- Comparison table highlighted column -->
  <td class="bg-highlight text-highlight-foreground">
    <span>rbee</span>
  </td>
</template>
```

**When to use:**
- ✅ Pricing cards (especially "Most Popular" tier)
- ✅ Comparison table highlighted columns
- ✅ Featured sections that need emphasis
- ✅ Call-to-action backgrounds

**When NOT to use:**
- ❌ Buttons (use `bg-primary` instead)
- ❌ Links (use `text-primary` instead)
- ❌ Small badges (use `bg-accent` instead)
- ❌ Regular cards (use `bg-card` instead)

## Color Values Reference

### Light Mode Colors
| Token | Value | Usage |
|-------|-------|-------|
| `--background` | `#ffffff` | Page background |
| `--foreground` | `#0f172a` | Text color |
| `--primary` | `#f59e0b` | Buttons, links |
| `--highlight` | `#f59e0b` | Emphasized backgrounds |
| `--highlight-foreground` | `#ffffff` | Text on highlight |
| `--muted` | `#f1f5f9` | Subtle backgrounds |
| `--border` | `#e2e8f0` | Borders |

### Dark Mode Colors
| Token | Value | Usage |
|-------|-------|-------|
| `--background` | `#0f172a` | Page background |
| `--foreground` | `#f1f5f9` | Text color |
| `--primary` | `#f59e0b` | Buttons, links (stays bright) |
| `--highlight` | `#78350f` | Emphasized backgrounds (muted!) |
| `--highlight-foreground` | `#fbbf24` | Text on highlight (lighter amber) |
| `--muted` | `#1e293b` | Subtle backgrounds |
| `--border` | `#334155` | Borders |

## Migration Guide

### Before (Too Bright in Dark Mode)
```vue
<template>
  <!-- ❌ BAD: Overwhelming orange in dark mode -->
  <div class="bg-primary text-primary-foreground">
    <h3>Most Popular</h3>
    <p>€99/month</p>
  </div>
</template>
```

### After (Balanced in Dark Mode)
```vue
<template>
  <!-- ✅ GOOD: Muted in dark mode, bright in light mode -->
  <div class="bg-highlight text-highlight-foreground">
    <h3>Most Popular</h3>
    <p>€99/month</p>
  </div>
</template>
```

## Visual Comparison

### Light Mode
- `bg-primary`: Bright orange `#f59e0b` - Perfect for buttons
- `bg-highlight`: Bright orange `#f59e0b` - Perfect for card backgrounds

### Dark Mode
- `bg-primary`: Bright orange `#f59e0b` - Still bright (good for buttons)
- `bg-highlight`: Dark brown `#78350f` - Muted (good for backgrounds)

## Implementation Notes

1. **Automatic switching:** Tokens automatically adapt when `.dark` class is added to `<html>`
2. **No manual dark: classes needed:** Just use `bg-highlight` and it works in both modes
3. **Consistent with design system:** Follows same pattern as other tokens
4. **Accessible contrast:** Dark mode values tested for WCAG AA compliance

## Examples in Codebase

### Pricing Cards
```vue
<!-- /frontend/libs/storybook/stories/molecules/PricingCard/PricingCard.vue -->
<Card :class="highlighted ? 'bg-highlight text-highlight-foreground' : 'bg-card'">
  <!-- Card content -->
</Card>
```

### Comparison Tables
```vue
<!-- Highlighted column in comparison table -->
<td class="bg-highlight text-highlight-foreground">
  <div class="font-semibold">rbee</div>
  <div class="text-sm">Full control</div>
</td>
```

## Testing

To test the new tokens:

1. **Light mode:**
   ```bash
   # Start storybook
   cd /frontend/libs/storybook
   pnpm story:dev
   # Navigate to component using bg-highlight
   # Verify bright orange background
   ```

2. **Dark mode:**
   ```bash
   # Click theme toggle in navigation
   # Verify muted brown background (not overwhelming)
   # Verify lighter amber text (good contrast)
   ```

## Future Considerations

If you need additional highlight variations:
- `--highlight-subtle` - Even more muted for less emphasis
- `--highlight-intense` - Brighter for maximum emphasis (use sparingly)

For now, the single `--highlight` token should cover 95% of use cases.
