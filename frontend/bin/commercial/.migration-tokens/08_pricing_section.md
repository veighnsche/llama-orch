# Work Unit 08: Pricing Section

**Priority:** MEDIUM  
**Component:** `/components/pricing-section.tsx`

---

## Current Hardcoded Colors

| Line | Current Class | Semantic Purpose |
|------|---------------|------------------|
| 6 | `bg-white` | Section background |
| 9 | `text-slate-900` | Section heading |
| 16 | `bg-white` | Tier 1 card background |
| 16 | `border-slate-200` | Tier 1 card border |
| 18 | `text-slate-900` | Tier heading |
| 20 | `text-slate-900` | Price |
| 21 | `text-slate-600` | Price subtext |
| 27, 31, 35, 39, 43 | `text-green-600` | Check icon |
| 28, 32, 36, 40, 44 | `text-slate-600` | Feature text |
| 52 | `text-slate-600` | Card footer text |
| 56 | `bg-amber-50` | Tier 2 card background (highlighted) |
| 56 | `border-amber-500` | Tier 2 card border |
| 58 | `bg-amber-500` | "Most Popular" badge background |
| 58 | `text-white` | "Most Popular" badge text |
| 62 | `text-slate-900` | Tier heading |
| 64 | `text-slate-900` | Price |
| 65 | `text-slate-600` | Price subtext |
| 67 | `text-slate-600` | Price detail |
| 72, 76, 80, 84, 88 | `text-green-600` | Check icon |
| 73 | `text-slate-900` | Feature text (emphasized) |
| 77, 81, 85, 89 | `text-slate-600` | Feature text |
| 93 | `bg-amber-500` | Primary button background |
| 93 | `hover:bg-amber-600` | Primary button hover |
| 93 | `text-slate-950` | Primary button text |
| 95 | `text-slate-600` | Card footer text |
| 99 | `bg-white` | Tier 3 card background |
| 99 | `border-slate-200` | Tier 3 card border |
| 101 | `text-slate-900` | Tier heading |
| 103 | `text-slate-900` | Price |
| 105 | `text-slate-600` | Price subtext |
| 110, 114, 118, 122, 126 | `text-green-600` | Check icon |
| 111 | `text-slate-900` | Feature text (emphasized) |
| 115, 119, 123, 127 | `text-slate-600` | Feature text |
| 135 | `text-slate-600` | Card footer text |
| 139 | `text-slate-600` | Section footer text |

---

## Proposed Token Replacements

### Section Layout

```tsx
// Line 6: Section background
className="py-24 bg-background"

// Line 9: Section heading
className="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance"

// Line 139: Section footer
className="text-center text-muted-foreground mt-12 max-w-2xl mx-auto"
```

### Tier 1: Free (Standard Card)

```tsx
// Line 16: Card container
className="bg-card border-2 border-border rounded-lg p-8 space-y-6"

// Lines 18, 20: Headings and price
className="text-2xl font-bold text-card-foreground"
className="text-4xl font-bold text-card-foreground"

// Lines 21, 28, 32, 36, 40, 44, 52: Subtext and features
className="text-muted-foreground"

// Lines 27, 31, 35, 39, 43: Check icons
className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5"

// Line 48: Button
className="w-full bg-transparent" variant="outline"
```

### Tier 2: Team (Highlighted Card)

```tsx
// Line 56: Card container (highlighted)
className="bg-primary/5 border-2 border-primary rounded-lg p-8 space-y-6 relative"

// Line 58: "Most Popular" badge
className="bg-primary text-primary-foreground px-4 py-1 rounded-full text-sm font-medium"

// Lines 62, 64: Headings and price
className="text-2xl font-bold text-card-foreground"
className="text-4xl font-bold text-card-foreground"

// Lines 65, 67, 77, 81, 85, 89, 95: Subtext and features
className="text-muted-foreground"

// Line 73: Emphasized feature
className="text-card-foreground font-medium"

// Lines 72, 76, 80, 84, 88: Check icons
className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5"

// Line 93: Primary button
className="w-full bg-primary hover:bg-primary/90 text-primary-foreground"
```

### Tier 3: Enterprise (Standard Card)

```tsx
// Line 99: Card container
className="bg-card border-2 border-border rounded-lg p-8 space-y-6"

// Lines 101, 103: Headings and price
className="text-2xl font-bold text-card-foreground"
className="text-4xl font-bold text-card-foreground"

// Lines 105, 115, 119, 123, 127, 135: Subtext and features
className="text-muted-foreground"

// Line 111: Emphasized feature
className="text-card-foreground font-medium"

// Lines 110, 114, 118, 122, 126: Check icons
className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5"

// Line 131: Button
className="w-full bg-transparent" variant="outline"
```

---

## New Tokens Required

**None** - All colors map to existing tokens.

---

## Implementation Notes

### Tier 2 Highlighting

The "Team" tier (most popular) is highlighted with:
- Background: `bg-primary/5` (subtle amber tint)
- Border: `border-primary` (amber border)
- Badge: `bg-primary text-primary-foreground`

This creates a clear visual hierarchy showing the recommended tier.

### Check Icons

All check icons use green (`text-chart-3`) to indicate included features. This is consistent across all tiers.

### Button Variants

- **Tier 1 (Free):** Outline button (secondary)
- **Tier 2 (Team):** Solid primary button (call-to-action)
- **Tier 3 (Enterprise):** Outline button (secondary)

This hierarchy guides users toward the Team tier.

### Feature Text Emphasis

Some features are emphasized with `font-medium` and darker text:
- Tier 2: "Everything in Home/Lab"
- Tier 3: "Everything in Team"

These should use `text-card-foreground` instead of `text-muted-foreground`.

---

## Verification Checklist

- [ ] Section background works in both themes
- [ ] All three pricing cards are readable in both themes
- [ ] Tier 2 (Team) is clearly highlighted
- [ ] "Most Popular" badge is visible
- [ ] Check icons (green) are visible in all tiers
- [ ] Price text is prominent
- [ ] Feature text is readable
- [ ] Emphasized features stand out
- [ ] Buttons have good contrast
- [ ] Card borders are visible
- [ ] Section footer is readable
- [ ] No hardcoded `slate-*`, `amber-*`, `green-*` classes remain

---

## Estimated Complexity

**Low-Medium** - Repetitive structure across three tiers, but straightforward replacements. Main challenge is maintaining the visual hierarchy.

---

## Design Considerations

### Pricing Psychology

The pricing section uses visual hierarchy to guide users:
1. **Tier 2 is highlighted:** Amber background and border
2. **"Most Popular" badge:** Creates social proof
3. **Primary button on Tier 2:** Strongest call-to-action

Ensure these elements remain prominent after token migration.

### Feature List Consistency

All tiers use the same structure:
- Green check icon
- Feature text
- Consistent spacing

This makes comparison easy for users.

### Accessibility

Ensure:
- Sufficient contrast for all text
- Check icons are visible
- Badge text is readable
- Button text has good contrast
