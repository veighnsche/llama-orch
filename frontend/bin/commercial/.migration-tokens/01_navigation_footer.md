# Work Unit 01: Navigation & Footer

**Priority:** HIGH (Always visible)  
**Components:**
- `/components/navigation.tsx`
- `/components/footer.tsx`

---

## Navigation Component (`navigation.tsx`)

### Current Hardcoded Colors

| Line | Current Class | Semantic Purpose |
|------|---------------|------------------|
| 12 | `bg-slate-950/95` | Navigation background (dark) |
| 12 | `border-slate-800` | Navigation bottom border |
| 18 | `text-amber-500` | Logo/brand color |
| 18 | `hover:text-amber-400` | Logo hover state |
| 26, 29, 32, 35, 38, 41 | `text-slate-300` | Navigation link text |
| 26, 29, 32, 35, 38, 41 | `hover:text-white` | Navigation link hover |
| 49 | `text-slate-300` | Icon color |
| 49 | `hover:text-white` | Icon hover |
| 54 | `bg-amber-500` | Primary button background |
| 54 | `hover:bg-amber-600` | Primary button hover |
| 54 | `text-slate-950` | Primary button text |
| 62 | `text-slate-300` | Mobile menu button |
| 62 | `hover:text-white` | Mobile menu button hover |
| 70 | `border-slate-800` | Mobile menu border |
| 73, 80, 87, 94, 101, 108, 117 | `text-slate-300` | Mobile link text |
| 73, 80, 87, 94, 101, 108, 117 | `hover:text-white` | Mobile link hover |
| 121 | `bg-amber-500` | Mobile button background |
| 121 | `hover:bg-amber-600` | Mobile button hover |
| 121 | `text-slate-950` | Mobile button text |

### Proposed Token Replacements

```tsx
// Line 12: Navigation bar
className="fixed top-0 left-0 right-0 z-50 bg-background/95 backdrop-blur-sm border-b border-border"

// Line 18: Logo/brand
className="flex items-center gap-2 text-xl font-semibold text-primary hover:text-primary/80 transition-colors"

// Lines 26, 29, 32, 35, 38, 41: Navigation links
className="text-muted-foreground hover:text-foreground transition-colors"

// Line 49: GitHub icon
className="text-muted-foreground hover:text-foreground transition-colors"

// Line 54: Primary button
className="bg-primary hover:bg-primary/90 text-primary-foreground"

// Line 62: Mobile menu button
className="md:hidden text-muted-foreground hover:text-foreground"

// Line 70: Mobile menu border
className="md:hidden py-4 space-y-4 border-t border-border"

// Lines 73, 80, 87, 94, 101, 108, 117: Mobile links
className="block text-muted-foreground hover:text-foreground transition-colors"

// Line 121: Mobile button
className="w-full bg-primary hover:bg-primary/90 text-primary-foreground"
```

### New Tokens Required

**None** - All colors map to existing tokens.

---

## Footer Component (`footer.tsx`)

### Current Hardcoded Colors

| Line | Current Class | Semantic Purpose |
|------|---------------|------------------|
| 5 | `bg-slate-950` | Footer background (dark) |
| 5 | `text-slate-400` | Footer text color |
| 10 | `text-white` | Section heading |
| 17, 27, 37, 47, 60, 69, 75, 81, 92, 97, 102, 111, 124, 129, 138, 147 | `hover:text-amber-400` | Link hover color |
| 158 | `border-slate-800` | Bottom bar border |
| 165, 169, 172 | `hover:text-amber-400` | Social icon hover |

### Proposed Token Replacements

```tsx
// Line 5: Footer background
className="bg-background text-muted-foreground py-16"

// Line 10: Section headings
className="text-foreground font-bold mb-4"

// Lines 17, 27, etc.: Link hover
className="hover:text-primary transition-colors"

// Line 158: Bottom bar border
className="border-t border-border pt-8 flex flex-col md:flex-row justify-between items-center gap-4"

// Lines 165, 169, 172: Social icon hover
className="hover:text-primary transition-colors"
```

### New Tokens Required

**None** - All colors map to existing tokens.

---

## Implementation Notes

### Navigation Considerations

The navigation currently uses `bg-slate-950/95` which is a dark background. In the token system:
- **Light mode:** `--background` is `#ffffff` (white)
- **Dark mode:** `--background` is `#0f172a` (slate-950)

This means the navigation will automatically adapt:
- Light mode: White background with dark text
- Dark mode: Dark background with light text

The `/95` opacity should be preserved: `bg-background/95`

### Footer Considerations

Same as navigation - the footer is currently dark (`bg-slate-950`). Using `bg-background` will make it:
- Light mode: White background
- Dark mode: Dark background

If you want the footer to **always be dark** regardless of theme, you would need to use a different approach (e.g., a separate `--footer-background` token or force dark mode on the footer section).

**Decision needed:** Should the footer always be dark, or should it adapt to the theme?

---

## Verification Checklist

- [ ] Navigation renders correctly in light mode
- [ ] Navigation renders correctly in dark mode
- [ ] Footer renders correctly in light mode
- [ ] Footer renders correctly in dark mode
- [ ] All hover states work
- [ ] Brand color (amber/primary) is consistent
- [ ] No hardcoded `slate-*`, `amber-*`, `white` classes remain
- [ ] Mobile menu works in both themes
- [ ] Backdrop blur still works on navigation
- [ ] Border visibility is good in both themes

---

## Estimated Complexity

**Low-Medium** - Straightforward replacements, but many instances. Main decision is whether navigation/footer should always be dark or theme-adaptive.
