# Work Unit 05: Problem Section

**Priority:** MEDIUM  
**Component:** `/components/problem-section.tsx`

---

## Current Hardcoded Colors

| Line | Current Class | Semantic Purpose |
|------|---------------|------------------|
| 5 | `bg-gradient-to-br from-red-950/20 via-slate-900 to-slate-900` | Section background gradient |
| 8 | `text-white` | Section heading |
| 15 | `bg-slate-900/50` | Card background |
| 15 | `border-red-900/30` | Card border |
| 15 | `hover:border-red-700/50` | Card hover border |
| 16 | `bg-red-500/10` | Icon background |
| 17 | `text-red-400` | Icon color |
| 19 | `text-white` | Card heading |
| 20 | `text-slate-300` | Card description |
| 27 | `bg-slate-900/50` | Card background |
| 27 | `border-red-900/30` | Card border |
| 27 | `hover:border-red-700/50` | Card hover border |
| 28 | `bg-red-500/10` | Icon background |
| 29 | `text-red-400` | Icon color |
| 31 | `text-white` | Card heading |
| 32 | `text-slate-300` | Card description |
| 39 | `bg-slate-900/50` | Card background |
| 39 | `border-red-900/30` | Card border |
| 39 | `hover:border-red-700/50` | Card hover border |
| 40 | `bg-red-500/10` | Icon background |
| 41 | `text-red-400` | Icon color |
| 43 | `text-white` | Card heading |
| 44 | `text-slate-300` | Card description |
| 52 | `text-slate-300` | Footer text |

---

## Proposed Token Replacements

### Section Background

```tsx
// Line 5: Section background
// DECISION NEEDED: Keep gradient or use solid background?

// Option 1: Solid background with destructive accent
className="py-24 bg-background"

// Option 2: Keep dark gradient (always dark, regardless of theme)
className="py-24 bg-gradient-to-br from-red-950/20 via-slate-900 to-slate-900"

// Option 3: Use destructive token with gradient
className="py-24 bg-gradient-to-br from-destructive/5 via-background to-background"
```

### Typography

```tsx
// Lines 8, 19, 31, 43: Headings
className="text-foreground"

// Lines 20, 32, 44, 52: Body text
className="text-muted-foreground"
```

### Problem Cards

```tsx
// Lines 15, 27, 39: Card container
className="bg-card/50 border border-destructive/30 rounded-lg p-8 space-y-4 hover:border-destructive/50 transition-colors"

// Lines 16, 28, 40: Icon background
className="h-12 w-12 rounded-lg bg-destructive/10 flex items-center justify-center"

// Lines 17, 29, 41: Icon color
className="h-6 w-6 text-destructive"
```

---

## New Tokens Required

**None** - The `--destructive` token already exists and is perfect for this section.

Current values:
- Light mode: `--destructive: #ef4444` (red-500)
- Dark mode: `--destructive: #ef4444` (red-500)

---

## Implementation Notes

### Semantic Use of Destructive Token

This section highlights **problems** and **risks**, making it semantically appropriate to use the `--destructive` token (typically used for errors, warnings, and negative states).

### Background Gradient Decision

The section currently uses a dark gradient with red tint to create a sense of urgency/danger. Options:

1. **Keep dark gradient:** Maintains the emotional impact, always dark regardless of theme
2. **Use destructive token:** More theme-adaptive, uses `destructive/5` for subtle red tint
3. **Solid background:** Simplest, fully theme-adaptive

**Recommendation:** Keep the dark gradient (Option 2) to maintain the "problem" atmosphere, or use destructive token gradient (Option 3) for theme adaptability.

### Card Hover States

The hover state changes border opacity from `30%` to `50%`. This pattern works well with tokens:
- Default: `border-destructive/30`
- Hover: `hover:border-destructive/50`

### Icon Backgrounds

Icon backgrounds use `bg-red-500/10` which translates to `bg-destructive/10` perfectly.

---

## Verification Checklist

- [ ] Section background works in both themes
- [ ] Problem cards are readable in both themes
- [ ] Destructive (red) color is visible and appropriate
- [ ] Card hover states work
- [ ] Icon backgrounds have good contrast
- [ ] Icons are visible
- [ ] Headings are readable
- [ ] Body text is readable
- [ ] Footer text is readable
- [ ] No hardcoded `slate-*`, `red-*` classes remain (except gradient if kept)

---

## Estimated Complexity

**Low-Medium** - Straightforward replacements, main decision is gradient handling. The destructive token is already perfect for this use case.

---

## Design Considerations

### Emotional Impact

This section is designed to create a sense of urgency and highlight problems. The red color scheme is intentional. Ensure:

1. Red/destructive color is prominent but not overwhelming
2. Cards stand out from the background
3. The section feels distinct from positive sections (features, solutions)

### Accessibility

Ensure sufficient contrast for:
- Red icons against card backgrounds
- White/foreground text against dark backgrounds
- Card borders are visible but not harsh
