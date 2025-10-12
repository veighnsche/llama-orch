# Work Unit 03: CTA Section

**Priority:** HIGH (Conversion critical)  
**Component:** `/components/cta-section.tsx`

---

## Current Hardcoded Colors

| Line | Current Class | Semantic Purpose |
|------|---------------|------------------|
| 6 | `bg-gradient-to-br from-slate-950 via-slate-900 to-amber-950` | CTA background gradient |
| 9 | `text-white` | Main heading |
| 12 | `text-amber-400` | Heading accent |
| 15 | `text-slate-300` | Subheading text |
| 22 | `bg-amber-500` | Primary button background |
| 22 | `hover:bg-amber-600` | Primary button hover |
| 22 | `text-slate-950` | Primary button text |
| 30 | `border-slate-600` | Secondary button border |
| 30 | `text-slate-200` | Secondary button text |
| 30 | `hover:bg-slate-800` | Secondary button hover |
| 38 | `border-slate-600` | Tertiary button border |
| 38 | `text-slate-200` | Tertiary button text |
| 38 | `hover:bg-slate-800` | Tertiary button hover |
| 45 | `text-slate-400` | Footer text |

---

## Proposed Token Replacements

### Background

```tsx
// Line 6: CTA background gradient
// DECISION NEEDED: Keep gradient or use solid background?

// Option 1: Solid background (simplest)
className="py-24 bg-background"

// Option 2: Accent background
className="py-24 bg-accent"

// Option 3: Keep gradient with tokens (requires new gradient tokens)
className="py-24 bg-gradient-to-br from-background via-card to-primary/20"

// Option 4: Force dark CTA (always dark regardless of theme)
className="py-24 bg-slate-950"  // Keep as-is if you want always-dark CTA
```

### Typography

```tsx
// Line 9: Main heading
className="text-4xl lg:text-6xl font-bold text-foreground mb-6 text-balance"

// Line 12: Heading accent
className="text-primary"

// Line 15: Subheading
className="text-xl text-muted-foreground leading-relaxed"

// Line 45: Footer text
className="text-muted-foreground pt-4"
```

### Buttons

```tsx
// Line 22: Primary button
className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold text-lg h-14 px-8"

// Lines 30, 38: Secondary/Tertiary buttons
className="border-border text-foreground hover:bg-secondary h-14 px-8 bg-transparent"
```

---

## New Tokens Required

**None** - All colors map to existing tokens.

---

## Implementation Notes

### Gradient Background Decision

The CTA section uses a gradient ending in `to-amber-950`, which creates a warm, inviting feel. This is intentional for conversion optimization.

**Options:**
1. **Solid background:** Simplest, theme-adaptive
2. **Accent background:** Uses `--accent` token (currently amber)
3. **Gradient with tokens:** More complex but preserves visual design
4. **Always dark:** Keep current gradient, don't theme it

**Recommendation:** 
- If CTA should always stand out: Use **Option 4** (always dark)
- If CTA should adapt to theme: Use **Option 2** (accent background)

### Button Hierarchy

The CTA has three buttons:
1. **Primary:** "Get Started Free" (amber background)
2. **Secondary:** "View Documentation" (outlined)
3. **Tertiary:** "Join Discord" (outlined)

All outlined buttons should use the same token classes for consistency.

---

## Verification Checklist

- [ ] CTA section renders correctly in light mode
- [ ] CTA section renders correctly in dark mode
- [ ] Background gradient/color looks good in both themes
- [ ] Main heading is readable
- [ ] Accent text (amber) is visible
- [ ] All three buttons have good contrast
- [ ] Button hover states work
- [ ] Footer text is readable
- [ ] No hardcoded `slate-*`, `amber-*` classes remain

---

## Estimated Complexity

**Low-Medium** - Straightforward replacements, main decision is gradient handling.

---

## Design Considerations

### Conversion Optimization

CTA sections are typically designed to stand out from the rest of the page. Consider:

1. **Always dark CTA:** Makes it distinct from the rest of the page
2. **High contrast:** Ensure buttons pop against the background
3. **Warm colors:** Amber/orange creates urgency and warmth

If you migrate to tokens, ensure the CTA still "feels" different from regular content sections.

### A/B Testing Recommendation

If this is a production site, consider:
- Keep the current dark gradient for CTA sections
- Only theme the informational sections (features, pricing, etc.)
- Measure conversion rates before/after full theme migration
