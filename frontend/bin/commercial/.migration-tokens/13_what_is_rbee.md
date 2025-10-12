# Work Unit 13: What Is Rbee

**Priority:** LOW  
**Component:** `/components/what-is-rbee.tsx`

---

## Current Hardcoded Colors

| Line | Current Class | Semantic Purpose |
|------|---------------|------------------|
| 3 | `bg-slate-50` | Section background |
| 6 | `text-slate-900` | Section heading |
| 8 | `text-slate-700` | Body text |
| 9 | `text-slate-900` | Emphasized text (rbee name) |
| 10 | `text-amber-600` | Emphasized text (platform description) |
| 15, 20, 25 | `bg-white` | Stat card background |
| 15, 20, 25 | `border-slate-200` | Stat card border |
| 16, 21, 26 | `text-amber-600` | Stat number |
| 17, 22, 27 | `text-slate-600` | Stat description |
| 31 | `text-slate-600` | Footer text |

---

## Proposed Token Replacements

```tsx
// Line 3: Section background
className="py-16 bg-secondary"

// Line 6: Section heading
className="text-3xl md:text-4xl font-bold text-foreground"

// Line 8: Body text
className="text-xl text-foreground leading-relaxed"

// Line 9: Emphasized text (rbee name)
className="font-semibold text-card-foreground"

// Line 10: Emphasized text (platform description)
className="font-semibold text-primary"

// Lines 15, 20, 25: Stat cards
className="bg-card p-6 rounded-lg border border-border"

// Lines 16, 21, 26: Stat numbers
className="text-4xl font-bold text-primary mb-2"

// Lines 17, 22, 27: Stat descriptions
className="text-sm text-muted-foreground"

// Line 31: Footer text
className="text-lg text-muted-foreground pt-4"
```

---

## New Tokens Required

**None** - All colors map to existing tokens.

---

## Implementation Notes

### Stat Cards

The three stat cards highlight key benefits:
1. **$0:** Monthly costs
2. **100%:** Privacy
3. **All:** GPUs working together

All use the primary (amber) color for the numbers to maintain brand consistency.

### Emphasized Text

The component emphasizes:
- **"rbee":** The product name
- **"open-source AI orchestration platform":** The value proposition

Both should use prominent colors to stand out from body text.

---

## Verification Checklist

- [ ] Section background works in both themes
- [ ] Section heading is readable
- [ ] Body text is readable
- [ ] Emphasized text stands out
- [ ] Stat cards are readable in both themes
- [ ] Stat numbers are prominent
- [ ] Stat descriptions are readable
- [ ] Footer text is readable
- [ ] No hardcoded `slate-*`, `amber-*` classes remain

---

## Estimated Complexity

**Low** - Simple component with straightforward replacements.

---

## Design Considerations

### Stat Card Prominence

The stat cards are a key visual element. Ensure:
- Numbers are large and prominent (primary color)
- Cards stand out from the background
- Descriptions are readable but de-emphasized

### Brand Consistency

The amber/primary color should be used consistently for:
- Product name emphasis
- Platform description
- Stat numbers

This reinforces brand identity throughout the page.
