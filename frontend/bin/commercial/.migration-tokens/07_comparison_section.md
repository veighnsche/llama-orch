# Work Unit 07: Comparison Section

**Priority:** MEDIUM  
**Component:** `/components/comparison-section.tsx`

---

## Current Hardcoded Colors

| Line | Current Class | Semantic Purpose |
|------|---------------|------------------|
| 5 | `bg-slate-50` | Section background |
| 8 | `text-slate-900` | Section heading |
| 14 | `bg-white` | Table background |
| 14 | `border-slate-200` | Table border |
| 16 | `border-slate-200` | Table header border |
| 17 | `text-slate-900` | Table header text |
| 18 | `text-amber-600` | rbee column header |
| 18 | `bg-amber-50` | rbee column background |
| 19, 20, 21 | `text-slate-600` | Competitor column headers |
| 25 | `border-slate-200` | Table row border |
| 26 | `text-slate-900` | Row label |
| 27 | `bg-amber-50` | rbee cell background |
| 28 | `text-slate-900` | rbee cell text |
| 29 | `text-slate-600` | rbee cell subtext |
| 31, 32, 33 | `text-slate-600` | Competitor cell text |
| 35 | `border-slate-200` | Table row border |
| 36 | `text-slate-900` | Row label |
| 37 | `bg-amber-50` | rbee cell background |
| 38 | `text-green-600` | Check icon (success) |
| 39 | `text-slate-600` | Cell subtext |
| 42 | `text-red-500` | X icon (negative) |
| 43 | `text-slate-600` | Cell subtext |
| 46 | `text-green-600` | Check icon (success) |
| 47 | `text-slate-600` | Cell subtext |
| 50 | `text-red-500` | X icon (negative) |
| 51 | `text-slate-600` | Cell subtext |
| 54 | `border-slate-200` | Table row border |
| 55 | `text-slate-900` | Row label |
| 56 | `bg-amber-50` | rbee cell background |
| 57 | `text-green-600` | Check icon (success) |
| 58 | `text-slate-600` | Cell subtext |
| 62 | `text-slate-600` | Cell text |
| 65 | `text-green-600` | Check icon (success) |
| 68 | `border-slate-200` | Table row border |
| 69 | `text-slate-900` | Row label |
| 70 | `bg-amber-50` | rbee cell background |
| 71 | `text-green-600` | Check icon (success) |
| 83 | `border-slate-200` | Table row border |
| 84 | `text-slate-900` | Row label |
| 85 | `bg-amber-50` | rbee cell background |
| 86 | `text-green-600` | Check icon (success) |
| 87 | `text-slate-600` | Cell subtext |
| 90 | `text-red-500` | X icon (negative) |
| 93 | `text-red-500` | X icon (negative) |
| 96 | `text-red-500` | X icon (negative) |
| 100 | `text-slate-900` | Row label |
| 101 | `bg-amber-50` | rbee cell background |
| 102 | `text-green-600` | Success text |
| 105 | `text-red-600` | Negative text |
| 108 | `text-green-600` | Success text |
| 111 | `text-red-600` | Negative text |

---

## Proposed Token Replacements

### Section Layout

```tsx
// Line 5: Section background
className="py-24 bg-secondary"

// Line 8: Section heading
className="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance"
```

### Table Structure

```tsx
// Line 14: Table container
className="w-full bg-card border border-border rounded-lg"

// Line 16: Table header row
className="border-b border-border"

// Line 17: Feature column header
className="text-left p-4 font-bold text-card-foreground"

// Line 18: rbee column header (highlighted)
className="p-4 font-bold text-primary bg-primary/5"

// Lines 19, 20, 21: Competitor column headers
className="p-4 font-medium text-muted-foreground"
```

### Table Rows

```tsx
// Lines 25, 35, 54, 68, 83: Row borders
className="border-b border-border"

// Lines 26, 36, 55, 69, 84, 100: Row labels
className="p-4 font-medium text-card-foreground"

// Lines 27, 37, 56, 70, 85, 101: rbee cells (highlighted)
className="p-4 text-center bg-primary/5"

// Lines 28, 102: rbee cell text (positive)
className="text-sm font-medium text-card-foreground"

// Lines 29, 31, 32, 33, 39, 43, 47, 51, 58, 62, 87: Cell subtext
className="text-xs text-muted-foreground"
```

### Icons & Status Indicators

```tsx
// Lines 38, 46, 57, 65, 71, 86: Check icons (success)
className="h-5 w-5 text-chart-3 mx-auto"

// Lines 42, 50, 80, 90, 93, 96: X icons (negative)
className="h-5 w-5 text-destructive mx-auto"

// Lines 102, 108: Success text
className="text-sm font-medium text-chart-3"

// Lines 105, 111: Negative text
className="text-sm text-destructive"
```

---

## New Tokens Required

**None** - All colors map to existing tokens:
- Success/positive: `chart-3` (green)
- Negative/error: `destructive` (red)
- Primary highlight: `primary` (amber)

---

## Implementation Notes

### rbee Column Highlighting

The rbee column is consistently highlighted with:
- Header: `text-primary bg-primary/5`
- Cells: `bg-primary/5`

This creates a subtle but clear visual distinction showing rbee's advantages.

### Status Icons

The comparison uses semantic colors:
- **Green check:** Feature available/positive
- **Red X:** Feature unavailable/negative
- **Text:** Neutral information

These should map to:
- Green: `chart-3` (already green in token system)
- Red: `destructive` (already red in token system)

### Table Accessibility

Ensure the table remains accessible:
- Sufficient contrast for all text
- Clear visual distinction between rbee and competitors
- Icons are supplemented with text where needed

---

## Verification Checklist

- [ ] Section background works in both themes
- [ ] Table is readable in both themes
- [ ] rbee column is clearly highlighted
- [ ] Check icons (green) are visible
- [ ] X icons (red) are visible
- [ ] Row labels are readable
- [ ] Cell text has good contrast
- [ ] Subtext is readable but de-emphasized
- [ ] Table borders are visible
- [ ] No hardcoded `slate-*`, `amber-*`, `green-*`, `red-*` classes remain

---

## Estimated Complexity

**Low-Medium** - Repetitive replacements, but straightforward. Main challenge is ensuring the rbee column remains visually distinct.

---

## Design Considerations

### Competitive Positioning

This table is designed to show rbee's advantages. Ensure:
1. rbee column stands out (primary color highlight)
2. Green checks are prominent for rbee's features
3. Red X's are clear for competitors' limitations
4. The visual hierarchy guides the eye to rbee's benefits

### Mobile Responsiveness

The table uses `overflow-x-auto` for mobile. Ensure:
- Horizontal scrolling works smoothly
- rbee column highlight is visible when scrolling
- Text remains readable at smaller sizes
