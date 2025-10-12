# Work Unit 06: Solution Section

**Priority:** MEDIUM  
**Component:** `/components/solution-section.tsx`

---

## Current Hardcoded Colors

| Line | Current Class | Semantic Purpose |
|------|---------------|------------------|
| 5 | `bg-slate-50` | Section background |
| 8 | `text-slate-900` | Section heading |
| 9 | `text-amber-600` | Heading accent |
| 11 | `text-slate-600` | Section description |
| 19 | `bg-white` | Diagram container background |
| 19 | `border-slate-200` | Diagram container border |
| 21 | `bg-amber-100` | Badge background |
| 21 | `text-amber-900` | Badge text |
| 29 | `bg-amber-500` | Queen node background |
| 29 | `text-white` | Queen node text |
| 32 | `bg-slate-300` | Connector line |
| 37 | `bg-amber-100` | Hive manager background |
| 37 | `text-amber-900` | Hive manager text |
| 37 | `border-amber-200` | Hive manager border |
| 49 | `bg-slate-300` | Connector line |
| 56 | `bg-slate-100` | Worker background |
| 56 | `text-slate-700` | Worker text |
| 56 | `border-slate-200` | Worker border |
| 75 | `bg-white` | Benefit card background |
| 75 | `border-slate-200` | Benefit card border |
| 76 | `bg-green-100` | Icon background (success) |
| 77 | `text-green-600` | Icon color (success) |
| 79 | `text-slate-900` | Card heading |
| 80 | `text-slate-600` | Card description |
| 85 | `bg-white` | Benefit card background |
| 85 | `border-slate-200` | Benefit card border |
| 86 | `bg-blue-100` | Icon background (info) |
| 87 | `text-blue-600` | Icon color (info) |
| 89 | `text-slate-900` | Card heading |
| 90 | `text-slate-600` | Card description |
| 95 | `bg-white` | Benefit card background |
| 95 | `border-slate-200` | Benefit card border |
| 96 | `bg-amber-100` | Icon background (primary) |
| 97 | `text-amber-600` | Icon color (primary) |
| 99 | `text-slate-900` | Card heading |
| 100 | `text-slate-600` | Card description |
| 105 | `bg-white` | Benefit card background |
| 105 | `border-slate-200` | Benefit card border |
| 106 | `bg-slate-100` | Icon background (neutral) |
| 107 | `text-slate-600` | Icon color (neutral) |
| 109 | `text-slate-900` | Card heading |
| 110 | `text-slate-600` | Card description |

---

## Proposed Token Replacements

### Section Layout

```tsx
// Line 5: Section background
className="py-24 bg-secondary"

// Line 8: Section heading
className="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance"

// Line 9: Heading accent
className="text-primary"

// Line 11: Section description
className="text-xl text-muted-foreground leading-relaxed text-pretty"
```

### Architecture Diagram

```tsx
// Line 19: Diagram container
className="bg-card border border-border rounded-lg p-8 shadow-lg"

// Line 21: Badge
className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 rounded-full text-foreground text-sm font-medium"

// Line 29: Queen node
className="bg-primary text-primary-foreground px-6 py-3 rounded-lg font-bold text-lg shadow-md"

// Line 32, 49: Connector lines
className="h-8 w-0.5 bg-border my-2"

// Line 37: Hive manager nodes
className="bg-primary/10 text-foreground px-4 py-2 rounded-lg font-medium text-sm border border-primary/20"

// Line 56: Worker nodes
className="bg-muted text-foreground px-3 py-2 rounded text-xs font-medium border border-border text-center"
```

### Benefit Cards

```tsx
// Lines 75, 85, 95, 105: Card containers
className="bg-card border border-border rounded-lg p-6 space-y-3"

// Lines 79, 89, 99, 109: Card headings
className="text-lg font-bold text-card-foreground"

// Lines 80, 90, 100, 110: Card descriptions
className="text-muted-foreground text-sm leading-relaxed"
```

### Icon Backgrounds (Semantic)

```tsx
// Line 76: Success icon background
className="h-10 w-10 rounded-lg bg-chart-3/10 flex items-center justify-center"

// Line 77: Success icon color
className="h-5 w-5 text-chart-3 stroke-[3]"

// Line 86: Info icon background
className="h-10 w-10 rounded-lg bg-chart-2/10 flex items-center justify-center"

// Line 87: Info icon color
className="h-5 w-5 text-chart-2"

// Line 96: Primary icon background
className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center"

// Line 97: Primary icon color
className="h-5 w-5 text-primary"

// Line 106: Neutral icon background
className="h-10 w-10 rounded-lg bg-muted flex items-center justify-center"

// Line 107: Neutral icon color
className="h-5 w-5 text-muted-foreground"
```

---

## New Tokens Required

**Optional:** If you want more semantic benefit icons:

```css
/* Add to globals.css */
:root {
  --success: #10b981;           /* green-500 */
  --info: #3b82f6;              /* blue-500 */
}

.dark {
  --success: #22c55e;           /* green-400 */
  --info: #60a5fa;              /* blue-400 */
}

@theme inline {
  --color-success: var(--success);
  --color-info: var(--info);
}
```

Then use:
```tsx
// Success icon
className="bg-success/10"
className="text-success"

// Info icon
className="bg-info/10"
className="text-info"
```

**However**, using `chart-3` (green) and `chart-2` (blue) works perfectly fine without new tokens.

---

## Implementation Notes

### Architecture Diagram Semantics

The "Bee Architecture" diagram uses color to convey hierarchy:
- **Queen (Primary):** Amber/Primary color - the orchestrator
- **Hive Managers (Primary/10):** Lighter amber - mid-level
- **Workers (Muted):** Neutral - leaf nodes

This hierarchy should be preserved with tokens:
- Queen: `bg-primary`
- Hive Managers: `bg-primary/10`
- Workers: `bg-muted`

### Benefit Card Icons

Each benefit card has a semantic icon:
1. **Zero Costs:** Green (success) - DollarSign icon
2. **Privacy:** Blue (info) - Shield icon
3. **Never Changes:** Amber (primary) - Anchor icon
4. **Use Hardware:** Neutral - Laptop icon

These semantics should be preserved with appropriate tokens.

### Connector Lines

The connector lines use `bg-slate-300` which should map to `bg-border` for theme consistency.

---

## Verification Checklist

- [ ] Section background works in both themes
- [ ] Architecture diagram is readable in both themes
- [ ] Queen node stands out (primary color)
- [ ] Hive managers are distinct from workers
- [ ] Worker nodes are readable
- [ ] Connector lines are visible
- [ ] Benefit cards are readable in both themes
- [ ] Icon backgrounds have good contrast
- [ ] Success (green) icon is distinct
- [ ] Info (blue) icon is distinct
- [ ] Primary (amber) icon is distinct
- [ ] Neutral icon is distinct
- [ ] No hardcoded `slate-*`, `amber-*`, `green-*`, `blue-*` classes remain

---

## Estimated Complexity

**Medium** - Architecture diagram needs careful color hierarchy, benefit cards have semantic icons that must be preserved.

---

## Design Considerations

### Visual Hierarchy

The architecture diagram is a key visual element. Ensure:
1. Queen node is most prominent (primary color)
2. Hive managers are secondary (primary/10)
3. Workers are tertiary (muted)
4. Connector lines are subtle but visible

### Benefit Card Consistency

All benefit cards should have the same structure:
- Card: `bg-card border border-border`
- Icon background: `bg-{semantic}/10`
- Icon: `text-{semantic}`
- Heading: `text-card-foreground`
- Description: `text-muted-foreground`
