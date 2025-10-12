# Work Unit 02: Hero Section

**Priority:** HIGH (First impression)  
**Component:** `/components/hero-section.tsx`

---

## Current Hardcoded Colors

| Line | Current Class | Semantic Purpose |
|------|---------------|------------------|
| 6 | `bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800` | Hero background gradient |
| 11 | `bg-amber-500/10` | Badge background |
| 11 | `border-amber-500/20` | Badge border |
| 11 | `text-amber-400` | Badge text |
| 13 | `bg-amber-400` | Pulse animation background |
| 14 | `bg-amber-500` | Pulse dot background |
| 19 | `text-white` | Main heading |
| 22 | `text-amber-400` | Heading accent |
| 25 | `text-slate-300` | Subheading text |
| 33 | `bg-amber-500` | Primary CTA background |
| 33 | `hover:bg-amber-600` | Primary CTA hover |
| 33 | `text-slate-950` | Primary CTA text |
| 41 | `border-slate-600` | Secondary button border |
| 41 | `text-slate-200` | Secondary button text |
| 41 | `hover:bg-slate-800` | Secondary button hover |
| 49, 53, 57, 63 | `text-slate-400` | Trust indicator text |
| 54 | `fill-amber-500 text-amber-500` | Star icon |
| 58, 64 | `border-slate-600` | Icon borders |
| 74 | `bg-slate-900` | Terminal background |
| 74 | `border-slate-700` | Terminal border |
| 75 | `bg-slate-800` | Terminal header |
| 75 | `border-slate-700` | Terminal header border |
| 77 | `bg-red-500` | Terminal button (red) |
| 78 | `bg-amber-500` | Terminal button (amber) |
| 79 | `bg-green-500` | Terminal button (green) |
| 81 | `text-slate-400` | Terminal title |
| 84 | `text-slate-400` | Terminal prompt line |
| 85 | `text-green-400` | Terminal prompt symbol |
| 87 | `text-slate-300` | Terminal output |
| 88 | `text-amber-400` | Terminal arrow |
| 90 | `text-slate-300` | Terminal output |
| 91 | `text-green-400` | Terminal checkmark |
| 93 | `text-slate-400` | Terminal label |
| 94 | `text-blue-400` | Terminal label accent |
| 96 | `text-slate-300` | Terminal output |
| 97 | `text-amber-400` | Terminal cursor |
| 102 | `text-slate-500` | GPU label |
| 105 | `text-slate-400` | GPU name |
| 106 | `bg-slate-800` | Progress bar background |
| 107 | `bg-amber-500` | Progress bar fill |
| 109 | `text-slate-400` | GPU percentage |
| 130 | `text-slate-500` | Cost label |
| 131 | `text-green-400` | Cost value |

---

## Proposed Token Replacements

### Background & Layout

```tsx
// Line 6: Hero background gradient
// DECISION NEEDED: Keep gradient or use solid background?
// Option 1: Solid background
className="relative min-h-screen flex items-center bg-background"

// Option 2: Keep gradient with tokens (requires new tokens)
className="relative min-h-screen flex items-center bg-gradient-to-br from-background via-card to-secondary"
```

### Badge Component

```tsx
// Line 11: Badge
className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 border border-primary/20 rounded-full text-primary text-sm"

// Line 13: Pulse animation
className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"

// Line 14: Pulse dot
className="relative inline-flex rounded-full h-2 w-2 bg-primary"
```

### Typography

```tsx
// Line 19: Main heading
className="text-5xl lg:text-7xl font-bold text-foreground leading-tight text-balance"

// Line 22: Heading accent
className="text-primary"

// Line 25: Subheading
className="text-xl text-muted-foreground leading-relaxed text-pretty"
```

### Buttons

```tsx
// Line 33: Primary CTA
className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold text-lg h-14 px-8"

// Line 41: Secondary button
className="border-border text-foreground hover:bg-secondary h-14 px-8 bg-transparent"
```

### Trust Indicators

```tsx
// Lines 49, 53, 57, 63: Trust indicator text
className="text-muted-foreground"

// Line 54: Star icon
className="fill-primary text-primary"

// Lines 58, 64: Icon borders
className="border-border"
```

### Terminal Visual

```tsx
// Line 74: Terminal container
className="bg-card border border-border rounded-lg overflow-hidden shadow-2xl"

// Line 75: Terminal header
className="flex items-center gap-2 px-4 py-3 bg-muted border-b border-border"

// Lines 77-79: Terminal buttons (keep as-is - decorative)
// These are macOS-style window buttons, semantic meaning is important
className="h-3 w-3 rounded-full bg-red-500"
className="h-3 w-3 rounded-full bg-amber-500"  
className="h-3 w-3 rounded-full bg-green-500"

// Line 81: Terminal title
className="text-muted-foreground text-sm ml-2 font-mono"

// Line 84: Terminal prompt line
className="text-muted-foreground"

// Line 85: Terminal prompt symbol (keep green - semantic)
className="text-green-400"  // Or use text-chart-3 (green chart color)

// Line 87, 90, 96: Terminal output
className="text-foreground pl-4"

// Line 88: Terminal arrow (keep amber - semantic)
className="text-primary"

// Line 91: Terminal checkmark (keep green - semantic)
className="text-chart-3"  // Using chart-3 which is green

// Line 93: Terminal label
className="text-muted-foreground pl-4"

// Line 94: Terminal label accent (keep blue - semantic)
className="text-chart-2"  // Using chart-2 which is blue

// Line 97: Terminal cursor
className="text-primary animate-pulse"

// Line 102: GPU label
className="text-muted-foreground text-xs"

// Line 105: GPU name
className="text-muted-foreground text-xs w-24"

// Line 106: Progress bar background
className="flex-1 h-2 bg-muted rounded-full overflow-hidden"

// Line 107: Progress bar fill
className="h-full bg-primary w-[85%]"

// Line 109: GPU percentage
className="text-muted-foreground text-xs"

// Line 130: Cost label
className="text-muted-foreground"

// Line 131: Cost value (keep green - semantic for $0)
className="text-chart-3 font-bold"  // Using chart-3 (green)
```

---

## New Tokens Required

**None** - We can use existing chart colors for semantic terminal output.

However, if you want more semantic tokens for terminal/code output:

```css
/* Optional: Add to globals.css */
:root {
  --terminal-success: #10b981;  /* green-500 */
  --terminal-error: #ef4444;    /* red-500 */
  --terminal-info: #3b82f6;     /* blue-500 */
  --terminal-warning: #f59e0b;  /* amber-500 */
}

.dark {
  --terminal-success: #22c55e;  /* green-400 */
  --terminal-error: #f87171;    /* red-400 */
  --terminal-info: #60a5fa;     /* blue-400 */
  --terminal-warning: #fbbf24;  /* amber-400 */
}
```

---

## Implementation Notes

### Gradient Background Decision

The hero currently uses a complex gradient: `from-slate-950 via-slate-900 to-slate-800`

**Options:**
1. **Simplify to solid:** Use `bg-background` (easiest, theme-adaptive)
2. **Keep gradient with tokens:** Create gradient tokens or use existing card/secondary
3. **Force dark hero:** Always use dark background regardless of theme

**Recommendation:** Start with solid `bg-background`, then add gradient tokens if needed.

### Terminal Color Semantics

The terminal uses colors with semantic meaning:
- **Green:** Success, prompts, $0 cost
- **Red/Amber/Green dots:** macOS window buttons (keep as-is)
- **Blue:** Labels, metadata
- **Amber:** Warnings, arrows, loading states

These should use `chart-*` tokens or new terminal-specific tokens.

### Progress Bars

Progress bars use `bg-amber-500` for fill. This should map to `bg-primary` to maintain brand consistency.

---

## Verification Checklist

- [ ] Hero renders correctly in light mode
- [ ] Hero renders correctly in dark mode
- [ ] Gradient/background looks good in both themes
- [ ] Badge pulse animation works
- [ ] Terminal visual is readable in both themes
- [ ] Terminal colors maintain semantic meaning
- [ ] Progress bars are visible in both themes
- [ ] Trust indicators are readable
- [ ] Buttons have good contrast
- [ ] No hardcoded `slate-*`, `amber-*` classes remain (except decorative terminal buttons)

---

## Estimated Complexity

**Medium-High** - Many color instances, gradient decision needed, terminal semantics must be preserved.
