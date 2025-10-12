# Work Unit 12: Email Capture & Audience Selector

**Priority:** LOW  
**Components:**
- `/components/email-capture.tsx`
- `/components/audience-selector.tsx`

---

## Email Capture Component

### Current Hardcoded Colors

| Line | Current Class | Semantic Purpose |
|------|---------------|------------------|
| 26 | `bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900` | Section background gradient |
| 28 | `bg-amber-500/10` | Badge background |
| 28 | `border-amber-500/20` | Badge border |
| 28 | `text-amber-500` | Badge text |
| 30 | `bg-amber-400` | Pulse animation background |
| 31 | `bg-amber-500` | Pulse dot background |
| 36 | `text-white` | Main heading |
| 38 | `text-slate-300` | Subheading |
| 46 | `text-slate-400` | Icon color |
| 53 | `bg-slate-800` | Input background |
| 53 | `border-slate-700` | Input border |
| 53 | `text-white` | Input text |
| 53 | `placeholder:text-slate-400` | Input placeholder |
| 59 | `bg-amber-500` | Button background |
| 59 | `hover:bg-amber-600` | Button hover |
| 59 | `text-slate-950` | Button text |
| 65 | `text-green-500` | Success message color |
| 71 | `text-slate-400` | Footer text |
| 73 | `border-slate-800` | Divider border |
| 74 | `text-slate-400` | CTA text |
| 79 | `text-amber-500` | Link text |
| 79 | `hover:text-amber-400` | Link hover |

### Proposed Replacements (Email Capture)

```tsx
// Line 26: Section background
// DECISION: Keep dark or make theme-adaptive?
// Option 1: Always dark (current)
className="py-24 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900"
// Option 2: Theme-adaptive
className="py-24 bg-background"

// Line 28: Badge
className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 border border-primary/20 rounded-full text-primary text-sm font-medium mb-6"

// Line 30: Pulse animation
className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"

// Line 31: Pulse dot
className="relative inline-flex rounded-full h-2 w-2 bg-primary"

// Line 36: Main heading
className="text-4xl md:text-5xl font-bold text-foreground mb-6"

// Line 38: Subheading
className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto leading-relaxed"

// Line 46: Icon
className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground"

// Line 53: Input
className="pl-10 bg-input border-border text-foreground placeholder:text-muted-foreground h-12"

// Line 59: Button
className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold h-12 px-8"

// Line 65: Success message
className="flex items-center justify-center gap-2 text-chart-3 text-lg"

// Line 71: Footer text
className="text-sm text-muted-foreground mt-6"

// Line 73: Divider
className="mt-12 pt-12 border-t border-border"

// Line 74: CTA text
className="text-muted-foreground mb-4"

// Lines 79: Link
className="inline-flex items-center gap-2 text-primary hover:text-primary/80 transition-colors font-medium"
```

---

## Audience Selector Component

### Current Hardcoded Colors

| Line | Current Class | Semantic Purpose |
|------|---------------|------------------|
| 8 | `bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950` | Section background gradient |
| 9 | `bg-gradient-to-r from-transparent via-amber-500/20 to-transparent` | Top border gradient |
| 13 | `text-amber-500` | Label text |
| 15 | `text-white` | Main heading |
| 18 | `text-slate-400` | Description text |
| 25 | `border-slate-800/50` | Card border (default) |
| 25 | `bg-slate-900/30` | Card background |
| 25 | `hover:border-blue-500/50` | Card hover border (developer) |
| 25 | `hover:bg-slate-900/60` | Card hover background |
| 25 | `hover:shadow-blue-500/20` | Card hover shadow (developer) |
| 26 | `from-blue-500/0 via-blue-500/0 to-blue-500/0` | Card gradient overlay (developer) |
| 26 | `group-hover:from-blue-500/5 group-hover:via-blue-500/10` | Card hover gradient (developer) |
| 28 | `bg-gradient-to-br from-blue-500 to-blue-600` | Icon background (developer) |
| 28 | `shadow-blue-500/30` | Icon shadow (developer) |
| 28 | `group-hover:shadow-blue-500/50` | Icon hover shadow (developer) |
| 29 | `text-white` | Icon color |
| 32 | `text-blue-400` | Card label (developer) |
| 33 | `text-white` | Card heading |
| 35 | `text-slate-400` | Card description |
| 40, 45, 50 | `text-slate-300` | Feature list text |
| 42, 46, 50 | `text-blue-400` | Feature list arrows (developer) |
| 56 | `bg-blue-600` | Button background (developer) |
| 56 | `text-white` | Button text |
| 56 | `hover:bg-blue-700` | Button hover (developer) |
| 56 | `hover:shadow-blue-500/30` | Button hover shadow (developer) |
| 64-100 | Similar pattern for GPU Providers (green) |
| 102-139 | Similar pattern for Enterprise (amber) |
| 143 | `text-slate-500` | Footer text |

### Proposed Replacements (Audience Selector)

```tsx
// Line 8: Section background
// DECISION: Keep dark or make theme-adaptive?
// Option 1: Always dark (current)
className="relative bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 py-24 sm:py-32"
// Option 2: Theme-adaptive
className="relative bg-background py-24 sm:py-32"

// Line 9: Top border gradient
className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/20 to-transparent"

// Line 13: Label
className="mb-4 font-sans text-sm font-medium uppercase tracking-wider text-primary"

// Line 15: Main heading
className="mb-6 font-sans text-3xl font-semibold tracking-tight text-foreground sm:text-4xl lg:text-5xl"

// Line 18: Description
className="font-sans text-lg leading-relaxed text-muted-foreground"

// Developer Card (Lines 25-61)
className="group relative overflow-hidden border-border bg-card/30 p-8 backdrop-blur-sm transition-all duration-300 hover:scale-[1.02] hover:border-chart-2/50 hover:bg-card/60 hover:shadow-2xl hover:shadow-chart-2/20"

// Developer gradient overlay
className="absolute inset-0 -z-10 bg-gradient-to-br from-chart-2/0 via-chart-2/0 to-chart-2/0 opacity-0 transition-all duration-500 group-hover:from-chart-2/5 group-hover:via-chart-2/10 group-hover:to-transparent group-hover:opacity-100"

// Developer icon
className="mb-6 flex h-14 w-14 items-center justify-center rounded-xl bg-gradient-to-br from-chart-2 to-chart-2 shadow-lg shadow-chart-2/30 transition-all duration-300 group-hover:scale-110 group-hover:shadow-chart-2/50"

// Developer label
className="mb-2 text-sm font-medium uppercase tracking-wider text-chart-2"

// Card heading
className="mb-3 font-sans text-2xl font-semibold text-card-foreground"

// Card description
className="mb-6 font-sans leading-relaxed text-muted-foreground"

// Feature list
className="mb-8 space-y-3 text-sm text-foreground"

// Feature arrows (developer)
className="mt-1 text-chart-2"

// Developer button
className="w-full bg-chart-2 font-medium text-primary-foreground transition-all hover:bg-chart-2/90 hover:shadow-lg hover:shadow-chart-2/30"

// GPU Providers Card (Lines 64-100) - Use chart-3 (green)
// Enterprise Card (Lines 102-139) - Use primary (amber)

// Line 143: Footer text
className="font-sans text-sm leading-relaxed text-muted-foreground"
```

---

## New Tokens Required

**None** - All colors map to existing tokens:
- Developer: `chart-2` (blue)
- GPU Providers: `chart-3` (green)
- Enterprise: `primary` (amber)

---

## Implementation Notes

### Email Capture Background

The email capture section uses a dark gradient. Decision needed:
- **Always dark:** Maintains current design, stands out
- **Theme-adaptive:** Uses `bg-background`, adapts to theme

**Recommendation:** Keep always dark for conversion optimization.

### Audience Selector Cards

Each card has a distinct color theme:
- **Developer:** Blue (`chart-2`)
- **GPU Providers:** Green (`chart-3`)
- **Enterprise:** Amber (`primary`)

These semantics should be preserved with appropriate tokens.

### Card Hover Effects

The cards have complex hover effects with gradients, shadows, and scale. Ensure all hover states work with tokens.

---

## Verification Checklist

- [ ] Email capture section renders correctly in both themes
- [ ] Badge pulse animation works
- [ ] Input field is usable in both themes
- [ ] Button has good contrast
- [ ] Success message is visible
- [ ] Audience selector renders correctly in both themes
- [ ] All three cards are readable
- [ ] Developer card (blue) is distinct
- [ ] GPU Providers card (green) is distinct
- [ ] Enterprise card (amber) is distinct
- [ ] Card hover effects work
- [ ] Icon backgrounds are visible
- [ ] Buttons have good contrast
- [ ] No hardcoded `slate-*`, `amber-*`, `blue-*`, `green-*` classes remain (except if keeping dark backgrounds)

---

## Estimated Complexity

**Medium** - Complex hover effects and gradients, decision needed on background treatment.
