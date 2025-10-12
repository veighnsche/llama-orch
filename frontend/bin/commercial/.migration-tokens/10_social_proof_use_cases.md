# Work Unit 10: Social Proof & Use Cases

**Priority:** LOW  
**Components:**
- `/components/social-proof-section.tsx`
- `/components/use-cases-section.tsx`

---

## Social Proof Section

### Current Hardcoded Colors

| Line | Current Class | Semantic Purpose |
|------|---------------|------------------|
| 3 | `bg-slate-50` | Section background |
| 6 | `text-slate-900` | Section heading |
| 14, 18, 22 | `text-amber-600` | Metric numbers (primary) |
| 15, 19, 23, 27 | `text-slate-600` | Metric descriptions |
| 26 | `text-green-600` | €0 metric (success) |
| 33, 47, 61 | `bg-white` | Testimonial card background |
| 33, 47, 61 | `border-slate-200` | Testimonial card border |
| 35 | `bg-gradient-to-br from-blue-400 to-blue-600` | Avatar gradient (blue) |
| 37, 51, 65 | `text-slate-900` | Testimonial name |
| 38, 52, 66 | `text-slate-600` | Testimonial title |
| 41, 55, 69 | `text-slate-600` | Testimonial text |
| 49 | `bg-gradient-to-br from-amber-400 to-amber-600` | Avatar gradient (amber) |
| 63 | `bg-gradient-to-br from-green-400 to-green-600` | Avatar gradient (green) |

### Proposed Replacements (Social Proof)

```tsx
// Line 3: Section background
className="py-24 bg-secondary"

// Line 6: Section heading
className="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance"

// Lines 14, 18, 22: Metric numbers
className="text-4xl font-bold text-primary mb-2"

// Line 26: €0 metric (success)
className="text-4xl font-bold text-chart-3 mb-2"

// Lines 15, 19, 23, 27: Metric descriptions
className="text-sm text-muted-foreground"

// Lines 33, 47, 61: Testimonial cards
className="bg-card border border-border rounded-lg p-6 space-y-4"

// Lines 35, 49, 63: Avatar gradients (keep as-is - decorative)
className="h-12 w-12 rounded-full bg-gradient-to-br from-blue-400 to-blue-600"
className="h-12 w-12 rounded-full bg-gradient-to-br from-amber-400 to-amber-600"
className="h-12 w-12 rounded-full bg-gradient-to-br from-green-400 to-green-600"

// Lines 37, 51, 65: Testimonial names
className="font-bold text-card-foreground"

// Lines 38, 52, 66, 41, 55, 69: Testimonial details and text
className="text-sm text-muted-foreground"
```

---

## Use Cases Section

### Current Hardcoded Colors

| Line | Current Class | Semantic Purpose |
|------|---------------|------------------|
| 5 | `bg-white` | Section background |
| 8 | `text-slate-900` | Section heading |
| 15, 36, 55, 74 | `bg-slate-50` | Use case card background |
| 15, 36, 55, 74 | `border-slate-200` | Use case card border |
| 16 | `bg-blue-100` | Icon background (developer) |
| 17 | `text-blue-600` | Icon color (developer) |
| 19, 40, 59, 78 | `text-slate-900` | Card heading |
| 21, 25, 42, 46, 61, 65, 80, 84 | `text-slate-600` | Card body text |
| 22, 26, 43, 47, 62, 66, 81, 85 | `text-slate-900` | Emphasized text |
| 29, 50, 69, 88 | `text-green-700` | Success text |
| 37 | `bg-amber-100` | Icon background (team) |
| 38 | `text-amber-600` | Icon color (team) |
| 56 | `bg-green-100` | Icon background (homelab) |
| 57 | `text-green-600` | Icon color (homelab) |
| 75 | `bg-slate-100` | Icon background (enterprise) |
| 76 | `text-slate-600` | Icon color (enterprise) |

### Proposed Replacements (Use Cases)

```tsx
// Line 5: Section background
className="py-24 bg-background"

// Line 8: Section heading
className="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance"

// Lines 15, 36, 55, 74: Use case cards
className="bg-secondary border border-border rounded-lg p-8 space-y-4"

// Line 16: Developer icon background
className="h-12 w-12 rounded-lg bg-chart-2/10 flex items-center justify-center"

// Line 17: Developer icon
className="h-6 w-6 text-chart-2"

// Line 37: Team icon background
className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center"

// Line 38: Team icon
className="h-6 w-6 text-primary"

// Line 56: Homelab icon background
className="h-12 w-12 rounded-lg bg-chart-3/10 flex items-center justify-center"

// Line 57: Homelab icon
className="h-6 w-6 text-chart-3"

// Line 75: Enterprise icon background
className="h-12 w-12 rounded-lg bg-muted flex items-center justify-center"

// Line 76: Enterprise icon
className="h-6 w-6 text-muted-foreground"

// Lines 19, 40, 59, 78: Card headings
className="text-xl font-bold text-card-foreground"

// Lines 21, 25, 42, 46, 61, 65, 80, 84: Body text
className="text-muted-foreground"

// Lines 22, 26, 43, 47, 62, 66, 81, 85: Emphasized text
className="font-medium text-card-foreground"

// Lines 29, 50, 69, 88: Success text
className="text-chart-3 font-medium"
```

---

## New Tokens Required

**None** - All colors map to existing tokens. Avatar gradients in testimonials are decorative and can remain as-is.

---

## Implementation Notes

### Social Proof Metrics

The metrics use:
- **Primary color (amber):** For most metrics
- **Success color (green):** For the €0 cost metric

This semantic distinction should be preserved.

### Testimonial Avatars

The avatar gradients are decorative and use different colors to distinguish testimonials. These can remain hardcoded or be replaced with semantic tokens if desired.

### Use Case Icons

Each use case has a distinct icon color:
- **Developer:** Blue (info)
- **Team:** Amber (primary)
- **Homelab:** Green (success)
- **Enterprise:** Neutral (muted)

These semantics should be preserved with appropriate tokens.

---

## Verification Checklist

- [ ] Social proof section renders correctly in both themes
- [ ] Metrics are readable and prominent
- [ ] Testimonial cards are readable
- [ ] Avatar gradients are visible
- [ ] Use cases section renders correctly in both themes
- [ ] Use case cards are readable
- [ ] Icon backgrounds have good contrast
- [ ] Icons are visible
- [ ] Success text (green) is distinct
- [ ] Emphasized text stands out
- [ ] No hardcoded `slate-*`, `amber-*`, `green-*`, `blue-*` classes remain (except decorative avatars)

---

## Estimated Complexity

**Low** - Straightforward replacements, mostly repetitive structure.
