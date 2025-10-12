# Work Unit 14: Developers Page Components

**Priority:** LOW  
**Directory:** `/components/developers/`

**Files:**
- `developers-hero.tsx`
- `developers-problem.tsx`
- `developers-solution.tsx`
- `developers-features.tsx`
- `developers-how-it-works.tsx`
- `developers-code-examples.tsx`
- `developers-use-cases.tsx`
- `developers-pricing.tsx`
- `developers-testimonials.tsx`
- `developers-cta.tsx`

---

## Migration Strategy

The developers page components follow the same patterns as the main landing page components. Apply the same token replacements:

### Common Patterns

| Current Pattern | Token Replacement |
|----------------|-------------------|
| `bg-slate-50` | `bg-secondary` |
| `bg-white` | `bg-background` or `bg-card` |
| `bg-slate-900` | `bg-background` (dark mode) or `bg-card` |
| `text-slate-900` | `text-foreground` or `text-card-foreground` |
| `text-slate-600` | `text-muted-foreground` |
| `text-slate-300` | `text-muted-foreground` |
| `text-amber-500/600` | `text-primary` |
| `bg-amber-500/600` | `bg-primary` |
| `text-blue-400/500/600` | `text-chart-2` |
| `bg-blue-100` | `bg-chart-2/10` |
| `text-green-400/500/600` | `text-chart-3` |
| `bg-green-100` | `bg-chart-3/10` |
| `border-slate-200` | `border-border` |

### Component-Specific Notes

#### developers-hero.tsx
- Similar to main `hero-section.tsx`
- Dark gradient background → `bg-background` or keep dark
- Terminal visual → Use `bg-card`, `bg-muted` for code blocks
- Badge → `bg-primary/10 border-primary/20 text-primary`

#### developers-problem.tsx
- Similar to main `problem-section.tsx`
- Use `destructive` token for problem indicators
- Dark background → Keep or use `bg-background`

#### developers-solution.tsx
- Similar to main `solution-section.tsx`
- Architecture diagram → Use `bg-primary` for hierarchy
- Benefit cards → Use semantic colors (chart-2, chart-3, primary)

#### developers-features.tsx
- Similar to main `features-section.tsx`
- Code blocks → `bg-muted` with syntax highlighting
- Callout boxes → Use `primary/10`, `chart-2/10`, `chart-3/10`

#### developers-how-it-works.tsx
- Similar to main `how-it-works-section.tsx`
- Step numbers → `bg-primary text-primary-foreground`
- Code blocks → `bg-muted` with chart colors for syntax

#### developers-code-examples.tsx
- Code blocks → `bg-muted`
- Syntax highlighting:
  - Keywords → `text-chart-4` (purple)
  - Strings → `text-chart-3` (green)
  - Functions → `text-chart-2` (blue)
  - Comments → `text-muted-foreground`

#### developers-use-cases.tsx
- Similar to main `use-cases-section.tsx`
- Icon backgrounds → Use semantic colors
- Success indicators → `text-chart-3`

#### developers-pricing.tsx
- Similar to main `pricing-section.tsx`
- Highlighted tier → `bg-primary/5 border-primary`
- Check icons → `text-chart-3`

#### developers-testimonials.tsx
- Similar to main `social-proof-section.tsx`
- Testimonial cards → `bg-card border-border`
- Avatar gradients → Keep as-is (decorative)

#### developers-cta.tsx
- Similar to main `cta-section.tsx`
- Primary button → `bg-primary hover:bg-primary/90 text-primary-foreground`
- Dark background → Keep or use `bg-accent`

---

## Implementation Approach

### Step 1: Read Each File
Read each component file to identify hardcoded colors.

### Step 2: Apply Pattern Matching
Use the common patterns table above to replace colors systematically.

### Step 3: Preserve Semantics
Ensure semantic colors are preserved:
- **Blue:** Developer-focused, info
- **Green:** Success, positive outcomes
- **Amber:** Primary brand, CTAs
- **Red:** Problems, errors

### Step 4: Test Each Component
Verify each component in both light and dark modes.

---

## Verification Checklist

For each component:
- [ ] Renders correctly in light mode
- [ ] Renders correctly in dark mode
- [ ] Semantic colors preserved
- [ ] Code syntax highlighting works
- [ ] Buttons have good contrast
- [ ] Cards are readable
- [ ] Icons are visible
- [ ] No hardcoded `slate-*`, `amber-*`, `blue-*`, `green-*` classes remain

---

## Estimated Complexity

**Medium** - 10 components with similar patterns to main landing page. Repetitive work but straightforward.

---

## Notes

- The developers page uses **blue** as the accent color (instead of amber) to differentiate from other audiences
- Code examples are prominent → ensure syntax highlighting is clear
- Terminal/CLI visuals → maintain readability in both themes
- Keep the developer-focused aesthetic while making it theme-adaptive
