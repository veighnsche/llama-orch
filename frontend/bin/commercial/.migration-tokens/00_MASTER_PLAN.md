# CSS Token Migration Master Plan

**Goal:** Replace all hardcoded Tailwind colors with CSS tokens from `globals.css` to enable dark/light mode support.

## Current Token System

The project uses CSS custom properties defined in `/app/globals.css`:

### Available Tokens

**Colors:**
- `--background` / `--foreground`
- `--card` / `--card-foreground`
- `--popover` / `--popover-foreground`
- `--primary` / `--primary-foreground`
- `--secondary` / `--secondary-foreground`
- `--muted` / `--muted-foreground`
- `--accent` / `--accent-foreground`
- `--destructive` / `--destructive-foreground`
- `--border` / `--input` / `--ring`
- `--chart-1` through `--chart-5`
- `--sidebar-*` variants

**Tailwind Mappings:**
- `bg-background` → `var(--background)`
- `text-foreground` → `var(--foreground)`
- `bg-primary` → `var(--primary)`
- `text-primary-foreground` → `var(--primary-foreground)`
- etc.

## Migration Strategy

### Phase 1: Core Components (Priority)
1. Navigation & Footer (always visible)
2. Hero Section (first impression)
3. CTA Sections (conversion critical)

### Phase 2: Feature Sections
4. Features Section
5. Problem/Solution Sections
6. Comparison Section

### Phase 3: Specialized Pages
7. Developers Page Components
8. Enterprise Page Components
9. Features Page Components

### Phase 4: Utility Components
10. Remaining utility components

## Common Patterns to Replace

| Hardcoded | Token Replacement |
|-----------|------------------|
| `bg-slate-950` | `bg-background` (dark mode) |
| `text-white` | `text-foreground` (inverted) |
| `text-slate-900` | `text-foreground` |
| `text-slate-300` | `text-muted-foreground` |
| `text-amber-500` | `text-primary` or `text-accent` |
| `bg-amber-500` | `bg-primary` or `bg-accent` |
| `border-slate-200` | `border-border` |
| `bg-slate-50` | `bg-secondary` |
| `text-green-600` | `text-chart-3` (success) |
| `text-red-500` | `text-destructive` |
| `bg-slate-800` | `bg-card` (dark mode) |

## New Tokens Needed

Some colors may need new tokens added to `globals.css`:

- **Success colors:** Currently using `green-*` → Need `--success` / `--success-foreground`
- **Warning colors:** Currently using `amber-*` (non-primary) → Need `--warning` / `--warning-foreground`
- **Info colors:** Currently using `blue-*` → Need `--info` / `--info-foreground`
- **Gradient stops:** Complex gradients may need semantic tokens

## Work Units

Each work unit document includes:
1. **Component file path**
2. **Current hardcoded colors** (line-by-line audit)
3. **Proposed token replacements**
4. **New tokens to add** (if any)
5. **Verification checklist**

## Verification Process

For each migrated component:
- [ ] Light mode renders correctly
- [ ] Dark mode renders correctly (add `.dark` class to root)
- [ ] No hardcoded colors remain (search for `slate-`, `amber-`, `red-`, `green-`, `blue-`)
- [ ] Semantic meaning preserved (primary = brand, destructive = danger, etc.)
- [ ] Gradients still work

## Work Unit Index

1. [Navigation & Footer](./01_navigation_footer.md)
2. [Hero Section](./02_hero_section.md)
3. [CTA Section](./03_cta_section.md)
4. [Features Section](./04_features_section.md)
5. [Problem Section](./05_problem_section.md)
6. [Solution Section](./06_solution_section.md)
7. [Comparison Section](./07_comparison_section.md)
8. [Pricing Section](./08_pricing_section.md)
9. [FAQ Section](./09_faq_section.md)
10. [Social Proof & Use Cases](./10_social_proof_use_cases.md)
11. [Technical & How It Works](./11_technical_how_it_works.md)
12. [Email Capture & Audience Selector](./12_email_audience.md)
13. [What Is Rbee](./13_what_is_rbee.md)
14. [Developers Page Components](./14_developers_components.md)
15. [Enterprise Page Components](./15_enterprise_components.md)
16. [Features Page Components](./16_features_components.md)

---

## Notes for Future Developers

### ⚠️ CRITICAL: Don't Port Colors 1:1

**The current hardcoded Tailwind colors are NOT perfect.** Use this migration as an opportunity to improve the design system.

**Examples of bad current choices:**
- Using 5 different shades of slate for text (`slate-300`, `slate-400`, `slate-600`, etc.)
- Inconsistent use of amber vs orange for primary actions
- Random blue/green colors that don't follow a semantic pattern
- Too many border colors (`slate-200`, `slate-700`, `slate-800`)

**Your job:** Consolidate and simplify using semantic tokens.

### When to Create New Tokens

**DO create a new token when:**
- The color has semantic meaning (success, warning, info, error)
- The color is used in multiple places for the same purpose
- The color needs different values in light/dark modes
- It improves consistency across the design system

**DON'T create a new token when:**
- It's a one-off decorative color
- It's part of a complex gradient that doesn't need theme switching
- Existing tokens can be reused with opacity modifiers
- You're just copying the current (imperfect) color choices

### Smart Token Decisions

**Example 1: Text Colors**
```tsx
// ❌ BAD: Porting every shade
text-slate-300 → text-slate-300
text-slate-400 → text-slate-400
text-slate-600 → text-slate-600

// ✅ GOOD: Consolidate to semantic tokens
text-slate-300 → text-muted-foreground  (de-emphasized)
text-slate-400 → text-muted-foreground  (de-emphasized)
text-slate-600 → text-muted-foreground  (de-emphasized)
text-slate-900 → text-foreground        (primary text)
```

**Example 2: Backgrounds**
```tsx
// ❌ BAD: Too many background shades
bg-slate-50 → bg-slate-50
bg-slate-100 → bg-slate-100
bg-white → bg-white

// ✅ GOOD: Simplify to two levels
bg-slate-50 → bg-secondary    (subtle background)
bg-slate-100 → bg-secondary   (subtle background)
bg-white → bg-background      (main background)
```

**Example 3: Status Colors**
```tsx
// ❌ BAD: Random colors
text-green-600 → text-green-600
text-green-700 → text-green-700
text-blue-400 → text-blue-400
text-blue-600 → text-blue-600

// ✅ GOOD: Semantic tokens
text-green-600 → text-chart-3  (success)
text-green-700 → text-chart-3  (success)
text-blue-400 → text-chart-2   (info)
text-blue-600 → text-chart-2   (info)
```

### Decision Framework

For each color, ask:
1. **What is its semantic purpose?** (not "what color is it")
2. **Can I use an existing token?** (check the list first)
3. **Is this color used consistently?** (if not, fix it)
4. **Does it need to change in dark mode?** (if yes, use tokens)

### Token Naming Convention

Follow the existing pattern:
```css
:root {
  --semantic-name: #hexcolor;
}

.dark {
  --semantic-name: #different-hexcolor;
}

@theme inline {
  --color-semantic-name: var(--semantic-name);
}
```

### Testing Dark Mode

Add the `dark` class to the root element:
```tsx
<html className="dark">
```

Or use the theme provider (already exists in `components/theme-provider.tsx`).

### Before You Start

1. **Read the existing tokens** in `globals.css`
2. **Understand the semantic meaning** of each token
3. **Look for patterns** across components
4. **Consolidate** similar colors to the same token
5. **Test both themes** to ensure your choices work
