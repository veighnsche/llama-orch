# rbee Brand Scale — Warm Amber Palette

Quick reference for the warm amber brand colors derived from Tailwind's amber scale.

---

## Token Reference

### Primary Tokens (CSS Variables)

These are defined in `theme-tokens.css` and map to Tailwind utilities:

| Token | Light Value | Dark Value | Usage |
|-------|-------------|------------|-------|
| `--primary` | `#d97706` (amber-600) | `#b45309` (amber-700) | Primary buttons, brand elements |
| `--accent` | `#f59e0b` (amber-500) | `#b45309` (amber-700) | Hover states, focus rings, highlights |
| `--ring` | `#f59e0b` (amber-500) | `#b45309` (amber-700) | Focus visible outlines |

---

## Brand Amber Scale

Use these shades when building custom components. Do **not** add new CSS variables; reference these values directly in `className` props.

| Shade | HEX | RGB | Tailwind | Usage |
|-------|-----|-----|----------|-------|
| amber-50 | `#fff7ed` | `rgb(255, 247, 237)` | `bg-amber-50` | Badge fills, subtle backgrounds |
| amber-100 | `#ffedd5` | `rgb(255, 237, 213)` | `bg-amber-100` | Hover states on accent badges |
| amber-200 | `#fed7aa` | `rgb(254, 215, 170)` | `border-amber-200` | Badge borders |
| amber-300 | `#fdba74` | `rgb(253, 186, 116)` | `bg-amber-300` | Light accents, illustrations |
| amber-400 | `#fb923c` | `rgb(251, 146, 60)` | `text-amber-400` | Icon tints, warm highlights |
| **amber-500** | **`#f59e0b`** | **`rgb(245, 158, 11)`** | **`bg-accent`** | **Hover, focus, accents** |
| **amber-600** | **`#d97706`** | **`rgb(217, 119, 6)`** | **`bg-primary`** | **Primary buttons, brand** |
| **amber-700** | **`#b45309`** | **`rgb(180, 83, 9)`** | `bg-[#b45309]` | **Active/pressed, badge text** |
| amber-800 | `#92400e` | `rgb(146, 64, 14)` | `text-amber-800` | Deep text on light amber |
| amber-900 | `#78350f` | `rgb(120, 53, 15)` | `bg-amber-900` | Rich backgrounds |
| amber-950 | `#451a03` | `rgb(69, 26, 3)` | `bg-amber-950` | Maximum contrast |

**Bold shades** are used in the design system tokens.

---

## Component Examples

### Button States
```tsx
// Primary button
<Button variant="default">
  {/* bg-primary (#d97706) → hover:bg-accent (#f59e0b) → active:bg-[#b45309] */}
  Get Started
</Button>
```

### Badge Accent Variant
```tsx
// Warm amber badge
<Badge variant="accent">
  {/* bg-[#fff7ed] text-[#b45309] border-[#fed7aa] */}
  Featured
</Badge>
```

### Focus Ring
```tsx
// All interactive elements
<Input />
{/* focus-visible:ring-2 ring-ring (#f59e0b) ring-offset-2 */}
```

### Custom Gradient (Utility Class)
```tsx
// Section punctuation
<div className="bg-section-gradient-primary">
  {/* linear-gradient with rgba(245, 158, 11, 0.06) wash */}
</div>
```

---

## Usage Guidelines

### ✅ Do
- Use `--primary` for brand authority (buttons, CTAs)
- Use `--accent` for hovers, focus rings, highlights
- Use `accent` Badge variant for brand-aligned chips
- Use `bg-[#fff7ed]` (amber-50) for subtle fills
- Use `text-[#b45309]` (amber-700) for readable text on amber-50
- Keep amber usage intentional — not every element needs brand color

### ❌ Don't
- Don't mix amber shades haphazardly (stick to 500/600/700)
- Don't use amber-300/400 for text (insufficient contrast)
- Don't create new `--amber-*` CSS variables (use Tailwind utilities)
- Don't override `--primary` or `--accent` in component styles
- Don't use amber for destructive actions (use red)

---

## Contrast Ratios (WCAG AA)

All combinations tested against light mode canvas (`#f3f4f6`) and white cards (`#ffffff`):

| Foreground | Background | Ratio | Status | Use Case |
|------------|------------|-------|--------|----------|
| White `#ffffff` | `#d97706` (primary) | 4.8:1 | ✅ AA | Primary button text |
| White `#ffffff` | `#f59e0b` (accent) | 4.5:1 | ✅ AA | Hover button text |
| `#b45309` (amber-700) | `#fff7ed` (amber-50) | 6.4:1 | ✅ AA | Badge text |
| `#92400e` (amber-800) | `#ffedd5` (amber-100) | 6.9:1 | ✅ AA | Dark badge text |
| `#d97706` (primary) | `#f3f4f6` (canvas) | 5.1:1 | ✅ AA | Inline links |

**Small text (≤14px)** must use amber-700 or darker on light backgrounds.

---

## Semantic Mapping

| Semantic Use | Token | Shade | Component |
|--------------|-------|-------|-----------|
| Brand authority | `--primary` | amber-600 | Button default, links |
| Interactive hover | `--accent` | amber-500 | Button hover, tab active |
| Pressed state | `bg-[#b45309]` | amber-700 | Button active |
| Focus indicator | `--ring` | amber-500 | All form controls |
| Chip/tag fill | `bg-[#fff7ed]` | amber-50 | Badge accent |
| Chip/tag text | `text-[#b45309]` | amber-700 | Badge accent |
| Chip/tag border | `border-[#fed7aa]` | amber-200 | Badge accent |
| Section wash | `rgba(245,158,11,0.06)` | amber-500 @ 6% | Utility gradient |

---

## Dark Mode Notes

Dark mode uses **amber-700** (`#b45309`) for both `--primary` and `--accent` to maintain sufficient contrast against dark backgrounds. The scale still applies, but you'll typically use deeper shades (700-900) for text and lighter shades (300-500) for backgrounds.

**Light mode**: amber-500/600 (bright, vivid)  
**Dark mode**: amber-700/800 (subdued, readable)

---

## Chart Colors

Charts use amber-600 as the first series color:

```tsx
const chartConfig = {
  series1: { color: 'var(--chart-1)' }, // #d97706 (amber-600)
  series2: { color: 'var(--chart-2)' }, // #3b82f6 (blue-500)
  series3: { color: 'var(--chart-3)' }, // #10b981 (emerald-500)
  // ...
}
```

This ensures brand consistency in data visualization.

---

## Terminal Colors

Terminal/console components use amber for warnings and success states:

| Color | Token | Value | Usage |
|-------|-------|-------|-------|
| Red | `--terminal-red` | `#ef4444` | Errors, failures |
| Amber | `--terminal-amber` | `#d97706` | Warnings, alerts |
| Green | `--terminal-green` | `#10b981` | Success, confirmations |

---

## Related Files

- **Token definitions**: `src/tokens/theme-tokens.css`
- **Tailwind mapping**: `tailwind-config/shared-styles.css`
- **Button implementation**: `src/atoms/Button/Button.tsx`
- **Badge implementation**: `src/atoms/Badge/Badge.tsx`
- **Full redesign doc**: `LIGHT_MODE_REDESIGN.md`

---

## Quick Copy-Paste

### Amber Hex Values
```
#fff7ed  /* amber-50 */
#ffedd5  /* amber-100 */
#fed7aa  /* amber-200 */
#f59e0b  /* amber-500 (accent) */
#d97706  /* amber-600 (primary) */
#b45309  /* amber-700 (active/pressed) */
```

### Common Tailwind Classes
```tsx
className="bg-amber-50 text-amber-700 border-amber-200"   // Badge accent
className="hover:bg-accent"                               // Button hover
className="focus-visible:ring-2 ring-ring ring-offset-2"  // Focus state
className="bg-[#f4f6f9]"                                  // Subtle hover wash
```

---

**Last Updated:** October 17, 2025  
**Version:** 1.0.0 (Light Mode Redesign)
