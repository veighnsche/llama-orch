# rbee Design System

## Brand Identity

### Primary Color: Honeycomb Yellow/Gold
- **Hex**: `#f59e0b`
- **HSL**: `hsl(45, 96%, 53%)`
- **Usage**: Primary actions, accents, highlights, links
- **Inspiration**: Bees, honeycomb, warmth, Dutch heritage

### Philosophy
- **Professional yet approachable**: Enterprise-grade but not corporate
- **Data-focused**: Clear information hierarchy
- **European aesthetic**: Clean, functional, trustworthy
- **Privacy-first**: Security and trust communicated through design

## Color System

### Light Mode
| Token | Value | Usage |
|-------|-------|-------|
| `--rbee-primary` | #f59e0b | CTAs, links, highlights |
| `--rbee-background` | #ffffff | Page background |
| `--rbee-foreground` | #0f172a | Body text |
| `--rbee-muted` | #f1f5f9 | Secondary backgrounds |
| `--rbee-border` | #e2e8f0 | Borders, dividers |

### Dark Mode
| Token | Value | Usage |
|-------|-------|-------|
| `--rbee-primary` | #f59e0b | CTAs, links (same) |
| `--rbee-background` | #0f172a | Page background |
| `--rbee-foreground` | #f1f5f9 | Body text |
| `--rbee-muted` | #1e293b | Secondary backgrounds |
| `--rbee-border` | #334155 | Borders, dividers |

## Typography

### Font Families
- **Sans-serif**: System font stack (Geist)
- **Serif**: Source Serif 4 (for editorial content, emphasis)
- **Monospace**: System monospace stack (Geist Mono)

### Type Scale
- **xs**: 0.75rem (12px)
- **sm**: 0.875rem (14px)
- **base**: 1rem (16px)
- **lg**: 1.125rem (18px)
- **xl**: 1.25rem (20px)
- **2xl**: 1.5rem (24px)
- **3xl**: 1.875rem (30px)
- **4xl**: 2.25rem (36px)

## Spacing

Using 4px base unit:

| Token | Value | Usage |
|-------|-------|-------|
| xs | 0.5rem (8px) | Tight spacing |
| sm | 1rem (16px) | Component padding |
| md | 1.5rem (24px) | Section spacing |
| lg | 2rem (32px) | Large sections |
| xl | 3rem (48px) | Hero spacing |
| 2xl | 4rem (64px) | Major sections |
| 3xl | 6rem (96px) | Page sections |

## Components

### Atomic Design Principles

#### Atoms
Small, single-purpose components:
- Button
- Badge
- Input
- Label
- Icon

#### Molecules
Composed components with specific function:
- Card (Header, Content, Footer)
- Form Field (Label + Input + Error)
- Alert
- Tooltip

#### Organisms (App-specific)
Complex components composed of molecules:
- Navigation bars
- Hero sections
- Feature grids
- Testimonial sections

## Accessibility

### Requirements
- **WCAG 2.1 Level AA** minimum
- **Color contrast**: 4.5:1 for normal text, 3:1 for large text
- **Keyboard navigation**: All interactive elements accessible
- **Screen readers**: Proper ARIA labels and semantic HTML
- **Focus indicators**: Clear visible focus states

### Focus States
- **Ring color**: Primary color (#f59e0b)
- **Ring width**: 2px
- **Ring offset**: 2px

## Responsive Design

### Breakpoints
- **sm**: 640px
- **md**: 768px
- **lg**: 1024px
- **xl**: 1280px
- **2xl**: 1536px

### Mobile-First Approach
All components designed mobile-first, enhanced for larger screens.

## Animation

### Principles
- **Subtle and purposeful**: Enhance UX, don't distract
- **Fast**: 150-300ms for most transitions
- **Respect preferences**: Honor `prefers-reduced-motion`

### Common Animations
- **Fade in**: opacity 0 → 1, 200ms
- **Slide up**: translateY(8px) → 0, 300ms
- **Button hover**: scale(0.98), 150ms
- **Color transitions**: 200ms ease

## Patterns

### Radial Glow
Background gradient for hero sections:
```css
.rbee-bg-radial-glow {
  background: radial-gradient(60rem 40rem at 50% -10%, hsl(45 96% 53% / 0.07) 0%, transparent 100%);
}
```

### Section Gradients
Subtle gradients for visual hierarchy:
```css
.rbee-bg-section-gradient {
  background: linear-gradient(to bottom, var(--rbee-background), var(--rbee-card));
}
```

## Usage Guidelines

### Do's ✅
- Use primary color for all CTAs
- Maintain consistent spacing rhythm
- Use design tokens for all colors
- Keep accessibility in mind
- Test in both light and dark modes

### Don'ts ❌
- Don't use arbitrary colors outside the system
- Don't mix inconsistent spacing values
- Don't ignore focus states
- Don't use text smaller than 14px for body content
- Don't override dark mode tokens without testing

## Design Tokens in Code

### CSS Variables
```css
.my-component {
  color: var(--rbee-foreground);
  background: var(--rbee-card);
  border-color: var(--rbee-border);
  border-radius: var(--rbee-radius-lg);
}
```

### TypeScript
```tsx
import { colors, spacing, radius } from '@rbee/ui/tokens';

const styles = {
  color: colors.primary,
  padding: spacing.md,
  borderRadius: radius.lg,
};
```

### Tailwind (with CSS variables)
```tsx
<div className="bg-[var(--rbee-card)] text-[var(--rbee-card-foreground)]">
```

## Extending the System

### Adding New Tokens
1. Add to `src/tokens/styles.css`
2. Add to `src/tokens/index.ts` (if TS export needed)
3. Document in this file
4. Update both apps to use new token

### Adding New Components
1. Determine category (atom/molecule)
2. Create in appropriate directory
3. Export from index.ts
4. Add JSDoc comments
5. Test in both apps
6. Document usage in README

## Resources

- [Tailwind CSS Docs](https://tailwindcss.com)
- [Radix UI Primitives](https://radix-ui.com)
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [CSS Variables MDN](https://developer.mozilla.org/en-US/docs/Web/CSS/Using_CSS_custom_properties)
