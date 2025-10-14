# @rbee/ui

Shared design system and component library for rbee applications.

## Overview

This package provides:
- **Design Tokens**: Shared CSS variables and theme configuration
- **Atoms**: Basic UI components (Button, Badge, etc.)
- **Molecules**: Composed components (Card, etc.)
- **Utilities**: Helper functions (cn for class merging)

## Design Language

The rbee design system is built around:
- **Primary Color**: Honeycomb Yellow/Gold (#f59e0b, hue: 45)
- **Dark Mode Support**: Automatic theme switching
- **Consistent Spacing**: Predefined spacing scale
- **Accessibility**: Focus states and ARIA support

## Installation

This package is part of the rbee monorepo and uses pnpm workspaces.

```bash
# In your app's package.json
{
  "dependencies": {
    "@rbee/ui": "workspace:*"
  }
}
```

## Usage

### Importing Styles

**Option 1: Import globals.css (recommended for full apps)**

The `globals.css` file includes Tailwind CSS, design tokens, utilities, and animations:

```tsx
// app/layout.tsx
import '@rbee/ui/globals';
```

Or in your CSS file:
```css
/* app/globals.css */
@import '@rbee/ui/globals';
```

**Option 2: Import styles.css (lightweight, for Nextra/docs)**

The `styles.css` file includes only design tokens without Tailwind:

```tsx
// app/layout.tsx
import '@rbee/ui/styles';
```

### Using Components

```tsx
import { Button, Badge } from '@rbee/ui/atoms';
import { Card, CardHeader, CardTitle, CardContent } from '@rbee/ui/molecules';

export function MyComponent() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Welcome</CardTitle>
        <Badge>New</Badge>
      </CardHeader>
      <CardContent>
        <Button variant="default">Get Started</Button>
      </CardContent>
    </Card>
  );
}
```

### Using Utilities

```tsx
import { cn } from '@rbee/ui/utils';

<div className={cn('base-class', conditionalClass && 'active', className)} />
```

### Using Design Tokens

```tsx
import { colors, spacing, radius } from '@rbee/ui/tokens';

const styles = {
  backgroundColor: colors.primary,
  padding: spacing.md,
  borderRadius: radius.lg,
};
```

## CSS Variables

All design tokens are available as CSS variables:

```css
/* Colors */
--rbee-primary
--rbee-primary-foreground
--rbee-background
--rbee-foreground
--rbee-card
--rbee-border

/* Spacing */
--rbee-radius
--rbee-radius-sm
--rbee-radius-lg

/* Typography */
--rbee-font-serif
```

## Component Library Structure

```
src/
├── tokens/           # Design tokens (CSS & TS)
│   ├── styles.css
│   └── index.ts
├── atoms/            # Basic components
│   ├── Button.tsx
│   ├── Badge.tsx
│   └── index.ts
├── molecules/        # Composed components
│   ├── Card.tsx
│   └── index.ts
└── utils/            # Helper functions
    └── index.ts
```

## Extending Components

Components use the `cn()` utility to allow className overrides:

```tsx
<Button className="custom-class">Click me</Button>
<Card className="shadow-xl">Custom styling</Card>
```

## Dark Mode

Dark mode is handled automatically via the `.dark` class on the root element:

```tsx
// Tailwind or next-themes will toggle this
<html className="dark">
```

All CSS variables automatically switch to dark mode values.

## Development

### Storybook

View all components interactively:

```bash
# Start Storybook
pnpm storybook

# Build Storybook for production
pnpm build-storybook
```

Storybook will run at `http://localhost:6006` and display all available components with:
- Live component previews
- Interactive controls
- Dark mode toggle
- Documentation
- Code examples

### Type Checking

```bash
# Type checking
pnpm typecheck
```

## Adding New Components

1. Create component file in `src/atoms/` or `src/molecules/`
2. Export from the corresponding `index.ts`
3. Follow existing patterns for styling and props
4. Use CSS variables for theming
5. Include TypeScript types
6. Add JSDoc comments

## Integration with Applications

### Commercial Site
The commercial site already uses similar components. Gradually migrate to use @rbee/ui components.

### User Docs (Nextra)
Import design tokens to ensure consistent theming with the commercial site while keeping Nextra's layout.

```tsx
// app/layout.tsx
import '@rbee/ui/styles';
```
