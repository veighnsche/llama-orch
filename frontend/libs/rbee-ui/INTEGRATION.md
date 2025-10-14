# Integration Guide

## Quick Start

### 1. Add dependency to your app

```json
{
  "dependencies": {
    "@rbee/ui": "workspace:*"
  }
}
```

Run `pnpm install` in your app directory.

### 2. Import styles in your root layout

**For user-docs (Nextra):**
```tsx
// app/layout.tsx
import "@rbee/ui/styles";
import "nextra-theme-docs/style.css";
```

**For commercial site:**
```tsx
// app/layout.tsx or app/globals.css
import "@rbee/ui/styles";
```

### 3. Use components

```tsx
import { Button } from '@rbee/ui/atoms';
import { Card, CardHeader, CardTitle } from '@rbee/ui/molecules';

<Card>
  <CardHeader>
    <CardTitle>Hello</CardTitle>
  </CardHeader>
  <Button>Click me</Button>
</Card>
```

## Tailwind Configuration

The shared library uses CSS variables, so it works with any Tailwind setup. To access the CSS variables in your Tailwind config:

```js
// tailwind.config.js
export default {
  theme: {
    extend: {
      colors: {
        primary: 'var(--rbee-primary)',
        background: 'var(--rbee-background)',
        foreground: 'var(--rbee-foreground)',
        // ... add more as needed
      },
      borderRadius: {
        lg: 'var(--rbee-radius-lg)',
        md: 'var(--rbee-radius-md)',
        sm: 'var(--rbee-radius-sm)',
      }
    }
  }
}
```

## Nextra Theme Customization

To customize Nextra to match rbee branding:

```tsx
// app/docs/layout.tsx
import { Layout } from 'nextra-theme-docs';

<Layout
  pageMap={pageMap}
  // Use rbee color scheme
  color={{ hue: 45 }} // Honeycomb yellow
  sidebar={{ defaultMenuCollapseLevel: 1 }}
>
  {children}
</Layout>
```

## Component Examples

### Button Variants
```tsx
<Button variant="default">Primary</Button>
<Button variant="secondary">Secondary</Button>
<Button variant="outline">Outline</Button>
<Button variant="ghost">Ghost</Button>
<Button variant="destructive">Delete</Button>
```

### Button Sizes
```tsx
<Button size="sm">Small</Button>
<Button size="md">Medium</Button>
<Button size="lg">Large</Button>
<Button size="icon"><Icon /></Button>
```

### Cards
```tsx
<Card>
  <CardHeader>
    <CardTitle>Feature Title</CardTitle>
    <CardDescription>Feature description here</CardDescription>
  </CardHeader>
  <CardContent>
    <p>Main content</p>
  </CardContent>
  <CardFooter>
    <Button>Action</Button>
  </CardFooter>
</Card>
```

### Badges
```tsx
<Badge>Default</Badge>
<Badge variant="secondary">Beta</Badge>
<Badge variant="outline">New</Badge>
<Badge variant="destructive">Deprecated</Badge>
```

## CSS Utilities

```tsx
// Radial glow background
<div className="rbee-bg-radial-glow">

// Section gradient
<section className="rbee-bg-section-gradient">

// Section gradient with primary accent
<section className="rbee-bg-section-gradient-primary">
```

## Migration Path

### From commercial site components

If you have existing components in the commercial site that match @rbee/ui components:

1. **Audit existing components** - Check which ones match @rbee/ui patterns
2. **Replace imports gradually** - Start with simple atoms like Button
3. **Test thoroughly** - Ensure styling remains consistent
4. **Remove duplicates** - Delete old component files once migrated

### Adding new components to @rbee/ui

1. Create component in `src/atoms/` or `src/molecules/`
2. Export from `index.ts`
3. Test in both apps (commercial + user-docs)
4. Update this guide with usage examples
