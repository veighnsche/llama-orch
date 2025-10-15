# BrandLogo Molecule

Reusable brand logo component combining the bee mark and wordmark atoms.

## Architecture

This molecule combines two atoms:
- **BrandMark** (bee icon)
- **BrandWordmark** ("rbee" text with Geist Mono font)

## Usage

```tsx
import { BrandLogo } from '@/components/molecules'

// Default (medium size, as link to home)
<BrandLogo />

// Small size
<BrandLogo size="sm" />

// Large size
<BrandLogo size="lg" />

// Without link (static display)
<BrandLogo href="" />

// Custom link destination
<BrandLogo href="/about" />

// With priority loading (for above-the-fold usage)
<BrandLogo priority />
```

## Using Atoms Separately

For custom layouts, use the atoms directly:

```tsx
import { BrandMark, BrandWordmark } from '@/components/atoms'

// Icon only
<BrandMark size="lg" />

// Wordmark inline with text
<h1>
  <BrandMark size="lg" className="inline-block align-middle mr-3" />
  <BrandWordmark size="5xl" inline />: your private AI
</h1>

// Wordmark in paragraph
<p>Welcome to <BrandWordmark inline /> platform</p>
```

## Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `className` | `string` | `undefined` | Additional CSS classes |
| `href` | `string` | `'/'` | Link destination (empty string disables link) |
| `priority` | `boolean` | `false` | Next.js Image priority loading |
| `size` | `'sm' \| 'md' \| 'lg'` | `'md'` | Size variant |

## Size Variants

### Small (`sm`)
- Icon: 20×20px
- Text: `text-sm`
- Gap: `gap-2`
- **Use case**: Compact headers, mobile menus

### Medium (`md`) - Default
- Icon: 24×24px
- Text: `text-base`
- Gap: `gap-2.5`
- **Use case**: Primary navigation, standard headers

### Large (`lg`)
- Icon: 32×32px
- Text: `text-xl`
- Gap: `gap-3`
- **Use case**: Footer, landing pages, hero sections

## Design Tokens

### Font
- **Family**: Geist Mono (`var(--font-geist-mono)`)
- **Weight**: Bold (`font-bold`)
- **Tracking**: Tight (`tracking-tight`)
- **Color**: Foreground (`text-foreground`)

### Icon
- **Source**: `/public/brand/bee-mark.svg`
- **Alt text**: "rbee orchestration platform - distributed AI infrastructure"
- **Border radius**: Small (`rounded-sm`)

## Accessibility

- Link has `aria-label="rbee home"` when `href` is provided
- Image has descriptive alt text for screen readers
- Proper semantic HTML (`<Link>` or `<div>`)

## Examples

### Navigation Header
```tsx
<nav>
  <BrandLogo priority />
</nav>
```

### Footer
```tsx
<footer>
  <BrandLogo size="lg" />
</footer>
```

### Icon-Only Button
```tsx
<button>
  <BrandLogo size="sm" showWordmark={false} href="" />
</button>
```

### Custom Styling
```tsx
<BrandLogo className="opacity-80 hover:opacity-100 transition-opacity" />
```

## Brand Alignment

The BrandLogo molecule embodies rbee's brand identity:

- **Geist Mono font**: Technical, developer-focused
- **Bold weight**: Strong, confident presence
- **Bee mark**: Orchestration, efficiency, community
- **Consistent sizing**: Professional, polished UI

## Related Components

- **GitHubIcon**: Social icon for GitHub links
- **NavLink**: Navigation link component
- **FooterColumn**: Footer section component

## Storybook

View all variants in Storybook:
```bash
pnpm storybook
```

Navigate to: **Molecules → BrandLogo**
