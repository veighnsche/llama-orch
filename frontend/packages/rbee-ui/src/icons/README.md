# Icons

React components for all SVG illustrations.

## Usage

```tsx
import { BeeMark, BeeGlyph, ComplianceShield } from '@rbee/ui/icons'

// Basic usage
<BeeMark />

// Custom size
<BeeMark size={32} />

// With className
<BeeMark className="text-primary" />

// All SVG props supported
<BeeMark 
  size={48}
  className="text-amber-500"
  aria-label="Bee logo"
  role="img"
/>
```

## Available Icons

All icons support:
- `size` prop (number | string) - defaults to 24 or original SVG size
- `className` prop for styling
- All standard SVGProps (aria-*, role, etc.)

Icons are converted from `src/assets/illustrations/*.svg` via `scripts/convert-svgs.mjs`.

## Re-generating Components

If SVG files are updated:

```bash
node scripts/convert-svgs.mjs
```

This will regenerate all icon components and update the index.ts file.
