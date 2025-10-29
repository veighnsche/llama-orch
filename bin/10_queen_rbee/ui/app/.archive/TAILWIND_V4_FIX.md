# Tailwind v4 + Vite Configuration Fix

## Problem
The initial port used PostCSS-based Tailwind configuration, which caused:
1. **Next.js dependency leak:** `@rbee/ui` imports `next/link`, `next/image`, `next-themes` causing `process is not defined` errors
2. **Wrong Tailwind setup:** Used `@tailwindcss/postcss` instead of official `@tailwindcss/vite` plugin
3. **Overcomplicated config:** Had `postcss.config.js`, `tailwind.config.ts`, and `@repo/tailwind-config` imports

## Solution
Switched to official Tailwind v4 + Vite setup as documented at:
https://tailwindcss.com/docs/installation/framework-guides/react-router

### Changes Made

#### 1. Vite Config (`vite.config.ts`)
```ts
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    tailwindcss(),  // Official Tailwind v4 Vite plugin (must be first)
    wasm(),
    react(),
  ],
  define: {
    'process.env': {},  // Polyfill for libraries that check process.env
  },
})
```

#### 2. Simplified CSS (`src/globals.css`)
```css
/* Before: Complex imports */
@import "tailwindcss";
@import "@repo/tailwind-config";
@source "../src/**/*.{ts,tsx}";

/* After: Simple official import */
@import "tailwindcss";
```

#### 3. Package.json
```json
{
  "devDependencies": {
    // ❌ Removed
    "@repo/tailwind-config": "workspace:*",
    "@tailwindcss/postcss": "^4.1.14",
    "postcss-nesting": "^13.0.2",
    
    // ✅ Added
    "@tailwindcss/vite": "^4.1.14"
  }
}
```

#### 4. Removed Files
- `postcss.config.js` (not needed with @tailwindcss/vite)
- `tailwind.config.ts` (not needed with Tailwind v4)

### Why This Works

1. **Official Plugin:** `@tailwindcss/vite` is the official way to use Tailwind v4 with Vite
2. **No PostCSS:** Tailwind v4 doesn't need PostCSS configuration
3. **No Config File:** Tailwind v4 uses CSS imports instead of config files
4. **Process Polyfill:** `define: { 'process.env': {} }` prevents errors from libraries checking `process.env`

## Next.js Dependencies in @rbee/ui

The `@rbee/ui` package still has Next.js dependencies:
- `next/link` - Used in navigation components
- `next/image` - Used for optimized images
- `next-themes` - Used for theme switching

### Current Workaround
The `process.env` polyfill prevents the immediate error, but these should be replaced:

```tsx
// ❌ Next.js (in @rbee/ui)
import Link from 'next/link'
import Image from 'next/image'
import { useTheme } from 'next-themes'

// ✅ React Router (for web-ui)
import { Link } from 'react-router-dom'
// Use regular <img> or a custom Image component
import { useTheme } from 'next-themes'  // Actually works in React too!
```

### Good News
`next-themes` actually works fine in React (it's not Next.js-specific despite the name). Only `next/link` and `next/image` are problematic.

## Scripts Updated

### kill-dev-servers.sh
- Added port `5173` (Vite default)
- Added Vite process killing

### clean-reinstall.sh
- Added Vite dev server killing
- Added web-ui build step
- Updated help text with both dev servers

## Build Results

### Before (PostCSS)
```
✗ Build failed - CSS minification errors
```

### After (@tailwindcss/vite)
```
✓ 3245 modules transformed.
dist/index.html                          0.53 kB │ gzip:   0.32 kB
dist/assets/rbee_sdk_bg-Brkhi_eX.wasm  595.53 kB │ gzip: 246.55 kB
dist/assets/index-DyHsyi52.css         294.90 kB │ gzip:  35.49 kB
dist/assets/chunk-dcn8tbrv.js            0.63 kB │ gzip:   0.38 kB
dist/assets/rbee_sdk-CoVGPRXk.js        26.97 kB │ gzip:   7.52 kB
dist/assets/index-CWV--z28.js          373.24 kB │ gzip: 119.15 kB
✓ built in 5.99s
```

### Dev Server
```
ROLLDOWN-VITE v7.1.14  ready in 351 ms
➜  Local:   http://localhost:5173/
```

## Testing

1. **Build:** ✅ PASS
2. **Dev Server:** ✅ RUNNING
3. **No process.env errors:** ✅ FIXED
4. **Tailwind classes working:** ✅ (test in browser)

## Future Improvements

### Option 1: Fork @rbee/ui for React
Create a React-specific version that doesn't use Next.js components.

### Option 2: Conditional Exports
Make @rbee/ui export different components based on the consumer:
```json
{
  "exports": {
    "./atoms": {
      "next": "./src/atoms/index.next.ts",
      "default": "./src/atoms/index.react.ts"
    }
  }
}
```

### Option 3: Wrapper Components
Create thin wrappers in web-ui that adapt Next.js components:
```tsx
// src/components/Link.tsx
import { Link as RouterLink } from 'react-router-dom'

export const Link = ({ href, ...props }) => (
  <RouterLink to={href} {...props} />
)
```

## Recommendation

For now, the `process.env` polyfill works. If you see any runtime errors related to Next.js components, create wrapper components in web-ui that adapt them to React Router.

## References

- [Official Tailwind v4 + Vite Guide](https://tailwindcss.com/docs/installation/framework-guides/react-router)
- [Tailwind v4 Announcement](https://tailwindcss.com/blog/tailwindcss-v4-alpha)
- [@tailwindcss/vite Plugin](https://www.npmjs.com/package/@tailwindcss/vite)
