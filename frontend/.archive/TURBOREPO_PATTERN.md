# Turborepo Tailwind CSS Pattern Implementation

## Overview
Migrated from `@source` directive approach to the official Turborepo pattern for Tailwind CSS in monorepos.

## The Problem with `@source`
The previous fix used `@source "../../../libs/rbee-ui/src/**/*.{ts,tsx}"` to tell Tailwind to scan rbee-ui source files. While this worked, it's **not** the Turborepo recommended pattern.

## Turborepo's Official Pattern

### Key Differences

| Aspect | Previous (@source) | Turborepo Pattern |
|--------|-------------------|-------------------|
| **CSS Build** | No build step | Pre-build CSS in UI package |
| **What's Exported** | Raw source CSS | Compiled CSS bundle |
| **Scanning** | Consumer scans UI source | UI package scans its own source |
| **Import** | `@import '@rbee/ui/globals'` | `import '@rbee/ui/styles.css'` |
| **Performance** | Consumer rebuilds all UI CSS | Consumer imports pre-built CSS |

### How It Works

1. **UI Package (`@rbee/ui`)**:
   - Has build scripts: `build:styles` and `dev:styles`
   - Scans its own source files during build
   - Outputs compiled CSS to `dist/index.css`
   - Exports via `"./styles.css": "./dist/index.css"`

2. **Consumer App (commercial)**:
   - Imports pre-built CSS: `import '@rbee/ui/styles.css'`
   - Only scans its own source files
   - No need for `@source` directive
   - Faster builds (no cross-package scanning)

## Implementation

### 1. UI Package Changes (`libs/rbee-ui/package.json`)

```json
{
  "sideEffects": ["**/*.css"],
  "exports": {
    "./styles.css": "./dist/index.css",
    // ... other exports
  },
  "files": ["src", "dist"],
  "scripts": {
    "build:styles": "tailwindcss -i ./src/tokens/globals.css -o ./dist/index.css",
    "dev:styles": "tailwindcss -i ./src/tokens/globals.css -o ./dist/index.css --watch"
  },
  "devDependencies": {
    "@tailwindcss/cli": "^4",
    // ... other deps
  }
}
```

### 2. Consumer App Changes (`bin/commercial/app/layout.tsx`)

```tsx
import '@rbee/ui/styles.css'  // Import pre-built CSS FIRST
import './globals.css'        // Then app-specific styles
```

### 3. App CSS Simplified (`bin/commercial/app/globals.css`)

```css
@import 'tailwindcss';
@import 'tw-animate-css';
/* No @source directive needed! */
```

## Development Workflow

### The Idiomatic Way (AUTOMATIC)

From the **repository root**, run:

```bash
pnpm run dev:commercial
```

This automatically:
1. ✅ Starts `@rbee/ui` CSS watcher (rebuilds on component changes)
2. ✅ Starts Next.js dev server for commercial app
3. ✅ Runs both in parallel with colored output
4. ✅ Hot-reloads when you edit Button, HeroSection, etc.

**No manual steps. No separate terminals. Just works.**

### How It Works

The root `package.json` uses `concurrently` to run both tasks:

```json
{
  "scripts": {
    "dev:commercial": "concurrently --names \"UI,APP\" -c \"cyan,green\" \"pnpm --filter @rbee/ui run dev\" \"pnpm --filter @rbee/commercial run dev\""
  }
}
```

- `--filter @rbee/ui run dev` → Watches CSS, rebuilds `dist/index.css`
- `--filter @rbee/commercial run dev` → Next.js dev server
- `concurrently` → Runs both, prefixes output with [UI] and [APP]

### Manual Control (if needed)

**Just UI watcher:**
```bash
pnpm run dev:ui
```

**Just commercial app:**
```bash
pnpm --filter @rbee/commercial run dev
```

### Production Build

```bash
pnpm run build:commercial
```

Builds UI CSS first, then the commercial app.

## Benefits

1. **Follows Official Pattern**: Matches Turborepo's documented approach
2. **Better Performance**: Consumer apps don't scan UI package source
3. **Cleaner Separation**: Each package manages its own CSS build
4. **Easier Debugging**: Clear boundary between UI CSS and app CSS
5. **Scalable**: Works well with multiple consumer apps

## Reference

- [Turborepo Tailwind Guide](https://turborepo.com/docs/guides/tools/tailwind)
- [Example: with-tailwind](https://github.com/vercel/turborepo/tree/main/examples/with-tailwind)
- [Tailwind CSS v4 Docs](https://tailwindcss.com/docs/v4-beta)

## Files Modified

- `/home/vince/Projects/llama-orch/frontend/libs/rbee-ui/package.json`
- `/home/vince/Projects/llama-orch/frontend/bin/commercial/app/layout.tsx`
- `/home/vince/Projects/llama-orch/frontend/bin/commercial/app/globals.css`

## Migration Checklist

- [x] Add build scripts to `@rbee/ui`
- [x] Add `@tailwindcss/cli` dependency
- [x] Export `./styles.css` → `./dist/index.css`
- [x] Build initial CSS bundle
- [x] Update commercial app to import `@rbee/ui/styles.css`
- [x] Remove `@source` directive from app CSS
- [x] Add automatic dev workflow with `concurrently`
- [x] Remove hack script `dev-ui-styles.sh`
- [ ] Update CI/CD to build UI styles before apps
