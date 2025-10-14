# 🚨 TURBOREPO GUARDRAILS - READ BEFORE EDITING CONFIG FILES

## Critical Rules

### ❌ NEVER DO THIS
1. **NEVER add `@source` directive in CSS files** to scan other packages
2. **NEVER use `content: ['../../packages/ui/src/**/*.tsx']`** in Tailwind config
3. **NEVER scan files outside a package's own directory**

### ✅ ALWAYS DO THIS
1. **Each package builds its own CSS** - UI package scans its own `src/**/*.{ts,tsx}`
2. **Apps import pre-built CSS** - `import '@rbee/ui/styles.css'` in layout.tsx
3. **Tailwind v4 auto-scans** - No config file needed, it scans the package automatically

## How It Works

### UI Package (`@rbee/ui`)
- **Builds:** `pnpm run dev` → watches `src/**/*.{ts,tsx}` → outputs `dist/index.css`
- **Exports:** `"./styles.css": "./dist/index.css"` in package.json
- **Scans:** ONLY its own files in `src/`

### Apps (`@rbee/commercial`, `@rbee/user-docs`)
- **Imports:** `import '@rbee/ui/styles.css'` FIRST in layout.tsx
- **Then:** `import './globals.css'` for app-specific styles
- **Scans:** ONLY their own files in `app/`

## Files With Guardrails

1. **`frontend/packages/rbee-ui/src/tokens/globals.css`** - Comments explain no @source
2. **`frontend/apps/commercial/app/globals.css`** - Comments explain no cross-package scanning
3. **`frontend/apps/commercial/app/layout.tsx`** - Comments explain import order

## Reference

Official Turborepo pattern: https://github.com/vercel/turborepo/tree/main/examples/with-tailwind

## Why This Matters

Breaking these rules causes:
- ❌ Turborepo package boundaries violated
- ❌ Slow builds (scanning unnecessary files)
- ❌ Cache invalidation issues
- ❌ Hard-to-debug CSS problems

**Follow the pattern. Don't break Turborepo.**
