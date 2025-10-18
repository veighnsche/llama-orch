# ✅ Turborepo Setup Fixed

## Issues Found & Fixed

### 1. Storybook Not Running in `turbo dev`

**Problem:** `@rbee/ui` package had `storybook` script but no `dev` script, so Turborepo couldn't start it.

**Fix:** Added `dev` script to `frontend/packages/rbee-ui/package.json`:
```json
{
  "scripts": {
    "dev": "storybook dev -p 6006",
    "build": "postcss ./src/tokens/globals.css -o ./dist/index.css"
  }
}
```

### 2. CSS Import Chain Explanation

**How it works:**

```
┌─────────────────────────────────────────────────────────────┐
│ @repo/tailwind-config (shared config package)              │
│ ├─ shared-styles.css                                       │
│ │  └─ @import 'tailwindcss'                               │
│ │  └─ @theme { breakpoints, brand colors }                │
│ └─ Exported as: '@repo/tailwind-config'                   │
└─────────────────────────────────────────────────────────────┘
                           ↓ imported by
┌─────────────────────────────────────────────────────────────┐
│ @rbee/ui (component library)                               │
│ ├─ src/tokens/globals.css                                 │
│ │  └─ @import "tailwindcss" prefix(ui)                    │
│ │  └─ @import '@repo/tailwind-config'  ← SHARED CONFIG   │
│ │  └─ @source '../**/*.{ts,tsx}'                          │
│ │  └─ Custom theme tokens                                 │
│ ├─ Build: postcss → dist/index.css                        │
│ └─ Exported as: '@rbee/ui/styles.css'                     │
└─────────────────────────────────────────────────────────────┘
                           ↓ imported by
┌─────────────────────────────────────────────────────────────┐
│ @rbee/commercial (Next.js app)                             │
│ ├─ app/layout.tsx                                          │
│ │  └─ import '@rbee/ui/styles.css'  ← UI with ui: prefix │
│ ├─ app/globals.css                                         │
│ │  └─ @import 'tailwindcss'                               │
│ │  └─ @import '@repo/tailwind-config'  ← SHARED CONFIG   │
│ └─ Result: App utilities (unprefixed) + UI utilities (ui:)│
└─────────────────────────────────────────────────────────────┘
```

### 3. Why Both Import the Shared Config

**@rbee/ui imports it:**
- Gets breakpoints for `ui:md:flex`, `ui:lg:px-8`, etc.
- Gets brand colors for consistency

**@rbee/commercial imports it:**
- Gets same breakpoints for app-specific `md:flex`, `lg:px-8`, etc.
- Gets same brand colors
- **No conflicts** because UI utilities are prefixed with `ui:`

### 4. What Gets Built

**dist/index.css contains:**
- ✅ All Tailwind utilities with `ui:` prefix
- ✅ Breakpoints from shared config
- ✅ Custom theme tokens
- ✅ Component styles
- ✅ Animation utilities

**Example classes in built CSS:**
```css
.ui\:hidden { display: none; }
.ui\:md\:flex { @media (width >= 48rem) { display: flex; } }
.ui\:bg-primary { background-color: var(--primary); }
```

## Running Development

Now `turbo dev` will start:
- ✅ `@rbee/commercial` on port 3000
- ✅ `@rbee/user-docs` on port 3100  
- ✅ `@rbee/ui` Storybook on port 6006

## Verification

```bash
# Build UI package
pnpm --filter @rbee/ui build

# Check prefixed classes exist
grep -c "ui:" frontend/packages/rbee-ui/dist/index.css

# Start all dev servers
turbo dev
```

All three services should now start correctly!
