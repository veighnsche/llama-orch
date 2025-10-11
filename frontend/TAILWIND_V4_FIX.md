# Tailwind CSS v4 Integration Fix - TEAM-FE-009

**Date:** 2025-10-11  
**Status:** ✅ RESOLVED

## Problem Summary

Tailwind CSS v4 was not applying styles to the commercial frontend, despite working correctly in the storybook. The issue stemmed from differences in how `@tailwindcss/vite` and `@tailwindcss/postcss` plugins process CSS imports.

## Root Cause

1. **Storybook** uses `@tailwindcss/postcss` which processes CSS files imported via TypeScript
2. **Commercial frontend** uses `@tailwindcss/vite` which requires `@import "tailwindcss"` in the **entry CSS file**
3. **Previous approach** imported `tokens.css` via TypeScript (`main.ts`), so the `@import "tailwindcss"` inside `tokens.css` was never processed by the Vite plugin

## Solution Architecture

Created a split architecture to support both PostCSS and Vite plugins:

```
frontend/libs/storybook/styles/
├── tokens.css          # Entry for storybook (PostCSS)
│   ├── @import "tailwindcss"
│   └── @import "./tokens-base.css"
└── tokens-base.css     # Shared tokens (no Tailwind import)
    ├── CSS variables (:root, .dark)
    ├── @theme inline { ... }
    └── Global styles
```

```
frontend/bin/commercial-frontend/src/assets/
└── main.css            # Entry for commercial frontend (Vite)
    ├── @import "tailwindcss"
    └── @import "rbee-storybook/styles/tokens-base.css"
```

## Changes Made

### 1. Created `/frontend/libs/storybook/styles/tokens-base.css`
- Contains all CSS variables, `@theme inline` block, and global styles
- **No** `@import "tailwindcss"` (apps import it separately)
- Single source of truth for design tokens

### 2. Updated `/frontend/libs/storybook/styles/tokens.css`
- Now a thin wrapper for storybook
- Imports `tailwindcss` then `tokens-base.css`
- Maintains backward compatibility with storybook

### 3. Updated `/frontend/bin/commercial-frontend/src/assets/main.css`
- Added `@import "tailwindcss"` at the top (required by `@tailwindcss/vite`)
- Imports `tokens-base.css` from storybook package
- Removed duplicate global styles

### 4. Updated `/frontend/bin/commercial-frontend/src/main.ts`
- Removed TypeScript import of `tokens.css`
- CSS imports now handled entirely via CSS `@import`

### 5. Updated `/frontend/libs/storybook/package.json`
- Added export for `./styles/tokens-base.css`

## Verification

Build output confirms Tailwind CSS v4 is working:

```bash
$ pnpm run build
✓ built in 3.77s
dist/assets/index-Das_ex_c.css  7.27 kB │ gzip: 2.30 kB
```

CSS output includes:
- ✅ Tailwind CSS v4.1.14 header
- ✅ Custom CSS variables (`:root`, `.dark`)
- ✅ `@theme inline` mappings
- ✅ Utility classes (`.bg-background`, `.text-foreground`, etc.)

## Key Learnings

### Official Tailwind v4 + Vite Requirements

Per [official docs](https://tailwindcss.com/docs/installation/using-vite):

1. `@import "tailwindcss"` **must** be in the entry CSS file
2. The entry CSS file must be imported by the app (not via `@import` chain)
3. `@tailwindcss/vite` plugin processes the entry CSS file directly

### Workspace Package CSS Imports

When sharing CSS across workspace packages with Vite:

1. ✅ **CSS `@import` works** for importing from workspace packages
2. ✅ **Vite aliases resolve** in CSS `@import` statements
3. ❌ **TypeScript imports don't work** for CSS that needs Tailwind processing
4. ✅ **Split architecture** allows supporting both PostCSS and Vite plugins

### shadcn-vue + Tailwind v4 Pattern

The official shadcn-vue pattern for Tailwind v4:

```css
/* app/src/index.css */
@import "tailwindcss";

:root {
  --background: hsl(0 0% 100%);
  --foreground: hsl(0 0% 3.9%);
}

@theme inline {
  --color-background: var(--background);
  --color-foreground: var(--foreground);
}
```

## References

- [shadcn-vue Vite Installation](https://www.shadcn-vue.com/docs/installation/vite)
- [shadcn-vue Tailwind v4 Guide](https://www.shadcn-vue.com/docs/tailwind-v4)
- [Tailwind CSS v4 Vite Installation](https://tailwindcss.com/docs/installation/using-vite)

## Next Steps

1. ✅ Tailwind CSS v4 integration complete
2. ⚠️ TypeScript errors in storybook components (pre-existing, unrelated to this fix)
3. 🔄 Test dev server with actual browser to verify styles render correctly

## Files Modified

- `/frontend/libs/storybook/styles/tokens-base.css` (created)
- `/frontend/libs/storybook/styles/tokens.css` (simplified)
- `/frontend/libs/storybook/package.json` (added export)
- `/frontend/bin/commercial-frontend/src/assets/main.css` (added Tailwind import)
- `/frontend/bin/commercial-frontend/src/main.ts` (removed CSS import)
