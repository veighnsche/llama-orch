# Font System Centralized - rbee-ui is Now Single Source of Truth

## Problem Statement

Fonts were duplicated across multiple locations:
- **Commercial app**: Loaded Geist fonts via `next/font` in `layout.tsx`
- **Storybook**: Had NO font loading → console/code components showed sans-serif instead of mono
- **rbee-ui**: Referenced fonts but didn't load them

This violated DRY principles and caused inconsistencies between environments.

## Solution Implemented

**rbee-ui is now the single source of truth for ALL fonts.**

### Changes Made

#### 1. Created Central Font Loading

**File:** `packages/rbee-ui/src/tokens/fonts.css`

```css
/* Geist Sans - Variable Font */
@font-face {
  font-family: 'Geist Sans';
  src: url('../../node_modules/geist/dist/fonts/geist-sans/Geist-Variable.woff2') format('woff2');
  font-weight: 100 900;
  font-display: swap;
}

/* Geist Mono - Variable Font */
@font-face {
  font-family: 'Geist Mono';
  src: url('../../node_modules/geist/dist/fonts/geist-mono/GeistMono-Variable.woff2') format('woff2');
  font-weight: 100 900;
  font-display: swap;
}

/* Source Serif 4 - from Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&display=swap');

/* CSS Variables */
:root {
  --font-geist-sans: 'Geist Sans', ...fallbacks;
  --font-geist-mono: 'Geist Mono', ...fallbacks;
  --font-source-serif: 'Source Serif 4', ...fallbacks;
}
```

#### 2. Integrated Fonts into Global CSS

**File:** `packages/rbee-ui/src/tokens/globals.css`

```css
@import "tailwindcss";
@import "@repo/tailwind-config";
@import "./fonts.css";  /* ← ADDED */
@import "./theme-tokens.css";
```

#### 3. Moved geist Dependency to rbee-ui

**Before:** `apps/commercial/package.json` had `geist` dependency  
**After:** `packages/rbee-ui/package.json` has `geist` dependency

```bash
# Removed from app
cd apps/commercial && pnpm remove geist

# Added to rbee-ui
cd packages/rbee-ui && pnpm add geist
```

#### 4. Simplified Next.js Layout

**File:** `apps/commercial/app/layout.tsx`

**Before:**
```tsx
import { GeistMono } from 'geist/font/mono'
import { GeistSans } from 'geist/font/sans'
import { Source_Serif_4 } from 'next/font/google'

const sourceSerif = Source_Serif_4({...})

<html className={`${GeistSans.variable} ${GeistMono.variable} ${sourceSerif.variable}`}>
```

**After:**
```tsx
// ✅ All fonts are loaded in @rbee/ui/styles.css
import '@rbee/ui/styles.css'

<html lang="en" suppressHydrationWarning>
```

## Architecture Benefits

### 1. Single Source of Truth

All fonts are loaded in one place: `rbee-ui/src/tokens/fonts.css`

**Works everywhere:**
- ✅ Next.js commercial app
- ✅ Storybook
- ✅ Any future apps or tools

### 2. No Environment Dependencies

Fonts are loaded via standard `@font-face`, not `next/font`:
- ✅ Works in Next.js
- ✅ Works in Vite/Storybook
- ✅ Works in any bundler
- ✅ No Next.js dependency

### 3. Automatic Distribution

When apps import `@rbee/ui/styles.css`, they automatically get:
- All design tokens
- All fonts (Geist Sans, Geist Mono, Source Serif 4)
- All Tailwind utilities
- All theme styles

**Apps don't need to:**
- Import fonts separately
- Configure `next/font`
- Worry about font loading

### 4. Consistent Rendering

**Before:** Commercial site had mono fonts, Storybook didn't  
**After:** Both render identically with proper mono fonts

## Font Families Available

All three fonts are now globally available:

```tsx
// Sans-serif (default body font)
<p className="font-sans">Geist Sans</p>

// Serif (editorial content)
<h1 className="font-serif">Source Serif 4</h1>

// Monospace (code/console)
<code className="font-mono">Geist Mono</code>
```

## CSS Variables Available

```css
var(--font-geist-sans)    /* Geist Sans with fallbacks */
var(--font-geist-mono)    /* Geist Mono with fallbacks */
var(--font-source-serif)  /* Source Serif 4 with fallbacks */
```

## Verification

### Test in Commercial App
```bash
cd apps/commercial
pnpm dev
```

Visit any page with console/code components → Should show Geist Mono

### Test in Storybook
```bash
cd packages/rbee-ui
pnpm storybook
```

Open TerminalWindow, ConsoleOutput, or CodeBlock stories → Should show Geist Mono

## What Was Fixed

1. ✅ **Storybook console fonts** - Now use Geist Mono instead of sans-serif
2. ✅ **Font duplication** - Removed from Next.js app, centralized in rbee-ui
3. ✅ **Single source of truth** - rbee-ui owns all font loading
4. ✅ **Environment consistency** - Same fonts in all contexts

## Migration Notes

If you have other apps consuming `@rbee/ui`:

**Before (manual font loading):**
```tsx
import { GeistSans } from 'geist/font/sans'
import '@rbee/ui/styles.css'

<html className={GeistSans.variable}>
```

**After (automatic from rbee-ui):**
```tsx
import '@rbee/ui/styles.css'

<html>
```

That's it! Fonts are included automatically.

## Files Changed

1. `packages/rbee-ui/src/tokens/fonts.css` - **NEW** - Central font loading
2. `packages/rbee-ui/src/tokens/globals.css` - Added fonts.css import
3. `packages/rbee-ui/package.json` - Added geist dependency
4. `apps/commercial/app/layout.tsx` - Removed font imports
5. `apps/commercial/package.json` - Removed geist dependency

## Related Documentation

- Font fix analysis: `packages/rbee-ui/FONT_FIX_ANALYSIS.md`
- Original serif fix: `frontend/FONT_SYSTEM_FIX_COMPLETE.md`
- Design system: `packages/rbee-ui/DESIGN_SYSTEM.md`
