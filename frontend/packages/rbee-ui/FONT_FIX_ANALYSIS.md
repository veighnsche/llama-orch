# Font System Fix Analysis

## Issues Found

### 1. Serif Font - NOT WORKING ❌

**Problem:** `font-serif` Tailwind utility falls back to system fonts

**Root Cause:**
1. `src/tokens/styles.css` defines `--rbee-font-serif: "Source Serif 4"` (just a string, not loaded)
2. `shared-styles.css` @theme block **completely missing** `--font-serif` definition
3. Source Serif 4 font is **never loaded** (no @font-face, no next/font import)

**Result:** Tailwind generates `.font-serif { font-family: var(--font-serif); }` but `--font-serif` uses system fallback

### 2. Mono Font - PARTIALLY WORKING ⚠️

**Problem:** `font-mono` may not work in all contexts

**Root Cause:**
1. Geist Mono IS loaded in `layout.tsx` via `next/font`
2. Creates CSS variable `--font-geist-mono` on the `<html>` element
3. `shared-styles.css` @theme references `var(--font-geist-mono)` 
4. But this only works if layout.tsx renders first

**Current State:** Works in Next.js app, may fail in Storybook or other contexts

## Fix Required

### Option A: Load Source Serif 4 via next/font (Recommended)

```tsx
// app/layout.tsx
import { Source_Serif_4 } from 'next/font/google'

const sourceSerif = Source_Serif_4({ 
  subsets: ['latin'],
  variable: '--font-source-serif',
  weight: ['400', '600', '700'],
})

// Add to className:
className={`${GeistSans.variable} ${GeistMono.variable} ${sourceSerif.variable}`}
```

Then update `shared-styles.css`:
```css
--font-serif: var(--font-source-serif);
```

### Option B: Use system serif fonts (Quick fix)

```css
/* shared-styles.css */
--font-serif: ui-serif, Georgia, Cambria, "Times New Roman", Times, serif;
```

This makes it explicit and matches what Tailwind's default is.

## Verification

Check compiled CSS at `.next/static/css/app/layout.css`:
- Line 10: `--font-serif: ui-serif, Georgia...` (currently using system fallback)
- Line 3777-3779: `.font-serif { font-family: var(--font-serif); }`

The variable is defined but uses system fonts, not Source Serif 4.
