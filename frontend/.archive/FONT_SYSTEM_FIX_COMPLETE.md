# Font System Fix - Serif and Mono Fonts Now Working

## Problem Statement

The `font-serif` and `font-mono` Tailwind utilities were not working correctly across the site:
- **font-serif**: Fell back to system serif fonts instead of Source Serif 4
- **font-mono**: May not work in all rendering contexts (Storybook, etc.)

## Root Causes Identified

### 1. Serif Font - COMPLETELY BROKEN

**Issues:**
1. Source Serif 4 was referenced in design tokens but **never loaded**
2. No `@font-face` declaration
3. No `next/font` import
4. `shared-styles.css` @theme block **missing** `--font-serif` definition entirely

**Result:** `.font-serif` utility generated CSS `font-family: var(--font-serif)` but the variable used system fallback.

### 2. Mono Font - DEPENDENCY ISSUE  

**Issues:**
1. Geist Mono was loaded in `layout.tsx`
2. But `--font-geist-mono` CSS variable only existed in Next.js context
3. Would fail in Storybook or standalone component rendering

## Fixes Applied

### 1. Load Source Serif 4 Font

**File:** `apps/commercial/app/layout.tsx`

Added font import and configuration:
```tsx
import { Source_Serif_4 } from 'next/font/google'

const sourceSerif = Source_Serif_4({
  subsets: ['latin'],
  variable: '--font-source-serif',
  weight: ['400', '600', '700'],
})

// Added to className:
className={`${GeistSans.variable} ${GeistMono.variable} ${sourceSerif.variable}`}
```

This creates the CSS variable `--font-source-serif` with properly loaded font files.

### 2. Add Serif to Tailwind Theme

**File:** `packages/tailwind-config/shared-styles.css`

Added missing font definition:
```css
/* Fonts */
--font-sans: var(--font-geist-sans);
--font-serif: var(--font-source-serif);  /* ← ADDED */
--font-mono: var(--font-geist-mono);
```

This maps the CSS variable to Tailwind's font utility classes.

## What Now Works

### Sans-serif (Already Working)
```tsx
<p className="font-sans">Uses Geist Sans</p>
```
→ Renders with Geist Sans font

### Serif (NOW FIXED ✅)
```tsx
<h1 className="font-serif">Uses Source Serif 4</h1>
```
→ Renders with Source Serif 4 (weights: 400, 600, 700)

### Monospace (Already Working, Now Explicit)
```tsx
<code className="font-mono">Uses Geist Mono</code>
```
→ Renders with Geist Mono font

## Verification

After rebuilding:

1. **Check compiled CSS:** `.next/static/css/app/layout.css` should show:
   ```css
   --font-serif: var(--font-source-serif);
   ```

2. **Check font loading:** Network tab should show Source Serif 4 font files loading from Google Fonts

3. **Visual test:** Any `font-serif` element should render with Source Serif 4, not Georgia/Times

## Design System Alignment

This fix aligns with the design system spec in `rbee-ui/DESIGN_SYSTEM.md`:
- ✅ Sans-serif: Geist (system font stack)
- ✅ Serif: Source Serif 4 (for editorial content, emphasis)
- ✅ Monospace: Geist Mono (for code)

## Files Changed

1. `apps/commercial/app/layout.tsx` - Added Source Serif 4 font loading
2. `packages/tailwind-config/shared-styles.css` - Added `--font-serif` to @theme

## Related Documentation

- Design system: `packages/rbee-ui/DESIGN_SYSTEM.md`
- Font analysis: `packages/rbee-ui/FONT_FIX_ANALYSIS.md`
- Typography tokens: `packages/rbee-ui/src/tokens/index.ts`
