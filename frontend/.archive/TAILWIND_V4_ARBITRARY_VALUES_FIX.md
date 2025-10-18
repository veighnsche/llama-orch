# Tailwind v4 Arbitrary Values Fix

**Date:** 2025-01-15  
**Issue:** Custom Tailwind values like `translate-y-[2rem]` were not working in Storybook and turborepo apps  
**Root Cause:** 
1. Missing `@source` directives in app-level CSS files
2. Storybook importing pre-built CSS instead of source CSS (preventing JIT compilation)

## Problem

In **Tailwind CSS v4**, arbitrary values (e.g., `translate-y-[2rem]`, `w-[300px]`, `bg-[#123456]`) require **JIT (Just-In-Time) compilation**. This means Tailwind must scan your source files to detect these custom values and generate the corresponding CSS classes.

### Architecture Before Fix

```
┌─────────────────────────────────────────────────────────┐
│ @rbee/ui package (source of truth)                      │
│ ✅ Has @source directive in globals.css                 │
│ ✅ Scans: ../src/**/*.{ts,tsx}                          │
│ ✅ Builds: dist/index.css (with arbitrary values)       │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Commercial App                                          │
│ ❌ NO @source directive                                 │
│ ❌ Only imports pre-built @rbee/ui/styles.css           │
│ ❌ Cannot generate arbitrary values for app files       │
└─────────────────────────────────────────────────────────┘
```

**Result:** Arbitrary values in `@rbee/ui` components work (already in built CSS), but arbitrary values used in app components or pages don't work.

## Solution

### Key Principle: Import Source CSS, Not Pre-Built CSS

**Tailwind v4 with Vite requires importing the SOURCE CSS file** (the one with `@import "tailwindcss"` and `@source` directives) so the Vite plugin can perform JIT compilation at build/dev time.

❌ **WRONG:** Importing pre-built CSS (static, no JIT)
```tsx
import '../dist/index.css'  // Pre-built, arbitrary values won't work
```

✅ **CORRECT:** Importing source CSS (dynamic, JIT enabled)
```tsx
import '../src/tokens/globals.css'  // Source file, JIT works
```

Each consuming app also needs its own `@source` directive to enable JIT compilation for its own files.

### Architecture After Fix

```
┌─────────────────────────────────────────────────────────┐
│ @rbee/ui package (source of truth)                      │
│ ✅ Has @source directive in globals.css                 │
│ ✅ Scans: ../src/**/*.{ts,tsx}                          │
│ ✅ Builds: dist/index.css (with arbitrary values)       │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Commercial App                                          │
│ ✅ HAS @source directive in globals.css                 │
│ ✅ Scans: ../app/**/*.{ts,tsx}                          │
│ ✅ Scans: ../components/**/*.{ts,tsx}                   │
│ ✅ Can generate arbitrary values for app files          │
│ ✅ Imports @rbee/ui/styles.css for design tokens        │
└─────────────────────────────────────────────────────────┘
```

## Changes Made

### 1. Storybook Configuration (`packages/rbee-ui/.storybook/preview.ts`)

**CRITICAL FIX:** Changed from importing pre-built CSS to source CSS

```tsx
// ❌ BEFORE: Pre-built CSS (no JIT)
import '../dist/index.css'

// ✅ AFTER: Source CSS (JIT enabled)
import '../src/tokens/globals.css'
```

This allows the `@tailwindcss/vite` plugin (configured in `main.ts`) to perform JIT compilation, enabling arbitrary values like `translate-y-[100px]` to work in Storybook.

### 2. Commercial App (`apps/commercial`)

**File:** `app/globals.css`

```css
@import 'tailwindcss';
@import '@repo/tailwind-config';
@import 'tw-animate-css';

/**
 * Tailwind v4 JIT: Scan app files for arbitrary values like translate-y-[2rem]
 * This enables custom values in the commercial app itself.
 * The @rbee/ui package scans its own files separately.
 */
@source "../app/**/*.{ts,tsx}";
@source "../components/**/*.{ts,tsx}";
```

**File:** `app/layout.tsx`

```tsx
// Import order: app CSS (JIT scanning) → UI CSS (tokens)
import './globals.css'
import '@rbee/ui/styles.css'
```

### 3. User Docs App (`apps/user-docs`)

**File:** `app/globals.css`

```css
@import "tailwindcss";
@import "@repo/tailwind-config";

/**
 * Tailwind v4 JIT: Scan app files for arbitrary values like translate-y-[2rem]
 * This enables custom values in the user-docs app itself.
 * The @rbee/ui package scans its own files separately.
 */
@source "../app/**/*.{ts,tsx,md,mdx}";
@source "../components/**/*.{ts,tsx}";
@source "../pages/**/*.{ts,tsx,md,mdx}";
```

**File:** `app/layout.tsx`

```tsx
// Import order: app CSS (JIT scanning) → UI CSS (tokens) → Nextra theme
import './globals.css'
import '@rbee/ui/styles.css'
import 'nextra-theme-docs/style.css'
```

## Key Principles

### 1. Source of Truth: `@rbee/ui`

- Design tokens, theme variables, and component styles live in `@rbee/ui`
- Apps import the pre-built CSS: `import '@rbee/ui/styles.css'`
- Apps **never** scan UI package files with `@source`

### 2. App-Level JIT Compilation

- Each app has its own `@source` directive to scan its own files
- This enables arbitrary values in app-specific components and pages
- Apps import `@repo/tailwind-config` to inherit the shared theme

### 3. Import Order Matters

```tsx
// ✅ CORRECT: App CSS first (JIT), then UI CSS (tokens)
import './globals.css'        // Has @source, enables JIT
import '@rbee/ui/styles.css'  // Pre-built tokens and components

// ❌ WRONG: UI CSS first blocks app JIT
import '@rbee/ui/styles.css'
import './globals.css'
```

## CSS Linter Warnings (Expected)

You may see these warnings in your IDE:

```
Unknown at rule @source
Unknown at rule @apply
```

**These are safe to ignore.** The CSS language server doesn't recognize Tailwind v4's directives, but they are valid and work correctly at runtime.

## Verification

✅ **VERIFIED:** Arbitrary values are now working in both the UI package and consuming apps.

### Build Verification

1. **UI package build:**
   ```bash
   cd packages/rbee-ui
   pnpm run build:styles
   ```
   ✅ Generates arbitrary values like `translate-y-[1px]`, `translate-y-[2px]`, etc.

2. **Commercial app build:**
   ```bash
   cd apps/commercial
   pnpm run build
   ```
   ✅ Builds successfully with JIT compilation enabled

### Runtime Verification

**In the browser:**
- Inspect an element using `translate-y-[1px]`
- Verify the CSS class is generated: `.active\:translate-y-\[1px\]:active { --tw-translate-y: 1px; translate: var(--tw-translate-x) var(--tw-translate-y); }`

**Examples found in the codebase:**
- `translate-y-[1px]` - Active button states (EnterpriseCTA, CTAOptionCard)
- `translate-y-[2px]` - Table checkbox alignment
- `translate-y-[-2px]` - Hover effects (TestimonialCard, PricingTier)
- `translate-y-[calc(-50%_-_2px)]` - Tooltip arrow positioning

## Examples of Arbitrary Values

Now working in all apps:

- **Spacing:** `p-[2.5rem]`, `m-[18px]`, `gap-[3.75rem]`
- **Sizing:** `w-[300px]`, `h-[calc(100vh-4rem)]`
- **Colors:** `bg-[#123456]`, `text-[rgb(255,0,0)]`
- **Transforms:** `translate-y-[2rem]`, `rotate-[45deg]`
- **Grid/Flex:** `grid-cols-[1fr_2fr_1fr]`, `gap-[clamp(1rem,5vw,3rem)]`

## Related Files

- `packages/rbee-ui/src/tokens/globals.css` - UI package source scanning
- `packages/tailwind-config/shared-styles.css` - Shared theme configuration
- `apps/commercial/app/globals.css` - Commercial app JIT scanning
- `apps/user-docs/app/globals.css` - User docs app JIT scanning

## References

- [Tailwind CSS v4 Documentation](https://tailwindcss.com/docs/v4-beta)
- [Turborepo with Tailwind Guide](https://turborepo.com/docs/guides/tools/tailwind)
- [Tailwind v4 @source Directive](https://tailwindcss.com/docs/v4-beta#using-source)
- [Integrating Storybook with Tailwind CSS v4.1](https://medium.com/@ayomitunde.isijola/integrating-storybook-with-tailwind-css-v4-1-f520ae018c10)
