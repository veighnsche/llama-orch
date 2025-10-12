# TEAM-FE-012 HANDOFF - CRITICAL BUG

**Status:** BLOCKED - Tailwind not scanning storybook package classes  
**Priority:** P0 - Blocks all shared component development

## Problem Statement

**Tailwind CSS in `frontend/bin/commercial` does NOT scan classes from `frontend/libs/storybook`**, even though commercial imports components from the storybook package.

### Reproduction

1. Add `cursor-pointer` to Button component in `frontend/libs/storybook/stories/atoms/Button/Button.vue`
2. Import Button in commercial frontend (already done via `rbee-storybook/stories`)
3. **Result:** `cursor-pointer` class is NOT generated in commercial's CSS
4. **Expected:** Tailwind should scan imported package and generate the class

### Evidence

- Added test classes (`bg-fuchsia-600`, `border-8`, `rotate-2`, etc.) to Button - **NOT rendered**
- HMR updates show storybook files are watched: `/@fs/.../storybook/stories/atoms/Button/Button.vue`
- But Tailwind CSS output does NOT include classes from storybook components

## Environment

- **Monorepo:** pnpm workspace with symlinked packages
- **Commercial frontend:** Nuxt 4 + Tailwind v4 + `@tailwindcss/vite`
- **Storybook lib:** Vue 3 components exported via `rbee-storybook` package
- **Import pattern:** `import { Button } from 'rbee-storybook/stories'`

## What We Tried (All Failed)

### 1. Created `tailwind.config.js` with content paths
```js
// frontend/bin/commercial/tailwind.config.js
export default {
  content: [
    './app/**/*.{vue,js,ts,jsx,tsx}',
    '../../libs/storybook/stories/**/*.{vue,js,ts,jsx,tsx}', // ❌ Doesn't work
  ],
}
```

### 2. Nuxt config comment says "scans automatically"
```js
// frontend/bin/commercial/nuxt.config.ts
vite: {
  plugins: [
    tailwindcss(),
    // ⛔ DO NOT configure content paths here
    // Tailwind v4 scans imported components automatically  // ❌ THIS IS A LIE
  ],
}
```

### 3. Added `cursor-pointer` to Button base classes
```js
// frontend/libs/storybook/stories/atoms/Button/Button.vue
const buttonVariants = cva(
  "... cursor-pointer ...", // ❌ Not generated in commercial CSS
```

## Critical Files

1. **`frontend/bin/commercial/nuxt.config.ts`** - Nuxt + Tailwind config
2. **`frontend/bin/commercial/tailwind.config.js`** - Created by TEAM-FE-011 (doesn't work)
3. **`frontend/libs/storybook/package.json`** - Package exports
4. **`frontend/libs/storybook/stories/atoms/Button/Button.vue`** - Test component with `cursor-pointer`
5. **`node_modules/@tailwindcss/vite`** - Vite plugin source (CHECK THIS!)

## Your Mission (TEAM-FE-012)

### Step 1: Read Engineering Rules
**MANDATORY:** Read `.windsurf/rules/engineering-rules.md` before starting

### Step 2: Deep Investigation Required

**DO NOT GUESS. VERIFY EVERYTHING.**

1. **Check Tailwind v4 + Vite plugin docs:**
   - How does `@tailwindcss/vite` scan for classes?
   - Does it support pnpm workspace symlinks?
   - Does it scan `node_modules`?

2. **Inspect `node_modules/@tailwindcss/vite`:**
   - Read the plugin source code
   - Find where it configures content scanning
   - Check if it respects `tailwind.config.js` content paths
   - Look for pnpm/symlink handling

3. **Check Nuxt 4 + Tailwind v4 integration:**
   - Search for "nuxt 4 tailwind v4 monorepo"
   - Search for "tailwind v4 pnpm workspace"
   - Check if there's a Nuxt-specific way to configure Tailwind v4

4. **Verify pnpm workspace behavior:**
   - Check if symlinked packages are treated differently
   - See if Tailwind needs explicit configuration for workspace packages

### Step 3: Test Your Solution

Add this test to Button and verify it renders:
```js
// frontend/libs/storybook/stories/atoms/Button/Button.vue
default: 'bg-fuchsia-600 text-lime-300 border-8 border-dashed border-rose-500'
```

If you see a bright purple button with lime text and thick pink dashed border, **IT WORKS**.

### Step 4: Document Root Cause

In your handoff, explain:
- **Why** it wasn't working
- **What** the actual solution is
- **Where** you found the answer (docs/source code/GitHub issue)

## Acceptance Criteria

✅ `cursor-pointer` class from Button component is generated in commercial CSS  
✅ Test classes (`bg-fuchsia-600`, etc.) render correctly  
✅ Solution works with pnpm workspace symlinks  
✅ Solution documented with root cause analysis  

## Notes

- This is a **fundamental build configuration issue**, not a Vue/component issue
- The answer is likely in Tailwind v4 or `@tailwindcss/vite` documentation/source
- pnpm workspaces use symlinks - this may affect how Tailwind scans files
- Nuxt 4 is very new - there may be limited documentation

## Resources

- Tailwind v4 docs: https://tailwindcss.com/docs/v4-beta
- `@tailwindcss/vite` on npm
- Nuxt 4 docs: https://nuxt.com/docs
- pnpm workspace docs: https://pnpm.io/workspaces

---

**TEAM-FE-011 OUT. GOOD LUCK TEAM-FE-012. READ THE RULES. INVESTIGATE DEEPLY. DON'T GUESS.**
