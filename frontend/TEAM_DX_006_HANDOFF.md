# TEAM-DX-006 HANDOFF: Shared Component Library Created

**Date:** 2025-10-12  
**Team:** TEAM-DX-006  
**Status:** ✅ COMPLETE - Shared component library ready for use

---

## 🎉 NEW: Shared Component Library

**Created:** `@orchyra/shared-components` at `frontend/libs/shared-components`

**Purpose:** Share atoms and molecules between `commercial` and `user-docs` frontends

**What's included:**
- ✅ 67 atoms (Button, Input, Card, Alert, etc.)
- ✅ 13 molecules (FormField, SearchBar, PricingCard, etc.)
- ✅ Design tokens and styles
- ✅ TypeScript support
- ✅ Full barrel exports

**Both bins configured:**
- ✅ Added to pnpm workspace
- ✅ Dependencies added to both package.json files
- ✅ Nuxt aliases configured
- ✅ Example page created for user-docs

See `frontend/MIGRATION_SHARED_COMPONENTS.md` for full documentation.

---

# PREVIOUS ISSUE: Commercial Histoire CSS Import Issue

**Date:** 2025-10-12  
**Team:** TEAM-DX-006  
**Status:** ❌ BLOCKED - CSS not loading in commercial Histoire (superseded by shared library)

---

## 🚨 CRITICAL ISSUE

**Problem:** Commercial Histoire (port 6007) has NO CSS at all. Components render but are completely unstyled.

**Evidence:**
- Screenshot shows unstyled text, no colors, no layout
- User reports: "the commercial histoire is missing the tailwind css completely"
- Previous issue: `cursor-pointer` missing from 6007 but present on 6006
- Current issue: ALL CSS missing from 6007

---

## What Was Attempted (TEAM-DX-006)

### Approach 1: Import Pre-Compiled CSS from Storybook ❌ FAILED

**Theory:** Import storybook's compiled CSS instead of trying to scan workspace dependencies.

**Implementation:**
1. Created `frontend/bin/commercial/app/assets/css/histoire.css`
2. Added `@import "rbee-storybook/styles/tokens.css"`
3. Updated `histoire.setup.ts` to import `histoire.css` instead of `main.css`
4. Removed PostCSS configuration from `histoire.config.ts`

**Result:** ❌ NO CSS loads at all. Completely unstyled.

**Why it failed:** Unknown. Possible causes:
- Vite not resolving workspace package CSS imports in Histoire
- CSS import path incorrect
- Histoire not processing CSS imports properly
- Missing Vite configuration for CSS handling

---

## Files Modified by TEAM-DX-006

### Configuration Files
1. `frontend/bin/commercial/histoire.config.ts`
   - Added Vue plugin
   - Removed PostCSS plugin
   - Added optimizeDeps exclude for rbee-storybook

2. `frontend/bin/commercial/histoire.setup.ts`
   - Changed from `main.css` to `histoire.css`

3. `frontend/bin/commercial/app/assets/css/histoire.css` (NEW)
   - Imports `rbee-storybook/styles/tokens.css`

4. `frontend/bin/commercial/package.json`
   - Removed unused dependencies: `@tailwindcss/postcss`, `postcss`, `chokidar`

### Component Files (Fixed Imports)
5-11. All template files (HomeView, DevelopersView, etc.)
   - Changed from barrel imports to local relative imports
   - Added TEAM-DX-006 signatures

12. `DevelopersFeatures.vue`
   - Added missing `computed` import

13. `frontend/.dx-tool/FEATURE_REQUESTS.md`
   - Documented DX tool CSS extraction bug

---

## Root Cause Analysis

### The Original Problem

**6006 (storybook):** ✅ Works
- Has its own Tailwind config
- Scans `stories/**/*.vue`
- Generates complete CSS
- `cursor-pointer` class works

**6007 (commercial Histoire):** ❌ Broken
- Tries to scan `../../libs/storybook/stories/**/*.vue`
- Tailwind doesn't scan workspace dependencies
- CSS missing classes from storybook
- Now: ALL CSS missing

### Why Importing Storybook CSS Failed

**Hypothesis 1:** Vite doesn't resolve workspace package CSS in Histoire context
- Histoire uses different Vite configuration than Nuxt
- CSS imports from workspace packages may not work in Histoire

**Hypothesis 2:** The CSS file path is wrong
- `rbee-storybook/styles/tokens.css` works in Nuxt
- But may not work in Histoire's Vite context

**Hypothesis 3:** Missing CSS loader configuration
- Histoire may need explicit CSS handling configuration
- Vite may not be processing `@import` statements

---

## Next Steps for Next Team

### Priority 1: Debug CSS Import (IMMEDIATE)

**Check if the CSS file is being loaded:**

1. Start Histoire server:
   ```bash
   cd frontend/bin/commercial
   pnpm run story:dev
   ```

2. Open browser dev tools → Network tab
3. Look for CSS file requests
4. Check if `tokens.css` is being requested and what the response is

**Expected outcomes:**
- ✅ If CSS file loads but is empty → Problem with storybook CSS generation
- ✅ If CSS file returns 404 → Problem with path resolution
- ✅ If CSS file not requested at all → Problem with import statement
- ✅ If CSS file loads with content → Problem with CSS application

### Priority 2: Try Alternative Solutions

**Option A: Use absolute file path instead of workspace package**

```typescript
// histoire.setup.ts
import '../../libs/storybook/styles/tokens.css'
```

**Option B: Copy the compiled CSS to commercial**

```bash
# Add to package.json scripts
"prebuild": "cp ../../libs/storybook/dist/tokens.css ./app/assets/css/"
```

Then import the local copy:
```typescript
import './app/assets/css/tokens.css'
```

**Option C: Revert to PostCSS + Fix Workspace Scanning**

Go back to using `@tailwindcss/postcss` but fix the content scanning:

```javascript
// tailwind.config.js
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = fileURLToPath(new URL('.', import.meta.url))

export default {
  content: [
    './app/**/*.{vue,js,ts,jsx,tsx}',
    // Use absolute path
    resolve(__dirname, '../../libs/storybook/stories/**/*.{vue,js,ts,jsx,tsx}'),
  ],
}
```

**Option D: Use Tailwind CLI to scan both directories**

```json
// package.json
"scripts": {
  "story:dev": "concurrently \"tailwindcss -i ./app/assets/css/main.css -o ./dist/tailwind.css --watch\" \"histoire dev --port 6007\""
}
```

Then import the generated CSS:
```typescript
import './dist/tailwind.css'
```

### Priority 3: Verify with DX Tool

Once CSS is loading, verify with the DX tool:

```bash
cd frontend/.dx-tool
cargo run --release -- css --class-exists "cursor-pointer" "http://localhost:6007"
```

**Expected:** `✓ Class 'cursor-pointer' found in stylesheet`

**Note:** The DX tool has a bug where it checks CSS rules but not HTML classes. See `FEATURE_REQUESTS.md` for details.

---

## Known Issues

### DX Tool Bug

The `dx css --class-exists` command is broken. It checks if a CSS rule exists (`.cursor-pointer { ... }`) but doesn't check if the class is in the HTML.

**Evidence:**
- HTML shows `cursor-pointer` in class attribute on both 6006 and 6007
- DX tool reports class NOT found on both servers
- User confirms class works on 6006 but not 6007

**Workaround:** Manually inspect HTML in browser dev tools.

**Fix needed:** Update DX tool to check HTML classes OR improve CSS rule extraction for Tailwind v4.

---

## Configuration Reference

### Working Configuration (Storybook - Port 6006)

**package.json:**
```json
{
  "dependencies": {
    "tailwindcss": "^4.1.9",
    "@tailwindcss/postcss": "^4.1.9",
    "postcss": "^8.5.1"
  }
}
```

**histoire.config.ts:**
```typescript
import tailwindcss from '@tailwindcss/postcss'

export default defineConfig({
  vite: {
    css: {
      postcss: {
        plugins: [tailwindcss()],
      },
    },
  },
})
```

**histoire.setup.ts:**
```typescript
import './styles/tokens.css'
```

**styles/tokens.css:**
```css
@import "tailwindcss";

@theme {
  /* design tokens */
}
```

### Broken Configuration (Commercial - Port 6007)

**package.json:**
```json
{
  "dependencies": {
    "@tailwindcss/vite": "^4.1.14",
    "tailwindcss": "^4.1.14"
  }
}
```

**histoire.config.ts:**
```typescript
// No PostCSS plugin
export default defineConfig({
  vite: {
    plugins: [vue()],
  },
})
```

**histoire.setup.ts:**
```typescript
import './app/assets/css/histoire.css'
```

**app/assets/css/histoire.css:**
```css
@import "rbee-storybook/styles/tokens.css";
```

**Result:** NO CSS loads.

---

## Verification Checklist

Before marking this issue as resolved:

- [ ] Commercial Histoire shows styled components (colors, spacing, layout)
- [ ] `cursor-pointer` class works on buttons (cursor changes to pointer on hover)
- [ ] All Tailwind classes from storybook components are available
- [ ] Hot reload works when storybook CSS changes
- [ ] No duplicate CSS (check file size)
- [ ] No console errors in browser
- [ ] DX tool reports class found (after DX tool is fixed)

---

## Lessons Learned

1. **Workspace package CSS imports don't work in Histoire** - Need alternative approach
2. **Tailwind v4 doesn't scan workspace dependencies** - Known limitation
3. **DX tool CSS extraction is broken** - Can't trust it for verification
4. **Always test in browser before declaring success** - Visual verification is critical

---

## Summary

**What works:**
- ✅ Storybook (6006) has complete CSS
- ✅ Component imports fixed (no more barrel imports)
- ✅ Missing `computed` import added

**What's broken:**
- ❌ Commercial Histoire (6007) has NO CSS at all
- ❌ Importing workspace package CSS doesn't work
- ❌ DX tool CSS verification is broken

**Blocking issue:**
- Cannot proceed without CSS loading in Histoire
- Need to find working approach for CSS in commercial Histoire

**Next team:** Try the alternative solutions in Priority 2, verify with browser dev tools, then test with DX tool once it's fixed.

---

**TEAM-DX-006 signing off. Sorry we couldn't solve it. Good luck!**
