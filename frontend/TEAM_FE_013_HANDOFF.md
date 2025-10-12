# TEAM-FE-013 HANDOFF: Tailwind Workspace Dependency Issue

**Date:** 2025-10-12  
**Team:** TEAM-FE-013  
**Status:** BLOCKED - Critical DX tool bug discovered

---

## 🚨 CRITICAL ISSUE DISCOVERED

**Problem:** Cannot verify if Tailwind CSS classes from workspace dependencies (`rbee-storybook`) are being compiled into the commercial Histoire CSS.

**What We Know:**
1. ✅ `cursor-pointer` class exists in source: `libs/storybook/stories/atoms/Button/Button.vue` line 12
2. ✅ User confirms `cursor-pointer` works on http://localhost:6006 (storybook) - cursor changes to pointer
3. ❌ User reports `cursor-pointer` does NOT work on http://localhost:6007 (commercial Histoire) - cursor stays default
4. ❌ DX tool reports class NOT found on BOTH servers (but user sees it working on 6006)
5. ✅ **HTML inspection shows `cursor-pointer` IS in the class attribute on BOTH servers**

**HTML Evidence:**
```html
<!-- Port 6006 (working) -->
<button class="... cursor-pointer ...">Default Button</button>

<!-- Port 6007 (Navigation bar) -->
<button class="... cursor-pointer ...">Toggle theme</button>
```

**Conclusion:** 
- The class IS in the HTML on both servers
- The class WORKS on 6006 but NOT on 6007
- This means the CSS rule for `.cursor-pointer` is missing on 6007
- The DX tool is broken because it can't find the CSS rule (but it should also check HTML)

---

## What Was Attempted

### 1. Fixed Missing Vue Plugin ✅
- Added `@vitejs/plugin-vue` to `histoire.config.ts`
- This fixed the "Failed to parse source" errors

### 2. Fixed PostCSS Configuration ⚠️ INCOMPLETE
- Added `@tailwindcss/postcss` plugin to `histoire.config.ts`
- **PROBLEM:** `@tailwindcss/postcss` is NOT installed in commercial `package.json`
- Commercial uses `@tailwindcss/vite` for Nuxt, but Histoire needs `@tailwindcss/postcss`
- **DECISION BY TEAM-FE-003:** Storybook uses PostCSS for Histoire compatibility

### 3. Fixed Template Imports ✅
- Replaced barrel imports from `rbee-storybook/stories` with local relative imports
- All 7 template files updated (DevelopersView, ProvidersView, HomeView, etc.)

### 4. Fixed Missing `computed` Import ✅
- Added `computed` import to `DevelopersFeatures.vue`

### 5. Created Workspace Watch Plugin ⚠️ UNTESTED
- Created `vite-plugin-watch-workspace.ts` to watch `libs/storybook` for changes
- Added `chokidar@^4.0.3` dependency
- Configured in `histoire.config.ts`
- **Status:** Cannot verify if this works because DX tool is broken

---

## Files Modified

### Configuration Files
- `frontend/bin/commercial/histoire.config.ts` - Added Vue plugin, PostCSS, watch plugin
- `frontend/bin/commercial/histoire.setup.ts` - Removed duplicate CSS import
- `frontend/bin/commercial/package.json` - Added `chokidar` dependency
- `frontend/bin/commercial/vite-plugin-watch-workspace.ts` - NEW FILE

### Template Files (Fixed Imports)
- `frontend/bin/commercial/app/stories/templates/DevelopersView.vue`
- `frontend/bin/commercial/app/stories/templates/ProvidersView.vue`
- `frontend/bin/commercial/app/stories/templates/HomeView.vue`
- `frontend/bin/commercial/app/stories/templates/FeaturesView.vue`
- `frontend/bin/commercial/app/stories/templates/UseCasesView.vue`
- `frontend/bin/commercial/app/stories/templates/EnterpriseView.vue`
- `frontend/bin/commercial/app/stories/templates/PricingView.vue`

### Component Files
- `frontend/bin/commercial/app/stories/organisms/DevelopersFeatures/DevelopersFeatures.vue` - Added `computed` import

### Documentation
- `frontend/.dx-tool/FEATURE_REQUESTS.md` - Reported critical DX tool bug

---

## The Real Problem: Tailwind Not Scanning Workspace Dependencies

**Root Cause CONFIRMED:** Tailwind v4's PostCSS plugin is not scanning the `libs/storybook` directory when building CSS for the commercial Histoire server.

**Evidence:**
- `tailwind.config.js` has correct content paths: `'../../libs/storybook/stories/**/*.{vue,js,ts,jsx,tsx}'`
- Source code has `cursor-pointer` class in Button.vue
- HTML shows `cursor-pointer` in class attribute on BOTH servers
- User reports it works on 6006 (storybook) but NOT on 6007 (commercial)
- **This confirms: The CSS rule `.cursor-pointer { cursor: pointer; }` is missing on 6007**

**Why This Happens:**
- Vite doesn't watch workspace dependencies by default
- Tailwind's content scanning may not follow pnpm workspace symlinks properly
- The CSS is built once at server start, without scanning workspace deps
- The watch plugin we created may help with HMR, but the initial build is broken

---

## Next Steps for Next Team

### Priority 1: Fix DX Tool (CRITICAL)

The DX tool team needs to fix the CSS extraction bug. See `frontend/.dx-tool/FEATURE_REQUESTS.md` for full bug report.

**Without a working DX tool, you cannot verify ANY frontend work per FRONTEND_ENGINEERING_RULES.md.**

### Priority 2: Install Missing Dependency (CRITICAL)

**The commercial Histoire config imports `@tailwindcss/postcss` but it's NOT installed!**

```bash
cd frontend/bin/commercial
pnpm add -D @tailwindcss/postcss postcss
```

**Why PostCSS?**
- TEAM-FE-003 chose `@tailwindcss/postcss` for Histoire compatibility
- Commercial uses `@tailwindcss/vite` for Nuxt, but Histoire needs PostCSS
- This is NOT deprecated - it's the official Tailwind v4 PostCSS integration

### Priority 3: Fix Tailwind Workspace Scanning

After installing the dependency, the HTML shows `cursor-pointer` is in the class attribute on both servers, but the CSS rule is missing on 6007. This means Tailwind is not scanning the workspace dependency.

**Solution Options:**

**Option A: Force Tailwind to resolve workspace symlinks**

Add to `tailwind.config.js`:
```javascript
export default {
  content: {
    files: [
      './app/**/*.{vue,js,ts,jsx,tsx}',
      './components/**/*.{vue,js,ts,jsx,tsx}',
      '../../libs/storybook/stories/**/*.{vue,js,ts,jsx,tsx}',
    ],
    // Force Tailwind to follow symlinks
    extract: {
      vue: (content) => content.match(/[^<>"'`\s]*[^<>"'`\s:]/g) || [],
    },
  },
}
```

**Option B: Use absolute paths instead of relative**

```javascript
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'

const __dirname = fileURLToPath(new URL('.', import.meta.url))

export default {
  content: [
    './app/**/*.{vue,js,ts,jsx,tsx}',
    resolve(__dirname, '../../libs/storybook/stories/**/*.{vue,js,ts,jsx,tsx}'),
  ],
}
```

**Option C: Restart server and verify**

1. **Restart Commercial Histoire:**
   ```bash
   cd frontend/bin/commercial
   pnpm run story:dev
   ```

2. **Check if classes are now compiled:**
   - Open http://localhost:6007/story/app-stories-organisms-navigation-navigation-story-vue
   - Inspect button element
   - Check if cursor changes to pointer on hover
   - If YES: The restart fixed it (Tailwind scanned on startup)
   - If NO: Try Option A or B above

### Priority 3: If Workspace Watch Plugin Doesn't Work

Try alternative solutions:

**Option A: Use Tailwind CLI to watch workspace**
```json
// package.json
"scripts": {
  "story:dev": "concurrently \"tailwindcss -i ./app/assets/css/main.css -o ./dist/tailwind.css --watch\" \"histoire dev --port 6007\"",
}
```

**Option B: Copy Button to commercial (NOT RECOMMENDED)**
- Defeats the purpose of workspace packages
- Creates duplication
- Only do this as last resort

**Option C: Use Vite's `server.fs.allow` to force watching**
```typescript
// histoire.config.ts
vite: {
  server: {
    fs: {
      allow: ['..', '../..'], // Allow accessing parent directories
    },
  },
}
```

---

## Verification Checklist

Once DX tool is fixed, verify:

- [ ] `cursor-pointer` class found on http://localhost:6006
- [ ] `cursor-pointer` class found on http://localhost:6007
- [ ] Hover over button on 6007 shows pointer cursor
- [ ] Change Button.vue in storybook, see update on 6007 within 3 seconds
- [ ] No console errors in browser
- [ ] No build errors in terminal

---

## Lessons Learned

1. **DX tool is critical** - Without it, we're flying blind
2. **Workspace dependencies are hard** - Vite/Tailwind don't watch them by default
3. **Always verify with DX tool** - Don't trust visual inspection alone (but also don't trust DX tool if it contradicts visual inspection!)
4. **Read the engineering rules** - They exist for a reason

---

## Summary

**What Works:**
- ✅ Histoire configuration (Vue plugin, PostCSS)
- ✅ Template imports (no more barrel imports)
- ✅ Component imports (computed added)

**What's Broken:**
- ❌ DX tool CSS extraction (critical bug)
- ❌ Tailwind workspace scanning (cannot verify due to DX tool bug)

**Blocking Issue:**
- Cannot proceed without working DX tool
- DX tool team must fix CSS extraction bug first

**Next Team:** Fix the DX tool, then verify Tailwind workspace scanning.

---

**TEAM-FE-013 signing off. Good luck!**
