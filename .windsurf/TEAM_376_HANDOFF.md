# TEAM-376 HANDOFF - UI Fixes & SDK Loading Issue

## ✅ Mission Accomplished

Fixed iframe theme flash, styling inconsistencies, and improved connection status accuracy. **CRITICAL SDK loading issue remains for next team.**

---

## 💾 Code Delivered (TEAM-376)

### 1. **Fixed Iframe Theme Flash**
**Problem:** Iframes always started in light mode, then flashed to dark mode after JS loaded.

**Root Cause:** Theme was applied via JavaScript after page load, not during HTML parsing.

**Solution:** Added inline script to both iframe HTML files that reads theme from `localStorage` and applies it immediately.

**Files Modified:**
- `bin/10_queen_rbee/ui/app/index.html` (lines 8-14)
- `bin/20_rbee_hive/ui/app/index.html` (lines 8-14)

```html
<!-- TEAM-375: Apply theme immediately to prevent flash -->
<script>
  (function() {
    const theme = localStorage.getItem('theme') || 'dark';
    document.documentElement.classList.add(theme);
  })();
</script>
```

**Also Updated:** `frontend/packages/iframe-bridge/src/parentChild.ts` (line 81)
- `receiveThemeChanges()` now saves theme to `localStorage` so next page load picks it up

**Result:** ✅ Iframes now start in correct theme with no flash

---

### 2. **Fixed CardTitle Color**
**Problem:** Card titles were black in dark mode (no text color specified).

**Root Cause:** `CardTitle` component had no color class.

**Solution:** Added `text-card-foreground` to CardTitle component.

**File Modified:** `frontend/packages/rbee-ui/src/atoms/Card/Card.tsx` (line 48)
```typescript
// OLD:
className={cn("leading-tight font-semibold font-serif", className)}

// NEW:
className={cn("leading-tight font-semibold font-serif text-card-foreground", className)}
```

**Result:** ✅ All card titles now white in dark mode

---

### 3. **Fixed Root Text Color Inheritance**
**Problem:** Components had to manually specify `text-foreground` because root didn't set it.

**Root Cause:** App root divs had `bg-background` but no `text-foreground`.

**Solution:** Added `text-foreground` to root divs in both Queen and Hive.

**Files Modified:**
- `bin/10_queen_rbee/ui/app/src/App.tsx` (line 66)
- `bin/20_rbee_hive/ui/app/src/App.tsx` (line 162)

```typescript
// OLD:
<div className="min-h-screen bg-background font-sans">

// NEW:
<div className="min-h-screen bg-background text-foreground font-sans">
```

**Also Cleaned Up:** Removed redundant `text-foreground` from h1 elements (now inherit from root).

**Result:** ✅ All text now inherits white color in dark mode from root

---

### 4. **Fixed Hive Missing Component Styles**
**Problem:** Hive UI cards had no styling - just plain boxes with borders.

**Root Cause:** `bin/20_rbee_hive/ui/app/src/main.tsx` was missing `import '@rbee/ui/styles.css'`

**Solution:** Added the import to match Queen's pattern.

**File Modified:** `bin/20_rbee_hive/ui/app/src/main.tsx` (line 6)
```typescript
import './index.css'
import '@rbee/ui/styles.css'  // ← ADDED
import App from './App.tsx'
```

**Result:** ✅ Hive cards now have proper styling

---

### 5. **Fixed Connection Status Accuracy**
**Problem:** Queen UI showed "Connected" even when SDK wasn't working.

**Root Cause:** `connected` state was set to `true` when SSE connection opened, not when data was received.

**Solution:** Only set `connected=true` after receiving first heartbeat event (proves Queen is actually working).

**File Modified:** `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/hooks/useHeartbeat.ts`
- Lines 75-76: Added `hasReceivedDataRef` to track if data received
- Lines 82-84: Removed `setConnected(true)` from `onopen` handler
- Lines 91-95: Set `connected=true` only after first heartbeat event

```typescript
// TEAM-375: Track if we've received any data (not just SSE connection open)
const hasReceivedDataRef = React.useRef(false);

eventSource.onopen = () => {
  // TEAM-375: Don't set connected=true until we receive actual data
  setLoading(false);
  setError(null);
};

eventSource.addEventListener('heartbeat', (event) => {
  // TEAM-375: Mark that we've received data (Queen is actually working)
  if (!hasReceivedDataRef.current) {
    hasReceivedDataRef.current = true;
    setConnected(true);  // ← NOW shows Queen is operational, not just SSE open
  }
  // ... rest of handler
});
```

**Result:** ✅ "Connected" now means "Queen is sending heartbeats", not just "SSE handshake complete"

---

## 🚨 CRITICAL ISSUE FOR NEXT TEAM

### Queen SDK Not Resolving

**Error in Browser Console:**
```
[Warning] [sdk-loader] Attempt 1/3 failed, retrying in 496ms:
"Module name, '@rbee/queen-rbee-sdk' does not resolve to a valid URL."

[Warning] [sdk-loader] Attempt 2/3 failed, retrying in 876ms:
"Module name, '@rbee/queen-rbee-sdk' does not resolve to a valid URL."
```

**Also Seeing:**
```
[Warning] [TAURI] Couldn't find callback id 641076259.
This might happen when the app is reloaded while Rust is running an asynchronous operation.
```

---

## 🔍 Root Cause Analysis (For Next Team)

### What's Happening:

The Queen UI is trying to load the WASM SDK (`@rbee/queen-rbee-sdk`) but the module resolution is failing.

### Where to Look:

1. **SDK Loader:** `frontend/packages/sdk-loader/src/loader.ts`
   - This is the retry logic you're seeing in the console
   - It's trying to resolve `@rbee/queen-rbee-sdk` as an ES module

2. **Queen React Package:** `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/hooks/useRbeeSDK.ts`
   - This hook calls the SDK loader
   - Check how it's importing the SDK

3. **Vite Configuration:** `bin/10_queen_rbee/ui/app/vite.config.ts`
   - Check if WASM SDK is excluded from pre-bundling
   - Check if there's a resolve alias needed

4. **Import Map:** Check if Queen UI needs an import map in `index.html`
   - Hive has WASM plugins in vite.config.ts (lines 13-14)
   - Queen might be missing these

### Comparison with Hive (Working):

**Hive vite.config.ts:**
```typescript
import wasm from 'vite-plugin-wasm'
import topLevelAwait from 'vite-plugin-top-level-await'

plugins: [
  tailwindcss(),
  wasm(),              // ← WASM support
  topLevelAwait(),     // ← Top-level await support
  react(),
],
optimizeDeps: {
  exclude: ['@rbee/rbee-hive-sdk'],  // ← Don't pre-bundle WASM
},
```

**Queen vite.config.ts:**
```typescript
// Check if these plugins are missing!
```

### Likely Fix:

1. Add WASM plugins to Queen's vite.config.ts
2. Add `vite-plugin-wasm` and `vite-plugin-top-level-await` to Queen's package.json
3. Exclude `@rbee/queen-rbee-sdk` from optimizeDeps

### Files to Check:

- `bin/10_queen_rbee/ui/app/vite.config.ts` - Missing WASM plugins?
- `bin/10_queen_rbee/ui/app/package.json` - Missing WASM plugin dependencies?
- `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/` - Is WASM build output correct?
- `frontend/packages/sdk-loader/src/loader.ts` - Is import path correct?

---

## 📋 TODO for Next Team (Priority Order)

### Priority 1: Fix Queen SDK Loading ✅ FIXED BY TEAM-377
**Symptom:** `"Module name, '@rbee/queen-rbee-sdk' does not resolve to a valid URL."`

**Status:** ✅ **FIXED** - See `.windsurf/TEAM_377_COMPLETE.md`

**Root Cause Found:** Missing `"type": "module"` and `"exports"` field in package.json

**Fix Applied:**
- Added `"type": "module"` to enable ES module recognition
- Added `"exports": { ".": "./pkg/bundler/queen_rbee_sdk.js" }` for proper resolution
- Changed `"files"` from `["pkg/bundler"]` to `["pkg"]`

**Verification:** All automated checks pass (run `.windsurf/TEAM_377_VERIFICATION.sh`)

**Next:** Manual browser testing required

---

### Priority 2: Verify Tauri Callback Warnings
**Symptom:** `"Couldn't find callback id 641076259"`

**Context:** This happens when:
- Page reloads while Tauri command is in flight
- Tauri command completes but UI component is unmounted

**Action:** 
- Check if this is just a dev hot-reload issue (ignore if so)
- If happens in production, add cleanup in useEffect returns

---

### Priority 3: Refactor Hive UI Structure
**Problem:** Everything is in `App.tsx` (10KB file), Queen has proper `pages/` and `components/` structure

**Action:**
- Create `bin/20_rbee_hive/ui/app/src/pages/DashboardPage.tsx`
- Create `bin/20_rbee_hive/ui/app/src/components/` folder
- Extract HeartbeatStatus, ModelManagement, WorkerManagement, SpawnWorker, GPUUtilization into separate components
- Match Queen's structure for consistency

---

## 🎓 Lessons Learned

### 1. **Theme Flash Prevention**
Inline scripts in HTML are sometimes necessary for instant theme application. Don't rely solely on JavaScript for critical visual state.

### 2. **Connection Status Semantics**
"Connected" should mean "service is operational", not "TCP handshake complete". Users care about functionality, not technical connection state.

### 3. **Root CSS Inheritance**
Always set `text-foreground` on the root element alongside `bg-background`. Components should inherit, not specify.

### 4. **Component Library Consistency**
All components should have proper default colors. `CardTitle` should have had `text-card-foreground` from the start.

### 5. **WASM in Vite**
WASM modules need special handling:
- `vite-plugin-wasm` for WASM support
- `vite-plugin-top-level-await` for async WASM init
- Exclude from `optimizeDeps` to prevent pre-bundling

---

## ✅ Verification Checklist

- [x] Iframe theme flash fixed (Queen & Hive)
- [x] CardTitle color fixed (white in dark mode)
- [x] Root text color inheritance fixed (Queen & Hive)
- [x] Hive component styles loaded
- [x] Connection status accuracy improved
- [x] **Queen SDK loading** ✅ FIXED BY TEAM-377
- [ ] Hive UI refactored to match Queen structure ⚠️ FUTURE TEAM

---

## 📊 Files Modified Summary

**Frontend Packages:**
- `frontend/packages/rbee-ui/src/atoms/Card/Card.tsx` - Added text-card-foreground
- `frontend/packages/iframe-bridge/src/parentChild.ts` - Save theme to localStorage

**Queen UI:**
- `bin/10_queen_rbee/ui/app/index.html` - Inline theme script
- `bin/10_queen_rbee/ui/app/src/App.tsx` - Added text-foreground to root
- `bin/10_queen_rbee/ui/app/src/pages/DashboardPage.tsx` - Removed redundant text-foreground
- `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/hooks/useHeartbeat.ts` - Fixed connection status

**Hive UI:**
- `bin/20_rbee_hive/ui/app/index.html` - Inline theme script
- `bin/20_rbee_hive/ui/app/src/main.tsx` - Added @rbee/ui/styles.css import
- `bin/20_rbee_hive/ui/app/src/App.tsx` - Added text-foreground to root, removed redundant from h1

---

**TEAM-376 complete. SDK loading issue handed off to next team with detailed analysis.**

**Total LOC:** ~50 lines modified, 0 files created  
**Build status:** ✅ All passing  
**Visual bugs:** ✅ Fixed  
**SDK loading:** ⚠️ Next team priority
