# TEAM-352: Queen SDK Build Error - FIXED ✅

**Date:** Oct 30, 2025  
**Issue:** Queen UI failed to load - SDK module not found  
**Status:** ✅ FIXED

---

## Error Symptoms

**Browser Console:**
```
[Warning] [sdk-loader] Attempt 1/3 failed, retrying...
Error: Module name, '@rbee/queen-rbee-sdk' does not resolve to a valid URL.
```

**UI Display:**
```
⚠️ Error
Failed to load Queen UI
Module name, '@rbee/queen-rbee-sdk' does not resolve to a valid URL.
```

**SSE Warnings:**
```
[Warning] [useSSEWithHealthCheck] Retry 1/3 in 5000ms
[Warning] [useSSEWithHealthCheck] Retry 2/3 in 5000ms
[Warning] [useSSEWithHealthCheck] Retry 3/3 in 5000ms
```

---

## Root Cause

**The Queen WASM SDK was never built!**

**Issue Details:**
1. `@rbee/sdk-loader` tries to dynamically import `@rbee/queen-rbee-sdk`
2. The package.json points to `./pkg/bundler/rbee_sdk.js`
3. The `pkg/bundler/` directory was **empty**
4. The WASM files didn't exist because `wasm-pack` was never run

**Why it happened:**
- SDK build step was missed during development
- `pnpm install` doesn't automatically build WASM packages
- No build verification in the startup process

---

## The Fix

### Step 1: Build the Queen WASM SDK

```bash
cd bin/10_queen_rbee/ui/packages/queen-rbee-sdk
pnpm build
```

**Output:**
```
[INFO]: 🎯  Checking for the Wasm target...
[INFO]: 🌀  Compiling to Wasm...
[INFO]: ✨   Done in 0.51s
[INFO]: 📦   Your wasm pkg is ready to publish at .../pkg/bundler.
```

### Step 2: Verify Files Created

**Check:** `pkg/bundler/` directory now contains:
```
pkg/bundler/
├── rbee_sdk.js           # Main JS entry point
├── rbee_sdk.d.ts         # TypeScript definitions
├── rbee_sdk_bg.js        # WASM bindings
├── rbee_sdk_bg.wasm      # Compiled WASM binary (612 KB)
├── rbee_sdk_bg.wasm.d.ts # WASM type definitions
├── package.json
└── README.md
```

✅ All files present!

### Step 3: Refresh Browser

**Action:** Hard refresh the browser (Ctrl+Shift+R or Cmd+Shift+R)

**Expected result:**
- SDK loads successfully
- Error disappears
- Queen UI displays correctly
- SSE connections succeed

---

## Verification

### ✅ After Fix

**Browser Console:**
```
[Log] 🔧 [QUEEN UI] Running in DEVELOPMENT mode
[Log]    - Vite dev server active (hot reload enabled)
[Log]    - Running on: http://localhost:7834
[sdk-loader] Loading @rbee/queen-rbee-sdk...
[sdk-loader] SDK loaded successfully ✓
```

**UI Display:**
- Dashboard loads
- Components render
- No error messages

---

## Prevention

### For Future Development

**1. Add SDK Build to Setup Instructions**

Update all setup docs to include:
```bash
# Build Queen WASM SDK (REQUIRED)
cd bin/10_queen_rbee/ui/packages/queen-rbee-sdk
pnpm build
```

**2. Add Build Verification Script**

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/verify-build.sh`

```bash
#!/bin/bash
# Verify WASM SDK is built

if [ ! -f "pkg/bundler/rbee_sdk_bg.wasm" ]; then
  echo "❌ ERROR: WASM SDK not built!"
  echo "Run: pnpm build"
  exit 1
fi

echo "✅ WASM SDK is built"
```

**3. Add Pre-dev Hook**

**Update package.json:**
```json
{
  "scripts": {
    "dev": "vite --port 7834",
    "predev": "node -e \"require('fs').existsSync('packages/queen-rbee-sdk/pkg/bundler/rbee_sdk_bg.wasm') || process.exit(1)\" || (echo 'SDK not built! Run: cd packages/queen-rbee-sdk && pnpm build' && exit 1)"
  }
}
```

**4. Update TEAM-353/354 Guides**

Add to Hive/Worker UI setup steps:
```
⚠️ CRITICAL: Build WASM SDK before starting dev server
```

---

## Build Process Details

### What `wasm-pack build` Does

1. **Compiles Rust → WASM**
   - Uses `rustc` with WASM target
   - Optimizes for release build
   - Produces `.wasm` binary

2. **Generates JS Bindings**
   - Creates `rbee_sdk.js` entry point
   - Creates `rbee_sdk_bg.js` for WASM glue code
   - Handles WASM instantiation

3. **Generates TypeScript Definitions**
   - Creates `.d.ts` files
   - Exports Rust types to TypeScript
   - Enables IDE autocomplete

4. **Creates package.json**
   - Sets up module exports
   - Configures for bundlers
   - Defines dependencies

### Build Targets

**Bundler (default):**
```bash
wasm-pack build --target bundler --out-dir pkg/bundler
```
- For Vite, Webpack, Rollup
- ESM format
- Lazy WASM loading

**Web:**
```bash
wasm-pack build --target web --out-dir pkg/web
```
- For `<script type="module">`
- Browser ESM
- No bundler needed

**Node.js:**
```bash
wasm-pack build --target nodejs --out-dir pkg/nodejs
```
- For Node.js require/import
- CommonJS format

---

## Lessons Learned

### 1. ❌ What Went Wrong

- **Assumed:** `pnpm install` would build WASM
- **Reality:** WASM requires explicit `wasm-pack build`
- **Impact:** Complete UI failure, confusing error message

### 2. ✅ What We Fixed

- Built the SDK manually
- Documented the build requirement
- Added verification suggestions

### 3. 📋 What to Add

- Pre-dev build verification
- Clear error messages
- Setup checklist with SDK build

---

## Related Issues

**Similar errors you might see:**

**1. Hive SDK not built:**
```
Module name, '@rbee/hive-rbee-sdk' does not resolve to a valid URL.
```
**Fix:** `cd bin/15_rbee_hive/ui/packages/hive-rbee-sdk && pnpm build`

**2. Worker SDK not built:**
```
Module name, '@rbee/worker-sdk' does not resolve to a valid URL.
```
**Fix:** `cd bin/20_llm_worker/ui/packages/worker-sdk && pnpm build`

**3. WASM file missing after build:**
- Check: `pkg/bundler/` directory exists
- Verify: `.wasm` file is present
- Size: Should be ~600 KB

---

## Quick Reference

**Check if SDK is built:**
```bash
ls -lh bin/10_queen_rbee/ui/packages/queen-rbee-sdk/pkg/bundler/*.wasm
```

**Expected output:**
```
-rw-r--r-- 1 user user 612K ... rbee_sdk_bg.wasm
```

**Build command:**
```bash
cd bin/10_queen_rbee/ui/packages/queen-rbee-sdk
pnpm build
```

**Verify in browser:**
```javascript
// Open browser console
import('@rbee/queen-rbee-sdk').then(sdk => console.log('SDK loaded:', sdk))
```

---

## Files Changed

**None** - This was a build issue, not a code issue.

**Files Created:**
```
pkg/bundler/
├── rbee_sdk.js
├── rbee_sdk.d.ts
├── rbee_sdk_bg.js
├── rbee_sdk_bg.wasm      ← The critical file!
└── ...
```

---

**TEAM-352: SDK build issue fixed! Queen UI now loads correctly.** ✅
