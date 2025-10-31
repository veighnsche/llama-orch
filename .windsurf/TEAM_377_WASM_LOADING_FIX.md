# TEAM-377 - WASM Loading Fix (Static Import)

## 🐛 The Problem

**Error:**
```
[sdk-loader] Attempt 1/3: Importing @rbee/queen-rbee-sdk
[sdk-loader] Attempt 1/3 failed: "Module name, '@rbee/queen-rbee-sdk' does not resolve to a valid URL."
```

**Root Cause:**
Dynamic import with `@vite-ignore` doesn't work for **workspace packages with WASM files**.

The SDK loader was trying:
```typescript
import(/* @vite-ignore */ '@rbee/queen-rbee-sdk')
```

But Vite can't resolve:
1. The workspace package path
2. The relative WASM import inside the package (`./queen_rbee_sdk_bg.wasm`)

---

## ✅ The Fix

**Use static import instead of dynamic import.**

### Before (BROKEN)

```typescript
// Dynamic import via loader
const queenSDKLoader = createSDKLoader<RbeeSDK>({
  packageName: '@rbee/queen-rbee-sdk',
  requiredExports: ['QueenClient', 'HeartbeatMonitor', 'OperationBuilder', 'RhaiClient'],
  timeout: 15000,
  maxAttempts: 3,
})

export function useQueenSDK() {
  useEffect(() => {
    queenSDKLoader.loadOnce().then(result => {
      setSDK(result.sdk)
    })
  }, [])
}
```

**Problem:** `import('@rbee/queen-rbee-sdk')` fails to resolve.

### After (FIXED)

```typescript
// Static import - Vite can resolve this
import * as QueenSDK from '@rbee/queen-rbee-sdk'

export function useQueenSDK() {
  useEffect(() => {
    // SDK already loaded, just validate and set state
    console.log('[useQueenSDK] Validating SDK exports...', QueenSDK)
    
    if (!QueenSDK.QueenClient || !QueenSDK.HeartbeatMonitor) {
      throw new Error('SDK missing required exports')
    }
    
    console.log('[useQueenSDK] ✅ SDK loaded successfully')
    setSDK(QueenSDK as RbeeSDK)
    setLoading(false)
  }, [])
}
```

**Why this works:** Vite processes static imports at build time and can resolve workspace packages + WASM files.

---

## 📊 Before vs After

### Before (Dynamic Import)

```
Build time:
- Vite sees: import('@rbee/queen-rbee-sdk')
- Vite says: "I don't know what that is" ❌

Runtime:
- Browser tries to fetch: /@rbee/queen-rbee-sdk
- 404 Not Found ❌
```

### After (Static Import)

```
Build time:
- Vite sees: import * as QueenSDK from '@rbee/queen-rbee-sdk'
- Vite resolves: ../packages/queen-rbee-sdk/pkg/bundler/queen_rbee_sdk.js ✅
- Vite bundles: WASM file included ✅

Runtime:
- SDK already in bundle ✅
- WASM already loaded ✅
```

---

## 🎯 Why Dynamic Import Failed

**The generated SDK code:**
```javascript
// pkg/bundler/queen_rbee_sdk.js
import * as wasm from "./queen_rbee_sdk_bg.wasm";
export * from "./queen_rbee_sdk_bg.js";
```

**With dynamic import:**
1. Browser fetches `@rbee/queen-rbee-sdk` (fails - not a URL)
2. Even if it worked, the relative `./queen_rbee_sdk_bg.wasm` import would fail

**With static import:**
1. Vite resolves `@rbee/queen-rbee-sdk` to workspace package at build time
2. Vite processes the WASM import and bundles it
3. Everything works ✅

---

## 🔧 Files Changed

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/hooks/useQueenSDK.ts`

**Changes:**
1. ❌ Removed `createSDKLoader` usage
2. ✅ Added static import: `import * as QueenSDK from '@rbee/queen-rbee-sdk'`
3. ✅ Simplified hook - just validate exports and set state
4. ✅ Added console logging for debugging

---

## 📝 Console Output

**Success:**
```
[useQueenSDK] Validating SDK exports... { QueenClient: ƒ, HeartbeatMonitor: ƒ, ... }
🎉 [Queen SDK] WASM module initialized successfully!
[useQueenSDK] ✅ SDK loaded successfully
```

**Failure:**
```
[useQueenSDK] Validating SDK exports... { }
[useQueenSDK] ❌ SDK load failed: SDK missing required exports
```

---

## 🎓 Lessons Learned

### When to Use Static Import

✅ **Use static import when:**
- Package is in workspace (monorepo)
- Package contains WASM files
- Package is always needed (not code-split)

### When to Use Dynamic Import

✅ **Use dynamic import when:**
- Package is from npm (not workspace)
- Package is large and should be code-split
- Package is conditionally loaded

### For WASM in Workspaces

**Always use static import.** Dynamic import doesn't work because:
1. Vite can't resolve workspace package names at runtime
2. WASM files need to be processed at build time
3. Relative imports inside the package break

---

## ✅ Verification

```bash
# Rebuild
cd bin/10_queen_rbee/ui/packages/queen-rbee-react
pnpm build

# Restart dev server
cd ../app
pnpm dev

# Open browser console
# Should see:
# [useQueenSDK] Validating SDK exports...
# 🎉 [Queen SDK] WASM module initialized successfully!
# [useQueenSDK] ✅ SDK loaded successfully
```

---

**TEAM-377 | WASM loading fixed | Static import FTW! 🎉**
