# TEAM-352: Rule Zero Compliance Fixes

**Date:** Oct 30, 2025  
**Issue:** Original migration steps violated Rule Zero by creating backward compatibility wrappers  
**Status:** ✅ FIXED

---

## Rule Zero Refresher

**BREAKING CHANGES > BACKWARDS COMPATIBILITY**

Pre-1.0 software is ALLOWED to break. The compiler will catch breaking changes. Entropy from "backwards compatibility" functions is PERMANENT TECHNICAL DEBT.

❌ **BANNED:**
- Creating wrapper functions to avoid breaking existing imports
- Re-exporting from shared packages "for backward compatibility"
- Keeping old files "just in case"

✅ **REQUIRED:**
- Update existing files to import directly from shared packages
- Let the compiler find all call sites
- Fix compilation errors
- Delete deprecated code immediately

---

## Violations Found & Fixed

### Violation 1: SDK Loader Wrapper

**Location:** TEAM_352_STEP_1_SDK_LOADER_MIGRATION.md

**WRONG (backward compatibility wrapper):**
```typescript
// loader.ts
export const queenSDKLoader = createSDKLoader({ ... })
export const loadSDKOnce = queenSDKLoader.loadOnce  // ❌ WRAPPER
```

**CORRECT (delete file, update imports):**
```typescript
// loader.ts - DELETED/MINIMAL
export type { RbeeSDK } from './types'
// DO NOT create wrapper exports
```

```typescript
// hooks/useRbeeSDK.ts - Import directly
import { createSDKLoader } from '@rbee/sdk-loader'

const queenSDKLoader = createSDKLoader({ ... })
// Use directly, no wrapper needed
```

**Fix:**
- ✅ Removed wrapper export from loader.ts
- ✅ Updated useRbeeSDK.ts to import directly from @rbee/sdk-loader
- ✅ Removed export from index.ts

---

### Violation 2: Narration Bridge Wrapper

**Location:** TEAM_352_STEP_3_NARRATION_MIGRATION.md

**WRONG (backward compatibility wrapper):**
```typescript
// narrationBridge.ts
export function createNarrationStreamHandler(...) {
  return createStreamHandler(SERVICES.queen, ...)  // ❌ WRAPPER
}

// Re-export for backward compatibility  // ❌ LITERALLY SAYS IT
export { parseNarrationLine } from '@rbee/narration-client'
```

**CORRECT (delete file, update imports):**
```typescript
// narrationBridge.ts - DELETED/MINIMAL
export type { BackendNarrationEvent as NarrationEvent } from '@rbee/narration-client'
// DO NOT create wrapper functions
```

```typescript
// hooks/useRhaiScripts.ts - Import directly
import { createStreamHandler, SERVICES } from '@rbee/narration-client'

const narrationHandler = createStreamHandler(SERVICES.queen, onLocal, {
  debug: import.meta.env.DEV,
  silent: false,
  validate: true,
})
```

**Fix:**
- ✅ Removed wrapper function from narrationBridge.ts
- ✅ Removed "backward compatibility" re-export
- ✅ Updated useRhaiScripts.ts to import directly from @rbee/narration-client

---

## Why This Matters

### The Problem with Wrappers

**Entropy accumulates:**
```typescript
// Year 1: Original implementation
export function loadSDK() { /* 120 LOC */ }

// Year 2: WRONG - Create wrapper for "backward compatibility"
export const loadSDKv2 = sharedLoader.load  // ❌ WRAPPER
export function loadSDK() { /* old 120 LOC */ }  // ❌ KEEP OLD

// Year 3: Another wrapper
export const loadSDKv3 = betterLoader.load  // ❌ ANOTHER WRAPPER
export const loadSDKv2 = sharedLoader.load  // ❌ STILL HERE
export function loadSDK() { /* old 120 LOC */ }  // ❌ STILL HERE

// Result: 3 ways to do the same thing, bugs in all 3 places
```

**Correct approach:**
```typescript
// Year 1: Original implementation
export function loadSDK() { /* 120 LOC */ }

// Year 2: CORRECT - Just update it
import { load } from '@rbee/sdk-loader'
export const loadSDK = load  // Compiler finds all call sites, fix them

// Year 3: CORRECT - Update again if needed
import { betterLoad } from '@rbee/better-loader'
export const loadSDK = betterLoad  // Compiler finds call sites again
```

### Real Impact

**With wrappers (WRONG):**
- Bug in shared package → Must also fix wrapper → Must also update docs
- New developer → Confused by 3 different APIs
- Maintenance → 3× the work
- Tests → 3× the test suites
- **PERMANENT TECHNICAL DEBT**

**Without wrappers (CORRECT):**
- Bug in shared package → Fix once, done
- New developer → One clear API
- Maintenance → 1× the work
- Tests → Shared package already tested
- **NO TECHNICAL DEBT**

---

## What Changed in Migration Steps

### Step 1: SDK Loader Migration

**Before fix:**
- Created `loader.ts` with wrapper export
- Hooks imported from `../loader`
- Created "backward compatibility"

**After fix:**
- `loader.ts` is minimal (only types)
- Hooks import directly from `@rbee/sdk-loader`
- NO wrappers, NO backward compatibility

### Step 2: Hooks Migration

**Before fix:**
- Used wrapper `createNarrationStreamHandler` from `../utils/narrationBridge`

**After fix:**
- Imports directly from `@rbee/narration-client`
- Uses `createStreamHandler(SERVICES.queen, ...)` directly
- NO wrappers

### Step 3: Narration Migration

**Before fix:**
- Created wrapper function `createNarrationStreamHandler`
- Re-exported `parseNarrationLine` "for backward compatibility"

**After fix:**
- `narrationBridge.ts` is minimal (only type re-export)
- Hooks import directly from `@rbee/narration-client`
- NO wrappers, NO re-exports

---

## Verification Checklist

After TEAM-352 completes, verify:

- [ ] `loader.ts` does NOT export wrapper functions
- [ ] `narrationBridge.ts` does NOT export wrapper functions
- [ ] `useRbeeSDK.ts` imports directly from `@rbee/sdk-loader`
- [ ] `useRhaiScripts.ts` imports directly from `@rbee/narration-client`
- [ ] No files say "backward compatibility"
- [ ] No files have wrapper functions that just call shared packages
- [ ] index.ts does NOT export `loader` or `narrationBridge` modules

**If ANY wrapper exists: DELETE IT and fix the imports.**

---

## Pattern for Future Teams

**When migrating code to shared packages:**

1. ❌ **DON'T** create wrapper file that re-exports shared package
2. ✅ **DO** update imports to use shared package directly
3. ❌ **DON'T** keep old file "for backward compatibility"
4. ✅ **DO** delete old file or make it minimal (types only)
5. ❌ **DON'T** create `function_v2()` alongside `function()`
6. ✅ **DO** update `function()` and fix all call sites

**Remember:** Compiler errors are TEMPORARY. Technical debt is PERMANENT.

---

## Summary

**Violations fixed:** 2 major (SDK loader wrapper, narration bridge wrapper)

**Files updated:**
- TEAM_352_STEP_1_SDK_LOADER_MIGRATION.md
- TEAM_352_STEP_2_HOOKS_MIGRATION.md
- TEAM_352_STEP_3_NARRATION_MIGRATION.md

**Key changes:**
- ✅ Removed all wrapper exports
- ✅ Updated hooks to import directly from shared packages
- ✅ Deleted/minimized bridge files
- ✅ NO backward compatibility code

**Result:** Migration now follows Rule Zero correctly. No wrappers, no entropy, no technical debt.

---

**TEAM-352: Rule Zero compliant!** ✅
