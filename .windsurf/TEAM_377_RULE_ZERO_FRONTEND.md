# TEAM-377 - Rule Zero Compliance: Frontend Edition

## ❌ RULE ZERO VIOLATION FOUND

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/hooks/useRbeeSDK.ts`

**Line 74:**
```typescript
export const useRbeeSDK = useQueenSDK;  // ❌ DEPRECATED ALIAS
```

**This is entropy.** Creating an alias instead of renaming everywhere.

---

## ✅ PROPER FIX APPLIED

### DELETED the alias, RENAMED everywhere

**Files changed:**

1. **useRbeeSDK.ts** - Deleted the `useRbeeSDK` alias
2. **hooks/index.ts** - Changed export from `useRbeeSDK` to `useQueenSDK`
3. **src/index.ts** - Removed `useRbeeSDK` from exports
4. **useRhaiScripts.ts** - Changed import and usage to `useQueenSDK`

---

## 🔧 Changes Made

### 1. Delete Alias (useRbeeSDK.ts)

**BEFORE:**
```typescript
export function useQueenSDK() {
  // ...
}

// TEAM-377: Backward compatibility alias (deprecated)
// TODO: Remove this after all consumers are updated
export const useRbeeSDK = useQueenSDK;  // ❌ ENTROPY
```

**AFTER:**
```typescript
export function useQueenSDK() {
  // ...
}

// TEAM-377: DELETED useRbeeSDK alias - RULE ZERO violation
// Just renamed to useQueenSDK everywhere, let compiler find call sites
```

### 2. Update Exports (hooks/index.ts)

**BEFORE:**
```typescript
export { useRbeeSDK } from './useRbeeSDK'
```

**AFTER:**
```typescript
export { useQueenSDK } from './useRbeeSDK'
```

### 3. Update Public API (src/index.ts)

**BEFORE:**
```typescript
export { useQueenSDK, useRbeeSDK } from './hooks/useRbeeSDK'
```

**AFTER:**
```typescript
export { useQueenSDK } from './hooks/useRbeeSDK'
```

### 4. Fix Call Sites (useRhaiScripts.ts)

**BEFORE:**
```typescript
import { useRbeeSDK } from './useRbeeSDK'

export function useRhaiScripts() {
  const { sdk } = useRbeeSDK()
```

**AFTER:**
```typescript
import { useQueenSDK } from './useRbeeSDK'

export function useRhaiScripts() {
  const { sdk } = useQueenSDK()
```

---

## 📊 Impact

**Files changed:** 4  
**Lines removed:** ~10 (alias + comments)  
**Lines added:** ~4 (updated imports/exports)  
**Net:** -6 lines, no entropy

**Build:** ✅ SUCCESS

---

## 🎯 Why This Matters

**From RULE ZERO:**
> "Creating `function_v2()`, `function_new()` to avoid breaking `function()` is BANNED"
> "JUST UPDATE THE EXISTING FUNCTION"
> "DELETE deprecated code immediately"

**The alias pattern is the same problem:**
- Creates two ways to do the same thing
- Confuses new contributors ("which one should I use?")
- Permanent maintenance burden
- Never gets removed ("for compatibility")

**The right way:**
- Rename the function
- Update all call sites
- Let TypeScript compiler find any we missed
- Done

---

## ✅ Verification

```bash
# Build succeeds
cd bin/10_queen_rbee/ui
pnpm build:ui
# ✅ SUCCESS

# TypeScript compilation passes
# No errors about missing useRbeeSDK
```

---

**TEAM-377 | Rule Zero compliance | Frontend | No more deprecated aliases! 🎉**
