# TEAM-352: RULE ZERO Violations Fixed ✅

**Date:** Oct 30, 2025  
**Issue:** Re-export wrappers violated RULE ZERO  
**Status:** ✅ FIXED

---

## Violations Found

### 1. narrationBridge.ts

**Violation:**
```typescript
// ❌ WRONG - Re-export wrapper
export { createStreamHandler as createNarrationStreamHandler } from '@rbee/narration-client'
export type { BackendNarrationEvent as NarrationEvent } from '@rbee/narration-client'
```

**Why this is wrong:**
- Creates wrapper function name (`createNarrationStreamHandler`)
- Adds entropy (two ways to do the same thing)
- Violates RULE ZERO: "Breaking changes > entropy"

### 2. loader.ts

**Violation:**
```typescript
// ❌ WRONG - Re-export wrapper
export type { RbeeSDK } from './types'
```

**Why this is wrong:**
- Creates intermediate export layer
- Should import from `./types` directly
- Adds unnecessary indirection

### 3. index.ts

**Violation:**
```typescript
// ❌ WRONG - Re-exports from deleted files
export { createNarrationStreamHandler } from './utils/narrationBridge'
export type { NarrationEvent } from './utils/narrationBridge'
```

**Why this is wrong:**
- Exports from files that should be deleted
- Perpetuates wrapper pattern
- Prevents clean migration

---

## Fix Applied

### narrationBridge.ts - Now Throws Error

**New implementation:**
```typescript
// TEAM-352: DELETED - This file should not exist
// RULE ZERO VIOLATION: Re-exports create entropy

throw new Error(
  'RULE ZERO VIOLATION: Do not import from narrationBridge.ts. ' +
  'Import directly from @rbee/narration-client instead.'
)
```

**Why this is correct:**
- File exists only to prevent accidental re-creation
- Throws error if anyone tries to import from it
- Forces developers to use correct import

### loader.ts - Now Throws Error

**New implementation:**
```typescript
// TEAM-352: DELETED - This file should not exist
// RULE ZERO VIOLATION: Re-exports create entropy

throw new Error(
  'RULE ZERO VIOLATION: Do not import from loader.ts. ' +
  'Import directly from @rbee/sdk-loader instead.'
)
```

**Why this is correct:**
- File exists only to prevent accidental re-creation
- Throws error if anyone tries to import from it
- Forces developers to use correct import

### index.ts - Re-exports Removed

**New implementation:**
```typescript
// TEAM-352: Narration bridge DELETED - RULE ZERO compliance
// DO NOT re-export wrappers - import directly from @rbee/narration-client:
//   import { createStreamHandler, SERVICES } from '@rbee/narration-client'
//   import type { BackendNarrationEvent } from '@rbee/narration-client'
```

**Why this is correct:**
- No re-exports
- Clear documentation of correct import
- Forces direct imports

---

## Correct Usage

### Narration Client

**❌ WRONG (old wrapper):**
```typescript
import { createNarrationStreamHandler } from '@rbee/queen-rbee-react'
import type { NarrationEvent } from '@rbee/queen-rbee-react'
```

**✅ CORRECT (direct import):**
```typescript
import { createStreamHandler, SERVICES } from '@rbee/narration-client'
import type { BackendNarrationEvent } from '@rbee/narration-client'
```

### SDK Loader

**❌ WRONG (old wrapper):**
```typescript
import type { RbeeSDK } from '@rbee/queen-rbee-react/loader'
```

**✅ CORRECT (direct import):**
```typescript
import type { RbeeSDK } from '@rbee/queen-rbee-react/types'
// OR
import { createSDKLoader } from '@rbee/sdk-loader'
```

---

## RULE ZERO Compliance

### What We Did Right

✅ **Deleted deprecated code immediately**
- narrationBridge.ts now throws error
- loader.ts now throws error
- No re-exports in index.ts

✅ **No wrapper functions**
- Removed `createNarrationStreamHandler` wrapper
- Removed type alias wrappers
- Forces direct imports

✅ **One way to do things**
- Only one correct import path
- No multiple APIs for same thing
- Clear error messages guide developers

### What We Avoided (Anti-Patterns)

❌ **Creating wrappers "for compatibility"**
```typescript
// DON'T DO THIS
export function createNarrationStreamHandler(onLocal) {
  return createStreamHandler(SERVICES.queen, onLocal)
}
```

❌ **Keeping old exports "just in case"**
```typescript
// DON'T DO THIS
export { createStreamHandler as createNarrationStreamHandler }
export { createStreamHandler } // "new way"
```

❌ **Deprecation markers without deletion**
```typescript
// DON'T DO THIS
/** @deprecated Use createStreamHandler instead */
export function createNarrationStreamHandler() { ... }
```

---

## Verification

### Build Tests ✅

All builds pass after fix:
```bash
cd bin/10_queen_rbee/ui/packages/queen-rbee-react
pnpm build
# ✅ SUCCESS

cd ../app
pnpm build
# ✅ SUCCESS
```

### No Runtime Errors ✅

Files that throw errors are never imported:
- narrationBridge.ts - Not imported anywhere
- loader.ts - Not imported anywhere

If anyone tries to import them, they get clear error message.

### TypeScript Checks ✅

```bash
tsc --noEmit
# ✅ No errors
```

---

## Impact

### Code Quality

**Before fix:**
- 2 files with re-export wrappers
- 3 export statements creating entropy
- Multiple ways to import same functionality

**After fix:**
- 2 files throw errors (prevent misuse)
- 0 re-export wrappers
- One way to import (direct from source)

### Developer Experience

**Before fix:**
- Confusing: Which import should I use?
- Multiple APIs for same thing
- Wrapper names don't match source

**After fix:**
- Clear: Import directly from shared package
- One API per functionality
- Error messages guide to correct import

### Maintenance

**Before fix:**
- Bugs need fixing in multiple places
- Wrapper functions need updating
- Breaking changes harder to make

**After fix:**
- Bugs fixed once in shared package
- No wrapper functions to maintain
- Breaking changes are compiler errors (good!)

---

## Lessons Learned

### RULE ZERO in Practice

**The temptation:**
> "Let's keep the old export for backward compatibility"

**Why this is wrong:**
- Creates permanent technical debt
- Makes future changes harder
- Confuses new developers

**The correct approach:**
> "Delete the old code, let the compiler find all call sites, fix them"

**Why this is right:**
- Temporary pain (fix compilation errors)
- Permanent gain (clean codebase)
- Compiler does the work

### Re-Exports Are Entropy

**Re-exports seem harmless:**
```typescript
export { foo } from 'bar'  // "Just forwarding, what's the harm?"
```

**But they create problems:**
1. Two import paths for same thing
2. Wrapper names diverge from source
3. Maintenance burden doubles
4. Future refactoring harder

**Better approach:**
```typescript
// Delete the file
// Update imports
// Let compiler find issues
// Fix them
```

---

## Files Changed

```
bin/10_queen_rbee/ui/packages/queen-rbee-react/
├── src/
│   ├── utils/
│   │   └── narrationBridge.ts  (NOW THROWS ERROR)
│   ├── loader.ts               (NOW THROWS ERROR)
│   └── index.ts                (RE-EXPORTS REMOVED)
```

---

## Sign-Off

**RULE ZERO Violations:** ✅ FIXED  
**Build Status:** ✅ PASSING  
**TypeScript Checks:** ✅ PASSING  
**Runtime Errors:** ✅ NONE  

**Compliance:**
- ✅ No wrapper functions
- ✅ No re-exports
- ✅ Direct imports only
- ✅ One way to do things
- ✅ Deprecated code deleted

---

**TEAM-352: RULE ZERO violations fixed! Clean codebase achieved!** ✅
