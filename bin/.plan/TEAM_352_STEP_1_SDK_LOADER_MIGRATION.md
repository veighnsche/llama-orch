# TEAM-352 Step 1: Migrate SDK Loader to @rbee/sdk-loader

**Estimated Time:** 45-60 minutes  
**Priority:** CRITICAL  
**Prerequisites:** TEAM-356 complete  
**Next Step:** TEAM_352_STEP_2_HOOKS_MIGRATION.md

---

## Mission

Replace Queen's custom SDK loader (~140 LOC) with @rbee/sdk-loader package.

**Why This Matters:**
- Removes duplicate code (loader.ts + globalSlot.ts)
- Uses battle-tested retry logic with 34 passing tests
- Enables hot reload (HMR-safe global slot)
- Proves pattern works before Hive/Worker use it

**Code Reduction:** ~140 LOC → ~15 LOC (88% reduction)

---

## Deliverables Checklist

- [ ] Added @rbee/sdk-loader dependency
- [ ] Replaced loader.ts implementation
- [ ] Deleted globalSlot.ts (no longer needed)
- [ ] Updated useRbeeSDK hook to use new loader
- [ ] All imports updated
- [ ] Package builds successfully
- [ ] Tests pass (if any exist)
- [ ] TEAM-352 signatures added

---

## Step 1: Add Package Dependency

Navigate to Queen React package:

```bash
cd bin/10_queen_rbee/ui/packages/queen-rbee-react
```

Edit `package.json`, add to `dependencies` section:

```json
{
  "dependencies": {
    "@rbee/queen-rbee-sdk": "workspace:*",
    "@rbee/sdk-loader": "workspace:*"
  }
}
```

Install dependencies:

```bash
cd ../../../..  # Back to monorepo root
pnpm install
```

**Verification:**
```bash
cd bin/10_queen_rbee/ui/packages/queen-rbee-react
ls -la node_modules/@rbee/sdk-loader  # Should exist
```

---

## Step 2: Read Current Implementation

Before making changes, understand what we're replacing:

```bash
cd src
cat loader.ts    # ~120 LOC - Custom retry logic
cat globalSlot.ts  # ~20 LOC - Singleflight pattern
cat hooks/useRbeeSDK.ts  # Uses loadSDKOnce()
```

**Key observations:**
- `loader.ts` has retry logic (3 attempts)
- `loader.ts` has exponential backoff with jitter
- `loader.ts` validates exports (QueenClient, HeartbeatMonitor, OperationBuilder, RhaiClient)
- `globalSlot.ts` prevents duplicate loads
- `useRbeeSDK.ts` calls `loadSDKOnce()` from loader.ts

---

## Step 3: Replace loader.ts

**CRITICAL:** Back up current file first:

```bash
cp loader.ts loader.ts.backup
```

Replace entire contents of `loader.ts`:

```typescript
// TEAM-352: DELETED - Migrated to @rbee/sdk-loader package
// Old implementation: ~120 LOC of custom retry/backoff/timeout logic
// New: Import directly from @rbee/sdk-loader in hooks that need it
// Reduction: 120 LOC (100%)

// This file is intentionally empty/minimal
// DO NOT create wrapper exports "for backward compatibility"
// UPDATE imports in hooks to use @rbee/sdk-loader directly

export type { RbeeSDK } from './types'
```

**Save file** and verify it compiles:

```bash
pnpm build
```

---

## Step 4: Update types.ts

Check if `types.ts` needs updates:

```bash
cat types.ts
```

**Current types (should be fine):**
```typescript
export interface RbeeSDK {
  QueenClient: any
  HeartbeatMonitor: any
  OperationBuilder: any
  RhaiClient: any
}
```

**If LoadOptions or GlobalSlot are defined here, delete them** (now in @rbee/sdk-loader).

---

## Step 5: Delete globalSlot.ts

This file is no longer needed (singleflight logic now in @rbee/sdk-loader):

```bash
rm globalSlot.ts
```

**Verification:**
```bash
ls -la globalSlot.ts  # Should say "No such file"
```

---

## Step 6: Update utils.ts (if exists)

Check if `utils.ts` has `sleep`, `withTimeout`, or `jitter` functions:

```bash
cat utils.ts
```

**If it has these utilities, delete them:**
- `sleep()` - Now in @rbee/sdk-loader
- `withTimeout()` - Now in @rbee/sdk-loader  
- `addJitter()` - Now in @rbee/sdk-loader

**If utils.ts becomes empty, delete it:**
```bash
rm utils.ts  # Only if empty after removing loader utils
```

---

## Step 7: Update hooks/useRbeeSDK.ts

**CRITICAL:** Update the hook to import directly from @rbee/sdk-loader (NO WRAPPER):

```bash
cat hooks/useRbeeSDK.ts
```

**OLD import (WRONG - uses wrapper):**
```typescript
import { loadSDKOnce } from '../loader'
```

**NEW import (CORRECT - direct from shared package):**

Edit `hooks/useRbeeSDK.ts`:

```typescript
// TEAM-352: Import directly from @rbee/sdk-loader (no wrapper)
import { createSDKLoader } from '@rbee/sdk-loader'
import type { RbeeSDK } from '../types'

const queenSDKLoader = createSDKLoader<RbeeSDK>({
  packageName: '@rbee/queen-rbee-sdk',
  requiredExports: ['QueenClient', 'HeartbeatMonitor', 'OperationBuilder', 'RhaiClient'],
  timeout: 15000,
  maxAttempts: 3,
})

export function useRbeeSDK() {
  const [sdk, setSdk] = useState<RbeeSDK | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    queenSDKLoader.loadOnce()
      .then(result => {
        setSdk(result.sdk)
        setLoading(false)
      })
      .catch(err => {
        setError(err)
        setLoading(false)
      })
  }, [])

  return { sdk, loading, error }
}
```

**This is the CORRECT pattern:**
- ✅ Import directly from @rbee/sdk-loader
- ✅ Create loader in the hook that needs it
- ❌ NO wrapper exports in loader.ts

---

## Step 8: Update index.ts Exports

Check main exports file:

```bash
cat index.ts
```

**Remove exports for deleted/migrated files:**
- ❌ Remove `export * from './loader'` (loader.ts is now minimal, nothing to export)
- ❌ Remove `export * from './globalSlot'` (file deleted)
- ❌ Remove any util exports that were deleted

**Updated index.ts should look like:**
```typescript
// TEAM-352: SDK loader migrated to @rbee/sdk-loader package
// Hooks now import directly from @rbee/sdk-loader - NO WRAPPERS

export * from './types'
export * from './hooks'
export * from './utils/narrationBridge'

// NOTE: loader.ts is intentionally NOT exported
// Hooks import directly from @rbee/sdk-loader
```

---

## Step 9: Build and Verify

Build the package:

```bash
cd bin/10_queen_rbee/ui/packages/queen-rbee-react
pnpm build
```

**Expected output:**
```
✓ Built successfully
No errors
```

**If errors occur:**
1. Check import paths
2. Check deleted files aren't imported elsewhere
3. Check types are still exported

---

## Step 10: Verify App Still Works

Build the Queen UI app:

```bash
cd ../app
pnpm build
```

**Expected output:**
```
✓ vite build succeeded
✓ Built successfully
```

Test in dev mode:

```bash
pnpm dev
```

Open browser to http://localhost:7834

**Verify:**
- [ ] Page loads without errors
- [ ] SDK loads successfully
- [ ] No console errors about missing modules
- [ ] Heartbeat data appears
- [ ] RHAI IDE loads

---

## Step 11: Count Lines Removed

Calculate actual code reduction:

```bash
cd ../packages/queen-rbee-react/src

# Count old implementation
wc -l loader.ts.backup globalSlot.ts  # ~140 total

# Count new implementation  
wc -l loader.ts  # ~25 (including comments)

# Net reduction: ~115 LOC
```

**Record in summary:**
- Old: ~140 LOC (loader.ts + globalSlot.ts)
- New: ~25 LOC (loader.ts with imports)
- Removed: ~115 LOC (82% reduction)

---

## Step 12: Add TEAM-352 Signatures

Add team signature to modified files:

**loader.ts** (already added in Step 3)

**index.ts:**
```typescript
// TEAM-352: SDK loader migrated to @rbee/sdk-loader package
```

---

## Testing Checklist

Before moving to next step:

- [ ] `pnpm install` - no errors
- [ ] `pnpm build` (queen-rbee-react) - success
- [ ] `pnpm build` (queen-rbee-ui app) - success
- [ ] `pnpm dev` - app starts
- [ ] Browser loads http://localhost:7834 - no errors
- [ ] SDK loads (check console for "SDK loaded")
- [ ] Heartbeat works
- [ ] RHAI IDE loads
- [ ] No TypeScript errors
- [ ] No runtime errors

---

## Troubleshooting

### Issue: Cannot find module '@rbee/sdk-loader'

**Fix:**
```bash
cd /home/vince/Projects/llama-orch
pnpm install
cd bin/10_queen_rbee/ui/packages/queen-rbee-react
pnpm build
```

### Issue: Type errors in loader.ts

**Fix:** Check that `RbeeSDK` type matches required exports:
```typescript
export interface RbeeSDK {
  QueenClient: any
  HeartbeatMonitor: any
  OperationBuilder: any
  RhaiClient: any
}
```

### Issue: SDK not loading in browser

**Debug steps:**
1. Open DevTools console
2. Look for error messages
3. Check Network tab for failed WASM loads
4. Verify @rbee/queen-rbee-sdk is built

### Issue: Import errors after deletion

**Fix:** Search for deleted file imports:
```bash
cd src
grep -r "globalSlot" .
grep -r "from './utils'" .
```

Remove any lingering imports.

---

## Success Criteria

✅ Package builds without errors  
✅ App builds without errors  
✅ App runs in dev mode  
✅ SDK loads successfully  
✅ ~115 LOC removed  
✅ No duplicate retry logic  
✅ TEAM-352 signatures added

---

## Next Step

Continue to **TEAM_352_STEP_2_HOOKS_MIGRATION.md** to migrate hooks to @rbee/react-hooks.

---

**TEAM-352 Step 1: Prove the SDK loader pattern works!** ✅
