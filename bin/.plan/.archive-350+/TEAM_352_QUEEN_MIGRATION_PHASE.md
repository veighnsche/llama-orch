# TEAM-352: Queen UI Migration to Shared Packages

**Status:** 🔜 TODO  
**Assigned To:** TEAM-352  
**Estimated Time:** 1 day  
**Priority:** HIGH (Validates shared packages)  
**Dependencies:** TEAM-356 must be complete

---

## Mission

Migrate Queen UI from duplicate code to shared packages created by TEAM-356.

**Why This Matters:**
- Validates shared packages work correctly
- Reduces Queen UI codebase
- Proves pattern before Hive/Worker use it
- Prevents regression

---

## Prerequisites

- [ ] TEAM-356 complete (all packages built)
- [ ] Read `TEAM_356_EXTRACTION_EXTRAVAGANZA.md`
- [ ] Verify `@rbee/sdk-loader` builds (34 tests passing)
- [ ] Verify `@rbee/react-hooks` builds (19 tests passing)
- [ ] Verify `@rbee/shared-config` builds
- [ ] Verify `@rbee/narration-client` builds

---

## Deliverables Checklist

### Phase 3: Migrate Queen UI (from TEAM-356)

- [ ] Queen SDK loader replaced with `@rbee/sdk-loader`
- [ ] Queen hooks replaced with `@rbee/react-hooks`
- [ ] Queen narrationBridge replaced with `@rbee/narration-client`
- [ ] Queen iframe URL uses `@rbee/shared-config`
- [ ] Keeper message listener uses `@rbee/shared-config`
- [ ] Environment detection uses `@rbee/dev-utils`
- [ ] Old duplicate code removed (~300-400 LOC)
- [ ] Narration flow still works (tested)
- [ ] Both dev and prod modes tested
- [ ] All hardcoded URLs replaced with `getServiceUrl()`

---

## Phase 1: Update Queen UI Package Dependencies

### Step 1: Add Package Dependencies

```bash
cd bin/10_queen_rbee/ui/packages/queen-rbee-react
```

Edit `package.json`, add dependencies:

```json
{
  "dependencies": {
    "@rbee/sdk-loader": "workspace:*",
    "@rbee/react-hooks": "workspace:*",
    "@rbee/shared-config": "workspace:*",
    "@rbee/narration-client": "workspace:*",
    "@rbee/dev-utils": "workspace:*"
  }
}
```

Install:

```bash
pnpm install
```

### Step 2: Verify Imports Work

Create test file to verify:

```typescript
// test-imports.ts
import { createSDKLoader } from '@rbee/sdk-loader'
import { useSSEWithHealthCheck } from '@rbee/react-hooks'
import { useQuery } from '@tanstack/react-query'
import { getIframeUrl, getServiceUrl } from '@rbee/shared-config'
import { SERVICES } from '@rbee/narration-client'
import { logStartupMode } from '@rbee/dev-utils'

console.log('✅ All imports work')
```

Run: `npx tsx test-imports.ts`

---

## Phase 2: Migrate SDK Loader

### Step 1: Replace Custom Loader

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/loader.ts`

**OLD CODE (~120 lines):**
```typescript
// Custom loader with retry logic, exponential backoff, singleflight, etc.
export async function loadSDK() { ... }
```

**NEW CODE (~10 lines):**
```typescript
// TEAM-356: Use shared @rbee/sdk-loader package
import { createSDKLoader } from '@rbee/sdk-loader'

export const queenSDKLoader = createSDKLoader({
  packageName: '@rbee/queen-rbee-sdk',
  requiredExports: ['QueenClient', 'HeartbeatMonitor', 'RhaiClient'],
  timeout: 15000,
  maxAttempts: 3,
})

export const loadSDK = queenSDKLoader.loadOnce
```

**Lines saved:** ~110 lines

### Step 2: Delete globalSlot.ts

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/globalSlot.ts`

Delete this file (no longer needed - singleflight pattern in @rbee/sdk-loader).

---

## Phase 3: Migrate React Hooks

### Step 1: Replace useHeartbeat Hook

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/hooks/useHeartbeat.ts`

**OLD CODE (~90 lines):**
```typescript
export function useHeartbeat(baseUrl = 'http://localhost:7833') {
  const [data, setData] = useState(null)
  const [connected, setConnected] = useState(false)
  // ... 60+ lines of health check + SSE logic
}
```

**NEW CODE (~15 lines):**
```typescript
// TEAM-356: Use shared @rbee/react-hooks package
import { useSSEWithHealthCheck } from '@rbee/react-hooks'
import { getServiceUrl } from '@rbee/shared-config'

export function useHeartbeat(
  baseUrl = getServiceUrl('queen', 'prod')  // Use shared config!
) {
  return useSSEWithHealthCheck(
    (url) => new sdk.HeartbeatMonitor(url),
    baseUrl
  )
}
```

**Lines saved:** ~75 lines

### Step 2: Replace useRhaiScripts Hook

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/hooks/useRhaiScripts.ts`

**OLD CODE (~250 lines):**
```typescript
export function useRhaiScripts(baseUrl = 'http://localhost:7833') {
  const [scripts, setScripts] = useState([])
  const [loading, setLoading] = useState(true)
  // ... 150+ lines of async state management
}
```

**NEW CODE (~40 lines):**
```typescript
// TEAM-356: Use TanStack Query for data fetching
import { useQuery } from '@tanstack/react-query'
import { getServiceUrl } from '@rbee/shared-config'

export function useRhaiScripts(
  baseUrl = getServiceUrl('queen', 'prod')
) {
  const { data: sdk } = useSDK()
  
  const { data: scripts, isLoading, error, refetch } = useQuery({
    queryKey: ['rhai-scripts', baseUrl],
    queryFn: async () => {
      if (!sdk) return []
      const client = new sdk.RhaiClient(baseUrl)
      const result = await client.listScripts()
      return JSON.parse(JSON.stringify(result))
    },
    enabled: !!sdk,
  })
  
  // Keep save/delete/select functions (business logic)
  const save = async (script) => { /* ... */ }
  const deleteScript = async (id) => { /* ... */ }
  
  return { 
    scripts: scripts ?? [], 
    loading: isLoading, 
    error, 
    refetch, 
    save, 
    delete: deleteScript 
  }
}
```

**Lines saved:** ~210 lines

### Step 3: Fix All Hardcoded URLs

Search for `'http://localhost:7833'` and `'http://localhost:7834'` in Queen UI:

```bash
cd bin/10_queen_rbee/ui
grep -r "localhost:783" .
```

Replace with:
```typescript
import { getServiceUrl } from '@rbee/shared-config'

const baseUrl = getServiceUrl('queen', 'prod')  // or 'dev'
```

---

## Phase 4: Replace narrationBridge.ts

### Step 1: Locate Current Implementation

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/utils/narrationBridge.ts`

**Current code to replace:**
- `parseNarrationLine()` → Use from `@rbee/narration-client`
- `sendNarrationToParent()` → Use from `@rbee/narration-client`
- Environment detection → Use from `@rbee/narration-client`

### Step 2: Create New Implementation

Replace entire file with:

```typescript
// TEAM-352: Migrated to use @rbee/narration-client shared package
// Old implementation: ~100 LOC of duplicate code
// New implementation: ~15 LOC using shared package

import { 
  createStreamHandler, 
  SERVICES,
  type BackendNarrationEvent 
} from '@rbee/narration-client'

export type NarrationEvent = BackendNarrationEvent

/**
 * Create narration stream handler for Queen
 * Automatically parses SSE lines and sends to parent window
 */
export function createNarrationStreamHandler(
  onEvent?: (event: NarrationEvent) => void
): (line: string) => void {
  return createStreamHandler(SERVICES.queen, onEvent)
}

// Re-export for backward compatibility
export { parseNarrationLine } from '@rbee/narration-client'
```

### Step 3: Verify Usage Still Works

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/hooks/useRhaiScripts.ts`

Verify this line still works:

```typescript
const handleNarration = createNarrationStreamHandler()

// Usage in SSE stream:
for await (const line of stream) {
  handleNarration(line)
}
```

**No changes needed** - API is the same!

---

## Phase 5: Update Keeper UI Dependencies

### Step 1: Add Dependencies to Keeper

```bash
cd bin/00_rbee_keeper/ui
```

Edit `package.json`:

```json
{
  "dependencies": {
    "@rbee/shared-config": "workspace:*",
    "@rbee/dev-utils": "workspace:*"
  }
}
```

Install:

```bash
pnpm install
```

---

## Phase 6: Update Keeper QueenPage.tsx

### Step 1: Replace Hardcoded URL

**File:** `bin/00_rbee_keeper/ui/src/pages/QueenPage.tsx`

**OLD CODE:**
```typescript
const isDev = import.meta.env.DEV
const queenUrl = isDev 
  ? "http://localhost:7834"  // ❌ Hardcoded
  : "http://localhost:7833"   // ❌ Hardcoded
```

**NEW CODE:**
```typescript
// TEAM-352: Use shared config instead of hardcoded ports
import { getIframeUrl } from '@rbee/shared-config'

const isDev = import.meta.env.DEV
const queenUrl = getIframeUrl('queen', isDev)
```

### Step 2: Verify Component Renders

```bash
cd bin/00_rbee_keeper/ui
pnpm dev
```

Open http://localhost:5173, navigate to Queen page.

**Expected:** iframe loads correctly from Vite dev server.

---

## Phase 7: Update Keeper Message Listener

### Step 1: Replace Hardcoded Origins

**File:** `bin/00_rbee_keeper/ui/src/utils/narrationListener.ts`

**OLD CODE:**
```typescript
const allowedOrigins = [
  "http://localhost:7833",  // ❌ Hardcoded
  "http://localhost:7834",  // ❌ Hardcoded
]
```

**NEW CODE:**
```typescript
// TEAM-352: Use shared config for allowed origins
import { getAllowedOrigins } from '@rbee/shared-config'

const allowedOrigins = getAllowedOrigins()
// Automatically includes all services (queen, hive, worker)
```

### Step 2: Verify Message Reception

Test narration flow:

1. Start queen backend: `cargo run --bin queen-rbee`
2. Start keeper: `pnpm dev` in keeper UI
3. Navigate to Queen page
4. Press "Test" button in RHAI IDE
5. Check console for narration events

**Expected logs:**
```
[Keeper] Received narration from Queen: {actor: "queen_rbee", ...}
```

---

## Phase 8: Update App.tsx Startup Logs

### Step 1: Update Queen App.tsx

**File:** `bin/10_queen_rbee/ui/app/src/App.tsx`

**OLD CODE:**
```typescript
const isDev = import.meta.env.DEV
if (isDev) {
  console.log('🔧 [QUEEN UI] Running in DEVELOPMENT mode')
  console.log('   - Vite dev server active (hot reload enabled)')
} else {
  console.log('🚀 [QUEEN UI] Running in PRODUCTION mode')
  console.log('   - Serving embedded static files')
}
```

**NEW CODE:**
```typescript
// TEAM-352: Use shared dev-utils for startup logging
import { logStartupMode, isDevelopment, getCurrentPort } from '@rbee/dev-utils'

logStartupMode('QUEEN UI', isDevelopment(), getCurrentPort())
```

### Step 2: Update Keeper App.tsx

**File:** `bin/00_rbee_keeper/ui/src/App.tsx`

Same replacement as Queen.

---

## Phase 9: Remove Old Code

### Step 1: Identify Code to Remove

**DO NOT REMOVE** (these are still used):
- Type definitions that differ from shared packages
- Queen-specific hooks
- Component logic

**REMOVE** (these are now in shared packages):
- Hardcoded port numbers
- Duplicate origin arrays
- Manual postMessage origin detection
- Duplicate environment detection

### Step 2: Clean Up Imports

Remove any unused imports from files you modified:

```bash
# Check for unused imports
cd bin/00_rbee_keeper/ui
pnpm exec eslint --fix src/
```

---

## Phase 10: Build and Test

### Step 1: Build Queen UI

```bash
cd bin/10_queen_rbee/ui/packages/queen-rbee-react
pnpm build

cd ../app
pnpm build
```

**Verify:** No build errors

### Step 2: Build Keeper UI

```bash
cd bin/00_rbee_keeper/ui
pnpm build
```

**Verify:** No build errors

### Step 3: Test Development Mode

**Terminal 1:** Start Queen Vite dev server
```bash
cd bin/10_queen_rbee/ui/app
pnpm dev
```

**Terminal 2:** Start Queen backend
```bash
cargo run --bin queen-rbee
```

**Terminal 3:** Start Keeper
```bash
cd bin/00_rbee_keeper/ui
pnpm dev
```

**Browser:** Open http://localhost:5173

**Test checklist:**
- [ ] Queen page loads
- [ ] iframe shows Queen UI from Vite (port 7834)
- [ ] Hot reload works (edit Queen code, see changes)
- [ ] Navigate to RHAI IDE
- [ ] Press "Test" button
- [ ] Narration appears in keeper panel
- [ ] Function names extracted
- [ ] No console errors

### Step 4: Test Production Mode

**Build everything:**
```bash
cargo build --release --bin queen-rbee
cd bin/00_rbee_keeper/ui && pnpm build
```

**Run:**
```bash
cargo run --release --bin queen-rbee
# Open keeper in Tauri app
```

**Test checklist:**
- [ ] Queen page loads
- [ ] iframe shows embedded Queen UI (port 7833)
- [ ] Narration works
- [ ] No console errors

---

## Phase 11: Verify Code Reduction

### Step 1: Count Lines Removed

**Before migration:**
- Custom SDK loader: ~120 LOC
- Custom useHeartbeat: ~90 LOC
- Custom useRhaiScripts: ~250 LOC
- `narrationBridge.ts`: ~100 LOC
- Hardcoded ports: ~10 LOC
- Hardcoded origins: ~5 LOC
- Manual environment detection: ~15 LOC
- **Total:** ~590 LOC

**After migration:**
- SDK loader import: ~10 LOC
- useHeartbeat wrapper: ~15 LOC
- useRhaiScripts wrapper: ~40 LOC
- `narrationBridge.ts`: ~15 LOC
- Shared config imports: ~3 LOC
- **Total:** ~83 LOC

**Reduction:** ~507 LOC removed (86% reduction in duplicate code)

### Step 2: Document Changes

Create a summary file:

```bash
cat > bin/.plan/TEAM_352_MIGRATION_SUMMARY.md << 'EOF'
# TEAM-352 Migration Summary

## Changes Made

### Files Modified
1. `bin/10_queen_rbee/ui/packages/queen-rbee-react/package.json` - Added dependencies
2. `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/utils/narrationBridge.ts` - Replaced with shared package
3. `bin/00_rbee_keeper/ui/package.json` - Added dependencies
4. `bin/00_rbee_keeper/ui/src/pages/QueenPage.tsx` - Use getIframeUrl()
5. `bin/00_rbee_keeper/ui/src/utils/narrationListener.ts` - Use getAllowedOrigins()
6. `bin/10_queen_rbee/ui/app/src/App.tsx` - Use logStartupMode()
7. `bin/00_rbee_keeper/ui/src/App.tsx` - Use logStartupMode()

### Code Reduction
- Removed: ~507 LOC duplicate code
- Added: ~83 LOC using shared packages
- Net reduction: ~424 LOC (86%)

### Packages Used
- @rbee/sdk-loader - WASM/SDK loading with retry logic
- @rbee/react-hooks - useSSEWithHealthCheck (custom SSE hook)
- @tanstack/react-query - useQuery, useMutation (data fetching)
- @rbee/shared-config - Port configuration
- @rbee/narration-client - Narration handling
- @rbee/dev-utils - Environment detection

### Functionality Preserved
✅ Hot reload works
✅ Narration flows correctly
✅ Both dev and prod modes work
✅ No regressions

## Testing Results

### Development Mode
- [x] Queen loads from Vite (port 7834)
- [x] Hot reload works
- [x] Narration appears in keeper
- [x] Function names extracted
- [x] No console errors

### Production Mode
- [x] Queen loads embedded (port 7833)
- [x] Narration works
- [x] No console errors

## Next Steps for TEAM-353

Ready to implement Hive UI using same shared packages.
EOF
```

---

## Troubleshooting

### Issue: Package not found

**Symptom:** `Cannot find package '@rbee/shared-config'`

**Fix:**
```bash
cd frontend
pnpm install
```

### Issue: Narration not appearing

**Debug steps:**
1. Check console: `[Keeper] Received narration from Queen:`
2. Verify allowed origins includes both 7833 and 7834
3. Check Queen is sending to correct origin
4. Verify backend sends JSON (not plain text)

### Issue: Hot reload not working

**Debug steps:**
1. Verify Vite running on port 7834: `nc -zv localhost 7834`
2. Verify iframe src is `http://localhost:7834`
3. Check `import.meta.env.DEV` is true
4. Hard refresh (Ctrl+Shift+R)

### Issue: Build errors

**Fix:**
```bash
# Clean and rebuild
cd frontend/packages/shared-config && pnpm build
cd ../narration-client && pnpm build
cd ../../bin/10_queen_rbee/ui/packages/queen-rbee-react && pnpm build
```

---

## Acceptance Criteria

**Must all pass before handoff:**

- [ ] All packages install successfully
- [ ] Queen UI builds without errors
- [ ] Keeper UI builds without errors
- [ ] Development mode: Narration flows correctly
- [ ] Production mode: Narration flows correctly
- [ ] Hot reload works in development
- [ ] At least 80 LOC removed from Queen
- [ ] No hardcoded ports remain
- [ ] No hardcoded origins remain
- [ ] All tests pass
- [ ] Documentation updated

---

## Handoff to TEAM-353

**What you've proven:**
- Shared packages work correctly
- Migration is straightforward
- No functionality lost
- Significant code reduction

**Next team (TEAM-353) will:**
- Implement Hive UI using same pattern
- Reuse ALL shared packages from day 1
- No duplicate code!

**Files to hand off:**
- Modified Queen UI files
- Migration summary document
- Test results
- This document

---

## Success Criteria

✅ Queen UI uses shared packages  
✅ Code reduced by 80+ LOC  
✅ Narration flow works  
✅ Both modes tested  
✅ No regressions  
✅ Pattern validated for Hive/Worker

---

**TEAM-352: Prove the shared packages work perfectly!** ✅
