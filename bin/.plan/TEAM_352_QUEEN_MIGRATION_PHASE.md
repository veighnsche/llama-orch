# TEAM-352: Queen UI Migration to Shared Packages

**Status:** 🔜 TODO  
**Assigned To:** TEAM-352  
**Estimated Time:** 1 day  
**Priority:** HIGH (Validates shared packages)  
**Dependencies:** TEAM-351 must be complete

---

## Mission

Migrate Queen UI from duplicate code to shared packages created by TEAM-351.

**Why This Matters:**
- Validates shared packages work correctly
- Reduces Queen UI codebase
- Proves pattern before Hive/Worker use it
- Prevents regression

---

## Prerequisites

- [ ] TEAM-351 complete (all packages built)
- [ ] Read `TEAM_351_SHARED_PACKAGES_PHASE.md`
- [ ] Verify `@rbee/shared-config` builds
- [ ] Verify `@rbee/narration-client` builds
- [ ] Verify Rust constants generated

---

## Deliverables Checklist

- [ ] Queen narrationBridge replaced with `@rbee/narration-client`
- [ ] Queen iframe URL uses `@rbee/shared-config`
- [ ] Keeper message listener uses `@rbee/shared-config`
- [ ] Environment detection uses `@rbee/dev-utils`
- [ ] Old duplicate code removed
- [ ] Narration flow still works (tested)
- [ ] Both dev and prod modes tested

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
import { getIframeUrl } from '@rbee/shared-config'
import { SERVICES } from '@rbee/narration-client'
import { logStartupMode } from '@rbee/dev-utils'

console.log('✅ All imports work')
```

Run: `npx tsx test-imports.ts`

---

## Phase 2: Replace narrationBridge.ts

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

## Phase 3: Update Keeper UI Dependencies

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

## Phase 4: Update Keeper QueenPage.tsx

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

## Phase 5: Update Keeper Message Listener

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

## Phase 6: Update App.tsx Startup Logs

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

## Phase 7: Remove Old Code

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

## Phase 8: Build and Test

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

## Phase 9: Verify Code Reduction

### Step 1: Count Lines Removed

**Before migration:**
- `narrationBridge.ts`: ~100 LOC
- Hardcoded ports: ~10 LOC
- Hardcoded origins: ~5 LOC
- Manual environment detection: ~15 LOC
- **Total:** ~130 LOC

**After migration:**
- `narrationBridge.ts`: ~15 LOC
- Shared config imports: ~3 LOC
- **Total:** ~18 LOC

**Reduction:** ~110 LOC removed (85% reduction in duplicate code)

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
- Removed: ~110 LOC duplicate code
- Added: ~18 LOC using shared packages
- Net reduction: ~92 LOC (85%)

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
