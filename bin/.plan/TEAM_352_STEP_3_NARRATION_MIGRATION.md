# TEAM-352 Step 3: Migrate Narration Bridge to @rbee/narration-client

**Estimated Time:** 30-45 minutes  
**Priority:** CRITICAL  
**Previous Step:** TEAM_352_STEP_2_HOOKS_MIGRATION.md  
**Next Step:** TEAM_352_STEP_4_CONFIG_CLEANUP.md

---

## Mission

Replace custom narration bridge implementation with @rbee/narration-client package.

**Why This Matters:**
- Removes ~100 LOC of SSE parsing + postMessage logic
- Uses tested parser (battle-tested eventsource-parser library)
- Consistent narration handling across all UIs
- Proper [DONE] marker handling
- Automatic environment detection

**Code Reduction:** ~111 LOC → ~20 LOC (82% reduction)

---

## Deliverables Checklist

- [ ] Added @rbee/narration-client dependency
- [ ] Migrated narrationBridge.ts to use shared package
- [ ] Removed custom SSE parsing logic
- [ ] Removed custom postMessage logic
- [ ] Removed hardcoded environment detection
- [ ] Package builds successfully
- [ ] Narration still flows to parent
- [ ] TEAM-352 signatures added

---

## Step 1: Add Package Dependency

Navigate to Queen React package:

```bash
cd bin/10_queen_rbee/ui/packages/queen-rbee-react
```

Edit `package.json` (should already have it from Step 2, but verify):

```json
{
  "dependencies": {
    "@rbee/queen-rbee-sdk": "workspace:*",
    "@rbee/sdk-loader": "workspace:*",
    "@rbee/react-hooks": "workspace:*",
    "@rbee/shared-config": "workspace:*",
    "@rbee/narration-client": "workspace:*"
  }
}
```

Install (if not already installed):

```bash
cd ../../../..  # Back to monorepo root
pnpm install
```

**Verification:**
```bash
cd bin/10_queen_rbee/ui/packages/queen-rbee-react
ls -la node_modules/@rbee/narration-client  # Should exist
```

---

## Step 2: Analyze Current narrationBridge.ts

Read the current implementation:

```bash
cat src/utils/narrationBridge.ts
```

**Current implementation (~111 LOC):**
- Custom `NarrationEvent` type
- Custom `NarrationMessage` type
- `sendNarrationToParent()` - Manual postMessage with environment detection
- `parseNarrationLine()` - Custom JSON parsing with [DONE] handling
- `createNarrationStreamHandler()` - Combines parse + send

**Target:** Replace with @rbee/narration-client functions.

---

## Step 3: Migrate narrationBridge.ts

**CRITICAL:** Back up first:

```bash
cp src/utils/narrationBridge.ts src/utils/narrationBridge.ts.backup
```

Replace entire contents of `src/utils/narrationBridge.ts`:

```typescript
// TEAM-352: DELETED - Migrated to @rbee/narration-client package
// Old implementation: ~111 LOC of custom SSE parsing + postMessage
// New: Import directly from @rbee/narration-client in hooks that need it
// Reduction: 111 LOC (100%)

// This file is intentionally empty/minimal
// DO NOT create wrapper exports "for backward compatibility"
// UPDATE imports in hooks to use @rbee/narration-client directly

export type { BackendNarrationEvent as NarrationEvent } from '@rbee/narration-client'
```

**Key changes:**
- ✅ Removed custom `NarrationEvent` type (uses `BackendNarrationEvent`)
- ✅ Removed custom `NarrationMessage` type (in shared package)
- ✅ Removed `sendNarrationToParent()` (in `createStreamHandler`)
- ✅ Removed custom `parseNarrationLine()` (re-exported from shared)
- ✅ Removed environment detection (in `createStreamHandler`)
- ✅ Added `SERVICES.queen` configuration

---

## Step 4: Verify Service Configuration

The @rbee/narration-client package needs to know about the "queen" service.

Check that it's defined:

```bash
cat ../../../../frontend/packages/narration-client/src/config.ts | grep -A 5 "queen"
```

**Should see:**
```typescript
export const SERVICES = {
  queen: {
    name: 'queen-rbee',
    dev: { port: 7834 },
    prod: { port: 7833 },
  },
  // ...
}
```

**If "queen" is missing:**

This shouldn't happen if TEAM-351/356 was complete, but if it is missing, you need to add it to the shared package. Stop here and fix TEAM-351's work first.

---

## Step 5: Update Hook to Import Directly

**CRITICAL:** Update useRhaiScripts.ts to import directly from @rbee/narration-client (NO WRAPPER):

```bash
cat src/hooks/useRhaiScripts.ts
```

Find the import line:

**OLD import (WRONG - uses wrapper):**
```typescript
import { createNarrationStreamHandler } from '../utils/narrationBridge'
```

**NEW import (CORRECT - direct from shared package):**

Edit `src/hooks/useRhaiScripts.ts`:

```typescript
// TEAM-352: Import directly from @rbee/narration-client (no wrapper)
import { createStreamHandler, SERVICES } from '@rbee/narration-client'
```

Then update the usage in the `testScript` function:

**OLD usage (WRONG):**
```typescript
const narrationHandler = createNarrationStreamHandler((event) => {
  console.log('[RHAI Test] Narration event:', event)
})
```

**NEW usage (CORRECT):**
```typescript
const narrationHandler = createStreamHandler(SERVICES.queen, (event) => {
  console.log('[RHAI Test] Narration event:', event)
}, {
  debug: import.meta.env.DEV,
  silent: false,
  validate: true,
})
```

**This is the CORRECT pattern:**
- ✅ Import directly from @rbee/narration-client
- ✅ Use SERVICES.queen for config
- ❌ NO wrapper exports in narrationBridge.ts

---

## Step 6: Build and Verify

Build the package:

```bash
cd bin/10_queen_rbee/ui/packages/queen-rbee-react
pnpm build
```

**Expected output:**
```
✓ Built successfully
No TypeScript errors
```

**If errors occur:**
1. Check import paths
2. Verify @rbee/narration-client exports SERVICES
3. Check types match (BackendNarrationEvent)

---

## Step 7: Test Narration Flow

This is **CRITICAL** - narration must flow from Queen → Keeper.

**Terminal 1:** Start Queen backend
```bash
cargo run --bin queen-rbee
```

**Terminal 2:** Start Queen UI dev server
```bash
cd bin/10_queen_rbee/ui/app
pnpm dev
```

**Terminal 3:** Start Keeper UI
```bash
cd bin/00_rbee_keeper/ui
pnpm dev
```

**Browser:** Open Keeper at http://localhost:5173

Navigate to Queen page (iframe should load).

**Test narration:**
1. Open DevTools Console (keep it open)
2. Navigate to RHAI IDE in Queen iframe
3. Press "Test" button
4. **Watch for narration events in console**

**Expected console logs:**
```
[Queen] Sending to parent: { origin: "http://localhost:5173", action: "...", actor: "queen_rbee" }
[Keeper] Received narration from Queen: { actor: "queen_rbee", action: "...", human: "..." }
```

**If narration doesn't appear:**
- Check console for errors
- Verify postMessage origin is correct
- Verify Keeper is listening for messages
- See Troubleshooting section

---

## Step 8: Test [DONE] Marker Handling

The parser should gracefully handle [DONE] markers (not treat as error).

**In browser console, while testing RHAI:**

Look for `[DONE]` in SSE stream. Should NOT see warnings about "Failed to parse".

**Expected:** Silent handling (no console warnings for [DONE]).

**If you see warnings:**
- Parser may not be handling [DONE] correctly
- Check @rbee/narration-client implementation
- Report issue to TEAM-351/356

---

## Step 9: Count Lines Removed

Calculate code reduction:

```bash
cd src/utils

# Old implementation
wc -l narrationBridge.ts.backup
# ~111 LOC

# New implementation
wc -l narrationBridge.ts
# ~20 LOC

# Net reduction: ~91 LOC (82%)
```

**Record in summary:**
- Old: ~111 LOC (custom parsing + postMessage)
- New: ~20 LOC (using shared package)
- Removed: ~91 LOC (82% reduction)

---

## Step 10: Add TEAM-352 Signature

Signature already added in file (Step 3).

Verify:

```bash
grep -n "TEAM-352" src/utils/narrationBridge.ts
```

---

## Testing Checklist

Before moving to next step:

- [ ] `pnpm install` - no errors
- [ ] `pnpm build` (queen-rbee-react) - success
- [ ] `pnpm build` (queen-rbee-ui app) - success
- [ ] Narration events appear in Keeper console
- [ ] Narration panel in Keeper shows events
- [ ] Function names extracted from events
- [ ] [DONE] marker handled gracefully
- [ ] No parse errors in console
- [ ] Environment detection works (dev mode)
- [ ] No TypeScript errors
- [ ] No runtime errors

---

## Troubleshooting

### Issue: Narration not appearing in Keeper

**Debug steps:**

1. **Check Queen is sending:**
```javascript
// In Queen iframe console (DevTools)
window.addEventListener('message', (e) => console.log('Message sent:', e))
```

2. **Check Keeper is receiving:**
```javascript
// In Keeper console (DevTools)
window.addEventListener('message', (e) => console.log('Message received:', e))
```

3. **Check origins match:**
- Queen dev server: port 7834
- Keeper dev server: port 5173
- Message origin should be: `http://localhost:5173`

4. **Check message type:**
```javascript
// Should be 'NARRATION_EVENT'
console.log(event.data.type)
```

### Issue: SERVICES.queen not found

**Fix:**

```bash
# Check narration-client exports
cat frontend/packages/narration-client/src/config.ts

# Should export:
# export const SERVICES = { queen: { ... }, hive: { ... }, worker: { ... } }
```

If missing, TEAM-351/356 didn't complete properly.

### Issue: Parse errors for valid events

**Debug:**
```typescript
// In narrationBridge.ts, temporarily add debug logging:
export function createNarrationStreamHandler(onLocal) {
  return (line: string) => {
    console.log('[DEBUG] Raw line:', line)
    const handler = createStreamHandler(SERVICES.queen, onLocal)
    handler(line)
  }
}
```

Check what raw SSE lines look like.

### Issue: Environment detection wrong

**Check current port:**
```javascript
// In browser console
console.log(window.location.port)
// Dev: should be 7834
// Prod: should be 7833
```

**Check parent origin:**
```javascript
// Should be http://localhost:5173 in dev
// Should be * in prod (Tauri)
```

---

## Success Criteria

✅ Narration bridge migrated successfully  
✅ Narration flows from Queen to Keeper  
✅ [DONE] marker handled correctly  
✅ Environment detection works  
✅ Package builds without errors  
✅ ~91 LOC removed  
✅ TEAM-352 signatures added

---

## Next Step

Continue to **TEAM_352_STEP_4_CONFIG_CLEANUP.md** to remove remaining hardcoded URLs.

---

**TEAM-352 Step 3: Narration bridge migration complete!** ✅
