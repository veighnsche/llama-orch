# TEAM-352 Step 5: Comprehensive Testing & Verification

**Estimated Time:** 45-60 minutes  
**Priority:** CRITICAL  
**Previous Step:** TEAM_352_STEP_4_CONFIG_CLEANUP.md  
**Next Step:** TEAM_352_FINAL_SUMMARY.md

---

## Mission

Perform comprehensive end-to-end testing to ensure NO REGRESSIONS after migration.

**Why This Matters:**
- Validates shared packages work correctly
- Ensures ALL functionality preserved
- Catches integration issues
- Proves pattern works before Hive/Worker use it

**Test Coverage:**
1. Development mode (Vite dev server)
2. Production mode (embedded build)
3. Narration flow (Backend → Queen → Keeper)
4. Hot reload (HMR)
5. All Queen features (Heartbeat, RHAI IDE)

---

## Deliverables Checklist

- [ ] Dev mode: All features work
- [ ] Prod mode: All features work
- [ ] Narration flows correctly
- [ ] Hot reload works
- [ ] SDK loads successfully
- [ ] No console errors
- [ ] No TypeScript errors
- [ ] Performance acceptable
- [ ] Test results documented

---

## Test Matrix

| Test Case | Dev Mode | Prod Mode | Status |
|-----------|----------|-----------|--------|
| App loads | [ ] | [ ] | |
| SDK loads | [ ] | [ ] | |
| Heartbeat connects | [ ] | [ ] | |
| Heartbeat updates | [ ] | [ ] | |
| RHAI IDE loads | [ ] | [ ] | |
| RHAI test works | [ ] | [ ] | |
| Narration appears | [ ] | [ ] | |
| Function extraction | [ ] | [ ] | |
| Hot reload works | [ ] N/A | |
| No console errors | [ ] | [ ] | |

---

## Pre-Test Setup

### 1. Clean Build

Remove all build artifacts:

```bash
cd bin/10_queen_rbee/ui

# Clean Queen packages
cd packages/queen-rbee-react
rm -rf dist node_modules
cd ../queen-rbee-sdk
rm -rf pkg node_modules

# Clean Queen app
cd ../app
rm -rf dist node_modules

# Reinstall everything
cd ../../../../..  # Back to monorepo root
pnpm install
```

### 2. Build All Packages

```bash
# Build shared packages
cd frontend/packages/sdk-loader
pnpm build

cd ../react-hooks
pnpm build

cd ../shared-config
pnpm build

cd ../narration-client
pnpm build

cd ../dev-utils
pnpm build

# Build Queen WASM SDK
cd ../../bin/10_queen_rbee/ui/packages/queen-rbee-sdk
pnpm build:wasm

# Build Queen React package
cd ../queen-rbee-react
pnpm build
```

**All builds must succeed before testing.**

---

## Test Phase 1: Development Mode

### Terminal Setup

**Terminal 1:** Queen Backend
```bash
cd /home/vince/Projects/llama-orch
cargo run --bin queen-rbee
```

**Wait for:**
```
🚀 Queen HTTP server listening on 127.0.0.1:7833
✅ Health check endpoint: http://127.0.0.1:7833/health
```

**Terminal 2:** Queen UI Dev Server
```bash
cd bin/10_queen_rbee/ui/app
pnpm dev
```

**Wait for:**
```
  ➜  Local:   http://localhost:7834/
```

**Terminal 3:** Keeper UI Dev Server
```bash
cd bin/00_rbee_keeper/ui
pnpm dev
```

**Wait for:**
```
  ➜  Local:   http://localhost:5173/
```

### Test 1.1: App Loads

Open browser to: **http://localhost:7834**

**Expected:**
- [ ] Page loads without errors
- [ ] Theme toggle works (dark/light)
- [ ] Navigation works
- [ ] No 404 errors in Network tab

**Console should show:**
```
🔧 [QUEEN UI] DEVELOPMENT mode
├─ Vite dev server: http://localhost:7834
├─ Hot reload: ENABLED
└─ Environment: development
```

**NOT the old hardcoded port logs.**

### Test 1.2: SDK Loads

**In browser console, check for SDK loading:**

**Expected logs:**
```
[sdk-loader] Loading @rbee/queen-rbee-sdk...
[sdk-loader] SDK loaded successfully in XXXms (1 attempt)
```

**Verify SDK is available:**
```javascript
// In console, should return true:
window.__rbeeSDKInit_v1__ !== undefined
```

**Check for errors:**
- [ ] No "SDK load failed" errors
- [ ] No "WebAssembly" errors
- [ ] No timeout errors

### Test 1.3: Heartbeat Connects

**Navigate to Dashboard page (should be default).**

**Expected:**
- [ ] "Workers Online: 0" (or actual count)
- [ ] "Hives Online: 0" (or actual count)
- [ ] Timestamp updates every few seconds
- [ ] Connection indicator shows "Connected" (green)

**In console:**
```
[HeartbeatMonitor] Connected to http://localhost:7833/v1/heartbeat
[HeartbeatMonitor] Snapshot received: { workers_online: 0, ... }
```

**Verify no CORS errors:**
- [ ] No "Access-Control-Allow-Origin" errors
- [ ] No "preflight" errors

### Test 1.4: Heartbeat Updates

**Watch dashboard for 15 seconds.**

**Expected:**
- [ ] Timestamp updates (proves SSE working)
- [ ] Worker count stays accurate
- [ ] No disconnections

**If backend has workers running:**
- [ ] Worker list appears
- [ ] Worker details shown

### Test 1.5: RHAI IDE Loads

**Navigate to RHAI IDE (if nav exists, or check routes).**

**Expected:**
- [ ] IDE interface loads
- [ ] Code editor appears
- [ ] Toolbar buttons present
- [ ] No loading spinners stuck

**If scripts exist:**
- [ ] Scripts list loads
- [ ] Can select scripts

### Test 1.6: RHAI Test Works

**In RHAI IDE:**

1. Create new script or use existing
2. Add simple code: `print("test");`
3. Click "Test" or "Run" button

**Expected:**
- [ ] Button shows "Testing..." state
- [ ] Narration events appear (see next test)
- [ ] Test completes (success or error)
- [ ] Button returns to normal state

**Console should show:**
```
[RHAI Test] Starting test...
[RHAI Test] Client created, baseUrl: http://localhost:7833
[RHAI Test] Operation: { operation: "rhai_script_test", content: "..." }
[RHAI Test] SSE line: data: {"actor":"queen_rbee",...}
[RHAI Test] Narration event: { actor: "queen_rbee", ... }
[RHAI Test] Finished
```

### Test 1.7: Narration Flow (CRITICAL)

**With Queen loaded in Keeper iframe:**

Open **two browser windows:**
1. Keeper at http://localhost:5173 (Queen in iframe)
2. DevTools Console watching both windows

**In Queen iframe:**
- Press "Test" button in RHAI IDE

**Expected in Queen console:**
```
[Queen] Sending to parent: { origin: "http://localhost:5173", action: "...", actor: "queen_rbee" }
```

**Expected in Keeper console:**
```
[Keeper] Received narration from Queen: { actor: "queen_rbee", action: "...", human: "..." }
```

**Verify:**
- [ ] Events appear in BOTH consoles
- [ ] Keeper narration panel shows events
- [ ] Function names extracted (if applicable)
- [ ] [DONE] marker appears at end

**If narration doesn't appear:**
- STOP and debug (see TEAM_352_STEP_3 troubleshooting)
- This is CRITICAL for the system

### Test 1.8: Hot Reload Works

**With Queen UI dev server running:**

Edit `app/src/pages/DashboardPage.tsx`:

```typescript
// Add a comment or change text:
<h1>Dashboard (MODIFIED)</h1>
```

**Save file.**

**Expected:**
- [ ] Vite detects change (console log)
- [ ] Page updates without full reload
- [ ] SDK doesn't reload (stays initialized)
- [ ] Heartbeat connection maintained

**This proves HMR-safe global slot works.**

### Test 1.9: Error Handling

**Test error scenarios:**

1. **Backend offline:**
   - Stop Queen backend (Ctrl+C in Terminal 1)
   - Wait 5 seconds
   - Check dashboard shows "Disconnected" or error state
   - Restart backend
   - Should auto-reconnect (if retry enabled)

2. **Invalid RHAI script:**
   - Enter invalid code: `this is not valid`
   - Click Test
   - Should show error (not crash)

**Expected:**
- [ ] Errors handled gracefully
- [ ] No unhandled Promise rejections
- [ ] UI remains functional

---

## Test Phase 2: Production Mode

### Build Production Assets

**Terminal 1:** Build Queen UI
```bash
cd bin/10_queen_rbee/ui/app
pnpm build
```

**Expected:**
```
✓ Built successfully
dist/ directory created
```

**Terminal 2:** Build Queen backend (with embedded UI)
```bash
cd /home/vince/Projects/llama-orch
cargo build --release --bin queen-rbee
```

**This embeds the UI into the binary.**

### Run Production Build

**Terminal 1:** Start Queen in release mode
```bash
cargo run --release --bin queen-rbee
```

**Wait for:**
```
🚀 Queen HTTP server listening on 127.0.0.1:7833
📦 Serving embedded UI from /
```

**Terminal 2:** Start Keeper (if testing iframe)
```bash
cd bin/00_rbee_keeper/ui
pnpm dev  # Or Tauri app if available
```

### Test 2.1: Production App Loads

Open browser to: **http://localhost:7833**

**Expected:**
- [ ] Page loads from embedded files
- [ ] No Vite dev server messages
- [ ] Theme works
- [ ] Navigation works

**Console should show:**
```
🚀 [QUEEN UI] PRODUCTION mode
├─ Serving: embedded static files
├─ Backend: http://localhost:7833
└─ Environment: production
```

### Test 2.2: Production SDK Loads

**Same as Test 1.2, but check:**

**Expected:**
- [ ] SDK loads successfully
- [ ] No different behavior from dev mode
- [ ] Loading time reasonable (<2s)

### Test 2.3: Production Features Work

**Run ALL feature tests from Phase 1:**

- [ ] Heartbeat connects
- [ ] Heartbeat updates
- [ ] RHAI IDE loads
- [ ] RHAI test works
- [ ] Narration flows correctly

**Expected:** Identical behavior to dev mode.

### Test 2.4: Production Narration

**With Queen iframe in Keeper:**

**Expected origin:**
- Should be `*` (wildcard) for Tauri app
- OR `http://localhost:7833` if Keeper is web

**Verify:**
- [ ] Narration events reach Keeper
- [ ] No origin errors

---

## Test Phase 3: Performance & Bundle Size

### Check Bundle Size

```bash
cd bin/10_queen_rbee/ui/app
ls -lh dist/assets/*.js
```

**Expected:**
- Main bundle: <500 KB (after gzip)
- WASM file: <2 MB

**Compare to before migration:**
- Should be similar or smaller (we removed code)

### Check Load Time

**In browser DevTools Network tab:**

Hard refresh (Ctrl+Shift+R) and measure:

**Expected:**
- [ ] DOMContentLoaded: <1s
- [ ] Load: <2s
- [ ] SDK init: <500ms

### Check Memory Usage

**In browser DevTools Performance Monitor:**

**Expected:**
- [ ] Heap size stable (not growing)
- [ ] No memory leaks
- [ ] Event listeners cleaned up on unmount

---

## Test Phase 4: Regression Tests

### Specific Regressions to Check

1. **Multiple concurrent loads:**
   - Open 3 tabs to Queen UI
   - All should load successfully
   - Only 1 SDK load should occur (singleflight)

2. **Rapid navigation:**
   - Navigate between Dashboard and RHAI IDE 10 times quickly
   - No errors
   - No memory leaks

3. **Long-running connection:**
   - Leave heartbeat connected for 5 minutes
   - Should stay connected
   - No disconnections

4. **Backend restart:**
   - Stop backend while UI connected
   - Wait 5 seconds
   - Restart backend
   - UI should recover (if retry enabled)

---

## Test Results Summary

### Development Mode Results

| Feature | Status | Notes |
|---------|--------|-------|
| App loads | [ ] PASS / [ ] FAIL | |
| SDK loads | [ ] PASS / [ ] FAIL | |
| Heartbeat | [ ] PASS / [ ] FAIL | |
| RHAI IDE | [ ] PASS / [ ] FAIL | |
| Narration | [ ] PASS / [ ] FAIL | |
| Hot reload | [ ] PASS / [ ] FAIL | |

### Production Mode Results

| Feature | Status | Notes |
|---------|--------|-------|
| App loads | [ ] PASS / [ ] FAIL | |
| SDK loads | [ ] PASS / [ ] FAIL | |
| Heartbeat | [ ] PASS / [ ] FAIL | |
| RHAI IDE | [ ] PASS / [ ] FAIL | |
| Narration | [ ] PASS / [ ] FAIL | |

### Regressions Found

**List any issues discovered:**

1. Issue: _____
   - Expected: _____
   - Actual: _____
   - Fix: _____

*(If no regressions, write "NONE")*

---

## Troubleshooting Common Issues

### Issue: SDK fails to load in production

**Cause:** WASM file not embedded correctly

**Fix:**
```bash
cd bin/10_queen_rbee/ui/packages/queen-rbee-sdk
pnpm build:wasm

cd ../app
pnpm build

cargo build --release --bin queen-rbee
```

### Issue: Narration doesn't work in production

**Cause:** Origin mismatch

**Check:**
```javascript
// In Queen console
console.log('Current port:', window.location.port)
// Should be 7833

// In Keeper console
console.log('Expected origin:', getAllowedOrigins())
// Should include http://localhost:7833
```

### Issue: Heartbeat disconnects frequently

**Cause:** Backend SSE timeout or network issues

**Debug:**
```bash
# Check backend logs for SSE errors
# Check Network tab for failed SSE connections
```

### Issue: Hot reload breaks SDK

**Cause:** Global slot not HMR-safe

**Verify:**
```javascript
// Should persist across HMR:
window.__rbeeSDKInit_v1__
```

---

## Sign-Off Criteria

**Before declaring TEAM-352 complete:**

✅ ALL dev mode tests pass  
✅ ALL prod mode tests pass  
✅ NO regressions found  
✅ Narration flows correctly  
✅ Performance acceptable  
✅ Bundle size reasonable  
✅ Hot reload works  
✅ Test results documented

**If ANY test fails, STOP and fix before moving to summary.**

---

## Next Step

If all tests pass, continue to **TEAM_352_FINAL_SUMMARY.md** to document the migration.

---

**TEAM-352 Step 5: Testing complete!** ✅
