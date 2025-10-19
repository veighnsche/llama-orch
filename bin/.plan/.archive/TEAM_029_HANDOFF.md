# TEAM-029 Handoff: E2E Test Improvements

**Date:** 2025-10-10T00:08:00+02:00  
**From:** TEAM-028 (disconnected)  
**To:** TEAM-030  
**Status:** Phase 7 improved, e2e test blocked on model file

---

## Executive Summary

TEAM-029 continued TEAM-028's work and made critical improvements:

1. ✅ **Fixed SQLite connection issues** - Added `sqlite://` prefix and `?mode=rwc` parameter
2. ✅ **Fixed localhost handling** - Don't append `.home.arpa` to localhost
3. ✅ **Improved Phase 7 robustness** - Fail fast after 10 connection errors instead of waiting 5 minutes
4. ✅ **Created preflight check script** - Validates prerequisites before running e2e test

**Current Status:**
- Phases 1-6: ✅ Working
- Phase 7: ✅ Improved (fails fast now)
- Phase 8: ⚠️ Not tested yet (blocked on model file)
- E2E Test: ❌ Blocked - no model file available

---

## What TEAM-029 Fixed

### Fix 1: SQLite Connection String (CRITICAL)

**Problem:** SQLx couldn't open database file  
**Root Cause:** Missing `sqlite://` prefix and `?mode=rwc` parameter

**Solution:**
- Added `connection_string()` helper method in `worker-registry`
- Automatically adds `sqlite://` prefix
- Adds `?mode=rwc` to create database if it doesn't exist
- Creates parent directory if missing

**Files Modified:**
- `bin/shared-crates/worker-registry/src/lib.rs`

**Code:**
```rust
fn connection_string(&self) -> String {
    if self.db_path.starts_with("sqlite://") || self.db_path.starts_with(":memory:") || self.db_path.starts_with("file:") {
        self.db_path.clone()
    } else {
        format!("sqlite://{}?mode=rwc", self.db_path)
    }
}
```

### Fix 2: Localhost Handling

**Problem:** `rbee infer --node localhost` tried to connect to `localhost.home.arpa` (DNS error)

**Solution:**
- Special case for `localhost` and `127.0.0.1`
- Don't append `.home.arpa` suffix

**Files Modified:**
- `bin/rbee-keeper/src/commands/infer.rs`

**Code:**
```rust
let pool_url = if node == "localhost" || node == "127.0.0.1" {
    format!("http://{}:8080", node)
} else {
    format!("http://{}.home.arpa:8080", node)
};
```

### Fix 3: Phase 7 Fail-Fast Logic (CRITICAL)

**Problem:** Phase 7 waited 5 minutes even when worker was completely unreachable

**Solution:**
- Track consecutive connection failures
- Fail after 10 consecutive errors (20 seconds)
- Distinguish between:
  - Worker not ready yet (keep waiting)
  - Worker unreachable (fail fast)
  - Worker returning errors (fail fast)

**Files Modified:**
- `bin/rbee-keeper/src/commands/infer.rs`

**Behavior:**
- ✅ Connection succeeds, worker not ready → keep waiting
- ❌ 10 consecutive connection errors → fail with helpful message
- ❌ 10 consecutive HTTP errors → fail with helpful message

### Fix 4: Preflight Check Script

**Created:** `bin/.specs/.gherkin/test-001-mvp-preflight.sh`

**Checks:**
1. Rust toolchain installed
2. rbee-hive builds
3. rbee-keeper builds
4. llm-worker-rbee binary exists
5. Model file exists
6. Port 8080 available
7. SQLite installed

**Usage:**
```bash
./bin/.specs/.gherkin/test-001-mvp-preflight.sh
```

---

## Current Test Results

### Preflight Check Output:
```
✓ Rust toolchain
✓ rbee-hive builds
✓ rbee-keeper builds
✓ llm-worker-rbee binary exists
✗ Model file not found
✓ Port 8080 available
✓ SQLite installed
```

### E2E Test Progress:
```
[Phase 1] ✓ Worker registry check
[Phase 2] ✓ Pool preflight
[Phase 3-5] ✓ Worker spawn request
[Phase 6] ✓ Worker registration
[Phase 7] ❌ Worker ready - fails after 20s (worker not running)
[Phase 8] ⏸️ Not reached
```

**Why Phase 7 fails:**
- Worker binary spawns but can't load model (no model file)
- Worker process exits or never becomes ready
- Phase 7 now fails fast after 10 attempts (20 seconds)

---

## What TEAM-030 Must Do

### Priority 1: Get Model File (BLOCKING)

**Option A: Download test model**
```bash
cd bin/llm-worker-rbee
./download_test_model.sh
```

**Option B: Use existing model**
- Find where models are stored
- Update hardcoded path in `infer.rs` line 74:
  ```rust
  model_path: "/models/model.gguf".to_string(), // TODO: Get from catalog
  ```

**Option C: Mock the worker**
- Create a mock HTTP server that responds to `/v1/ready` and `/v1/inference`
- Test orchestration logic without real inference

### Priority 2: Complete E2E Test

Once model is available:

1. **Run preflight check:**
   ```bash
   ./bin/.specs/.gherkin/test-001-mvp-preflight.sh
   ```

2. **Run local e2e test:**
   ```bash
   ./bin/.specs/.gherkin/test-001-mvp-local.sh
   ```

3. **Verify all 8 phases work:**
   - Phase 1: Worker registry ✅
   - Phase 2: Pool preflight ✅
   - Phase 3-5: Worker spawn ✅
   - Phase 6: Worker registration ✅
   - Phase 7: Worker ready ⏳
   - Phase 8: Inference execution ⏳

### Priority 3: Document Results

Create `TEAM_030_COMPLETION_SUMMARY.md` with:
- E2E test results
- Any bugs found
- Any additional fixes made

---

## Files Created by TEAM-029

1. `bin/.specs/.gherkin/test-001-mvp-local.sh` - Local e2e test
2. `bin/.specs/.gherkin/test-001-mvp-preflight.sh` - Preflight checks
3. `bin/.plan/TEAM_029_HANDOFF.md` - This document

## Files Modified by TEAM-029

1. `bin/shared-crates/worker-registry/src/lib.rs` - SQLite connection fix
2. `bin/rbee-keeper/src/commands/infer.rs` - Localhost handling + Phase 7 fail-fast

---

## Known Issues

### Issue 1: Hardcoded Model Path
**Location:** `bin/rbee-keeper/src/commands/infer.rs:74`  
**Problem:** `model_path: "/models/model.gguf"` is hardcoded  
**Impact:** Won't work if model is elsewhere  
**Solution:** Implement model catalog lookup or make it configurable

### Issue 2: Hardcoded Backend
**Location:** `bin/rbee-keeper/src/commands/infer.rs:72`  
**Problem:** `backend: "cpu"` is hardcoded  
**Impact:** Can't use GPU even if available  
**Solution:** Detect backend from node capabilities or make it a CLI arg

### Issue 3: No Worker Cleanup
**Problem:** If test fails, worker process may keep running  
**Impact:** Port conflicts, resource leaks  
**Solution:** Add cleanup trap in test script

### Issue 4: No Model Validation
**Problem:** Worker spawns even if model doesn't exist  
**Impact:** Worker fails to start, Phase 7 times out  
**Solution:** Validate model exists before spawning worker

---

## Testing Strategy

### Manual Testing (Recommended First)

**Terminal 1: Start daemon**
```bash
./target/debug/rbee-hive daemon
```

**Terminal 2: Check health**
```bash
curl http://localhost:8080/v1/health | jq .
```

**Terminal 3: Run inference**
```bash
./target/debug/rbee infer \
  --node localhost \
  --model "test" \
  --prompt "hello world" \
  --max-tokens 5 \
  --temperature 0.7
```

### Automated Testing

**Step 1: Preflight**
```bash
./bin/.specs/.gherkin/test-001-mvp-preflight.sh
```

**Step 2: E2E Test**
```bash
./bin/.specs/.gherkin/test-001-mvp-local.sh
```

---

## Success Criteria

### Minimum (Unblock E2E)
- [ ] Model file available
- [ ] Preflight check passes
- [ ] Worker starts successfully
- [ ] Phase 7 completes (worker ready)
- [ ] Phase 8 completes (inference works)

### Target (Full MVP)
- [ ] All 8 phases work end-to-end
- [ ] Test script passes
- [ ] Documentation updated
- [ ] Known issues documented

### Stretch (Production Ready)
- [ ] Model catalog integration
- [ ] Backend auto-detection
- [ ] Worker cleanup on failure
- [ ] Comprehensive error messages

---

## Advice for TEAM-030

1. **Run preflight first** - Don't waste time debugging if prerequisites are missing
2. **Test manually before automation** - Easier to debug
3. **Check worker logs** - If Phase 7 fails, check why worker isn't starting
4. **One phase at a time** - Don't try to fix everything at once
5. **Document what you find** - Future teams will thank you

**Remember:** The infrastructure is solid now. The only blocker is the model file.

---

**Signed:** TEAM-029  
**Date:** 2025-10-10T00:08:00+02:00  
**Status:** ✅ Phase 7 improved, ❌ E2E blocked on model  
**Next Team:** TEAM-030 - Get model file and complete e2e test! 🚀
