# TEAM-055 QUICK START GUIDE

**Status:** 42/62 passing → Target: 62/62 passing  
**Timeline:** 6-7 days  
**Priority:** Fix 20 failing tests

---

## 🎯 Your Mission in 3 Phases

### Phase 1: HTTP Retry Logic (Days 1-2) 🔴 P0
**Impact:** +6 scenarios (42 → 48)

**File:** `test-harness/bdd/src/steps/beehive_registry.rs`

**What to do:**
1. Add retry helper function with exponential backoff
2. Replace all `.send().await.expect()` with retry logic
3. Add 5-second timeout per request
4. Retry 3 times with 100ms, 200ms, 400ms delays

**Code pattern:**
```rust
// TEAM-055: HTTP retry with exponential backoff
for attempt in 0..3 {
    match client.post(url).json(payload).timeout(Duration::from_secs(5)).send().await {
        Ok(resp) => return Ok(resp),
        Err(e) if attempt < 2 => {
            tokio::time::sleep(Duration::from_millis(100 * 2_u64.pow(attempt))).await;
        }
        Err(e) => return Err(e),
    }
}
```

### Phase 2: Exit Codes (Days 3-5) 🟡 P1
**Impact:** +13 scenarios (48 → 61)

**Files:**
- `bin/rbee-keeper/src/commands/infer.rs` - Fix exit code 2→0 and None→0
- `bin/rbee-keeper/src/commands/install.rs` - Fix exit code 1→0
- `bin/rbee-keeper/src/commands/workers.rs` - Fix exit code 1→0

**What to do:**
1. Ensure all functions return `anyhow::Result<()>`
2. Success = `Ok(())` (exit code 0)
3. Error = `anyhow::bail!()` (exit code 1)
4. Add debug logging to find where errors occur

**Code pattern:**
```rust
// TEAM-055: Proper exit code handling
pub async fn handle_command() -> anyhow::Result<()> {
    // Do work...
    if success {
        Ok(())  // Exit code 0
    } else {
        anyhow::bail!("Error message")  // Exit code 1
    }
}
```

### Phase 3: Missing Step (Day 6) 🟢 P2
**Impact:** +1 scenario (61 → 62)

**What to do:**
1. Check line 452 in `tests/features/test-001.feature`
2. Implement missing step definition
3. Add to appropriate file in `test-harness/bdd/src/steps/`

---

## 🔍 Quick Diagnostics

### Check Mock Servers
```bash
cargo run --bin bdd-runner 2>&1 | grep "Mock servers ready"
# Should show:
#   - queen-rbee: http://127.0.0.1:8080
#   - rbee-hive:  http://127.0.0.1:9200
```

### Test Manually
```bash
curl http://localhost:8080/health
curl http://localhost:9200/v1/health
```

### Run Tests
```bash
cd test-harness/bdd
cargo run --bin bdd-runner
```

### Debug Exit Codes
```bash
./target/debug/rbee-keeper infer --node workstation --model tinyllama --prompt "test"
echo $?  # Check exit code
```

---

## 📊 Failure Breakdown

| Category | Count | Priority | Impact |
|----------|-------|----------|--------|
| HTTP IncompleteMessage | 6 | P0 | +6 scenarios |
| Exit code 2→0 | 1 | P1 | +1 scenario |
| Exit code 1→0 | 2 | P1 | +2 scenarios |
| Exit code None→0 | 1 | P1 | +1 scenario |
| Exit code None→1 | 9 | P1 | +9 scenarios |
| Missing step | 1 | P2 | +1 scenario |
| **TOTAL** | **20** | | **+20 scenarios** |

---

## 🚨 Critical Rules

### Rule 1: Don't Change Ports
All ports are correct. Use:
- queen-rbee: **8080**
- rbee-hive: **9200**
- workers: **8001+**

### Rule 2: Don't Add Features
Focus on fixing the 20 failing tests. No new functionality.

### Rule 3: Follow Patterns
Use the code patterns provided in `HANDOFF_TO_TEAM_055.md`.

### Rule 4: Test After Every Change
Run full test suite after each fix to verify progress.

---

## 📚 Essential Reading

1. **`HANDOFF_TO_TEAM_055.md`** - Complete handoff (READ FIRST!)
2. **`PORT_ALLOCATION.md`** - Port reference
3. **`TEAM_054_SUMMARY.md`** - What TEAM-054 did

---

## ✅ Success Checklist

### Phase 1 Complete
- [ ] HTTP retry function implemented
- [ ] All HTTP calls use retry logic
- [ ] 48+ scenarios passing
- [ ] No more IncompleteMessage errors

### Phase 2 Complete
- [ ] Infer command returns correct exit codes
- [ ] Install command returns 0
- [ ] Worker shutdown returns 0
- [ ] SSE streaming returns 0
- [ ] Edge cases return 1
- [ ] 61+ scenarios passing

### Phase 3 Complete
- [ ] Missing step definition added
- [ ] **62/62 scenarios passing** 🎉

---

## 💡 Pro Tips

1. **Start with Phase 1** - HTTP fixes unlock 6 scenarios immediately
2. **Use debug logging** - Add `tracing::info!()` to track execution
3. **Test incrementally** - Don't fix everything at once
4. **Check exit codes** - Use `echo $?` after running commands
5. **Read the handoff** - All answers are in `HANDOFF_TO_TEAM_055.md`

---

**Target: 62/62 scenarios passing (100%)**

**You've got this!** 🚀
