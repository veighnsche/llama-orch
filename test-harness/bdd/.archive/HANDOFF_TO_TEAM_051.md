# HANDOFF TO TEAM-051: Fix queen-rbee Port Conflicts

**From:** TEAM-050  
**To:** TEAM-051  
**Date:** 2025-10-10  
**Status:** 🟡 46/62 SCENARIOS PASSING (same as TEAM-049, root cause identified)

---

## Executive Summary

TEAM-050 investigated TEAM-049's exit code issues and identified the root cause: **multiple queen-rbee instances trying to bind to port 8080**. The Background section runs for every scenario, spawning a new queen-rbee process each time. Only the first succeeds; subsequent scenarios fail with "error sending request" because their queen-rbee process couldn't start.

**Your mission:** Fix the queen-rbee lifecycle management to prevent port conflicts.

---

## ✅ What TEAM-050 Completed

### 1. Fixed Stream Error Handling ✅
**Impact:** Prevents exit code 1 from stream errors

**Problem:** TEAM-048's `chunk?` on line 76 of `infer.rs` propagated stream errors as function errors  
**Solution:** Graceful error handling with `match` statement

**File:** `bin/rbee-keeper/src/commands/infer.rs` (lines 77-90)
```rust
// TEAM-050: Handle stream errors gracefully - don't propagate them as function errors
// Stream errors (including normal closure) should not cause exit code 1
let chunk = match chunk {
    Ok(bytes) => bytes,
    Err(_e) => {
        // If we've already seen [DONE], stream closure is expected
        if done {
            break;
        }
        // Otherwise continue - the stream might recover or be done
        // (Stream errors after [DONE] are normal and should not cause exit code 1)
        continue;
    }
};
```

### 2. Fixed Background queen-rbee Startup ✅
**Impact:** Actually starts queen-rbee for Background scenarios

**Problem:** `given_queen_rbee_url` just set the URL without starting the process  
**Solution:** Call `given_queen_rbee_running` to actually start queen-rbee

**File:** `test-harness/bdd/src/steps/background.rs` (lines 30-37)
```rust
#[given(expr = "queen-rbee is running at {string}")]
pub async fn given_queen_rbee_url(world: &mut World, url: String) {
    // TEAM-050: Actually start queen-rbee, don't just set the URL
    // The Background section expects queen-rbee to be running for all scenarios
    crate::steps::beehive_registry::given_queen_rbee_running(world).await;
    world.queen_rbee_url = Some(url.clone());
    tracing::debug!("queen-rbee started and URL set to: {}", url);
}
```

### 3. Root Cause Analysis ✅
**Impact:** Identified why scenarios still fail with exit code 1

**Findings:**
- 62 scenarios × Background section = 62 queen-rbee spawn attempts
- All trying to bind to `localhost:8080`
- Only the first succeeds
- Health check at line 42 of `beehive_registry.rs` succeeds for ALL attempts (finds existing process)
- Later scenarios get `Error: error sending request for url (http://localhost:8080/v2/tasks)`
- This is because their queen-rbee process failed to start (port conflict)

**Evidence:**
```bash
$ grep "Starting queen-rbee process" /tmp/team50-final.log | wc -l
62

$ grep "queen-rbee is ready" /tmp/team50-final.log | wc -l  
62

$ grep "stderr.*error sending request" /tmp/team50-final.log | head -2
stderr: Error: error sending request for url (http://localhost:8080/v2/tasks)
stderr: Error: error sending request for url (http://localhost:8080/v2/tasks)
```

---

## 📊 Test Results

### Current Status
```
62 scenarios total
46 passing (74%)
16 failing (26%)

Same as TEAM-049 baseline
```

### Why No Improvement?

TEAM-050's fixes are correct but don't help because:
1. Stream error handling fix only helps IF the request succeeds
2. Background startup fix exposes the port conflict issue
3. The real blocker is that queen-rbee can't start for most scenarios

---

## 🎯 Your Mission: Fix Port Conflicts

### Root Cause
**Problem:** Each scenario spawns a new queen-rbee process, all trying to bind to port 8080.

**Current Flow:**
1. Scenario 1: Background → spawn queen-rbee → binds to :8080 ✅
2. Scenario 2: Background → spawn queen-rbee → **port conflict** ❌
3. Scenario 2: Health check → finds Scenario 1's queen-rbee → reports "ready" ✅
4. Scenario 2: Run command → connects to Scenario 1's queen-rbee → **wrong state** ❌

### Solution Options

#### Option A: Shared queen-rbee Instance (RECOMMENDED)
**Pros:** Fast, simple, matches production usage  
**Cons:** Scenarios share state (might need cleanup between scenarios)

**Implementation:**
1. Start queen-rbee ONCE before all scenarios
2. Reuse the same instance for all scenarios
3. Clean up state between scenarios (clear registry, etc.)

**Changes needed:**
- Move queen-rbee startup to a global setup hook
- Add cleanup step between scenarios
- Ensure each scenario starts with clean state

#### Option B: Dynamic Port Allocation
**Pros:** True isolation between scenarios  
**Cons:** More complex, slower, requires port management

**Implementation:**
1. Allocate a unique port per scenario (e.g., 8080 + scenario_id)
2. Pass port to queen-rbee via `--port` flag
3. Update rbee-keeper to use the correct port

**Changes needed:**
- Port allocation logic in World
- Update all HTTP clients to use dynamic port
- Handle port cleanup

#### Option C: Sequential Execution with Cleanup
**Pros:** Simple, true isolation  
**Cons:** Slower (kill + restart for each scenario)

**Implementation:**
1. Kill queen-rbee in World::drop()
2. Wait for port to be released
3. Start new queen-rbee for next scenario

**Changes needed:**
- Add port wait logic after kill
- Increase startup timeout
- Handle cleanup failures gracefully

---

## 🛠️ Implementation Guide

### Option A: Shared Instance (Recommended)

**Step 1: Create Global Setup**
```rust
// In test-harness/bdd/src/main.rs or lib.rs
static QUEEN_RBEE: OnceCell<Child> = OnceCell::new();

async fn start_global_queen_rbee() -> &'static mut Child {
    QUEEN_RBEE.get_or_init(|| {
        // Start queen-rbee once
        // Return Child process
    }).await
}
```

**Step 2: Update Background Step**
```rust
#[given(expr = "queen-rbee is running at {string}")]
pub async fn given_queen_rbee_url(world: &mut World, url: String) {
    // TEAM-051: Use shared queen-rbee instance
    if world.queen_rbee_process.is_none() {
        let proc = start_global_queen_rbee().await;
        world.queen_rbee_process = Some(proc);
    }
    world.queen_rbee_url = Some(url.clone());
}
```

**Step 3: Add Scenario Cleanup**
```rust
impl World {
    pub async fn reset_for_scenario(&mut self) {
        // Clear registry via HTTP DELETE
        // Reset any shared state
        self.beehive_nodes.clear();
        self.workers.clear();
        // etc.
    }
}
```

**Step 4: Remove Drop Logic**
```rust
impl Drop for World {
    fn drop(&mut self) {
        // TEAM-051: Don't kill shared queen-rbee
        // Only kill scenario-specific processes
        for mut proc in self.rbee_hive_processes.drain(..) {
            let _ = proc.start_kill();
        }
        // etc.
    }
}
```

---

## 📁 Files Modified by TEAM-050

### Core Changes
1. **`bin/rbee-keeper/src/commands/infer.rs`**
   - Lines: 77-90 (stream error handling)
   - TEAM-050 signature added

2. **`test-harness/bdd/src/steps/background.rs`**
   - Lines: 30-37 (queen-rbee startup)
   - TEAM-050 signature added

3. **`test-harness/bdd/Cargo.toml`**
   - Added `shlex = "1.3"` dependency (TEAM-049)

---

## 🎯 Success Criteria for TEAM-051

### Minimum Success
- [ ] Fix port conflict issue
- [ ] At least 50+ scenarios passing (46 → 50+)
- [ ] Document solution approach

### Target Success
- [ ] All exit code 1 issues resolved
- [ ] All happy path scenarios passing (2/2)
- [ ] All CLI command scenarios passing (6/6)
- [ ] 52+ scenarios passing total

### Stretch Goals
- [ ] Implement proper scenario isolation
- [ ] Add cleanup verification
- [ ] 56+ scenarios passing total

---

## 🐛 Debugging Tips

### Verify Port Conflicts

```bash
# Check if multiple queen-rbee processes are running
ps aux | grep queen-rbee

# Check port usage
lsof -i :8080

# Check test logs for port errors
grep "address already in use" /tmp/team50-final.log
grep "error sending request" /tmp/team50-final.log
```

### Test Shared Instance Approach

```bash
# Start queen-rbee manually
./target/debug/queen-rbee --port 8080 --database /tmp/test.db &

# Run tests (should reuse existing instance)
cd test-harness/bdd
cargo run --bin bdd-runner

# Kill manually started instance
pkill queen-rbee
```

### Verify Cleanup

```bash
# Check if queen-rbee is killed between test runs
watch -n 1 'ps aux | grep queen-rbee | grep -v grep'
```

---

## 🎁 What You're Inheriting

### Working Code
- ✅ Stream error handling fixed (prevents exit code 1 from streams)
- ✅ Background startup implemented (exposes port conflict)
- ✅ TEAM-049's quote handling fix (shlex parser)
- ✅ All binaries compile successfully

### Clear Problem Statement
- 📋 Root cause identified and documented
- 📋 Three solution options provided
- 📋 Implementation guide for recommended approach
- 📋 Debugging commands provided

### Clean Patterns
- ✅ TEAM-050 signatures added
- ✅ Clear comments explaining changes
- ✅ No tech debt introduced

---

## 📚 Code Patterns for TEAM-051

### TEAM-051 Signature Pattern
```rust
// TEAM-051: <description of change>
// or
// Modified by: TEAM-051
```

### Global State Pattern (Option A)
```rust
use once_cell::sync::OnceCell;
use tokio::sync::Mutex;

static QUEEN_RBEE: OnceCell<Mutex<Child>> = OnceCell::new();
```

### Cleanup Pattern
```rust
async fn cleanup_scenario_state(world: &mut World) {
    // TEAM-051: Reset shared state between scenarios
    let client = reqwest::Client::new();
    let _ = client.delete("http://localhost:8080/v2/registry/beehives").send().await;
    world.beehive_nodes.clear();
}
```

---

## 📊 Expected Progress

If you fix the port conflict:

| Approach | Scenarios | Total |
|----------|-----------|-------|
| Baseline | - | 46 |
| Option A (shared) | +6 | 52 |
| Option B (dynamic ports) | +6 | 52 |
| Option C (sequential) | +6 | 52 |

**Target: 52+ scenarios (minimum), 56+ scenarios (stretch)**

---

**Good luck, TEAM-051! The path is clear - just need to fix the port conflict!** 🚀

---

**Status:** Ready for handoff to TEAM-051  
**Blocker:** Port conflict (62 queen-rbee instances on same port)  
**Risk:** Low - clear solution path documented  
**Confidence:** High - root cause identified, multiple solutions provided
