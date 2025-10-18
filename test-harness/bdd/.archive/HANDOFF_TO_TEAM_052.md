# HANDOFF TO TEAM-052

**From:** TEAM-051  
**Date:** 2025-10-10T19:08:00+02:00  
**Status:** Ready for handoff

---

## Executive Summary

TEAM-051 successfully implemented **global queen-rbee instance** to fix port conflicts. We now have **1 shared queen-rbee** across all 62 scenarios (not 62 instances!). We also updated test-001 to target **workstation with CUDA device 1** instead of mac with Metal.

**Test Results:**
- ✅ **32/62 scenarios passing** (real state, no port conflicts)
- ✅ **1 queen-rbee instance** (shared across all scenarios)
- ✅ Fast test execution (0ms startup after first launch)

**Documentation Updates:**
- ✅ Updated all specs to reflect **rbee-keeper is the USER INTERFACE** (not a testing tool)
- ✅ Updated README.md with new architecture understanding
- ✅ Updated CRITICAL_RULES.md and COMPONENT_RESPONSIBILITIES_FINAL.md

**Your mission:** Enhance rbee-hive registry to track backend capabilities and implement proper lifecycle management.

---

## ✅ What TEAM-051 Completed

### 1. Global queen-rbee Instance ✅
**Impact:** Prevents port conflicts, enables proper test isolation

**Problem:** 62 queen-rbee instances trying to bind to port 8080  
**Solution:** Single shared instance started once before all tests

**Files Created:**
- `test-harness/bdd/src/steps/global_queen.rs` - Global instance management

**Files Modified:**
- `test-harness/bdd/src/main.rs` - Calls `start_global_queen_rbee()` before tests
- `test-harness/bdd/src/steps/background.rs` - Uses global instance
- `test-harness/bdd/src/steps/beehive_registry.rs` - Uses global instance
- `test-harness/bdd/src/steps/world.rs` - Drop doesn't kill queen-rbee

**Results:**
```
Before: 62 queen-rbee instances → port conflicts → connection errors
After:  1 queen-rbee instance → clean isolation → 32/62 passing
```

### 2. Test Updates: workstation + CUDA ✅
**Impact:** All tests now target workstation with CUDA device 1

**Changed:** All references from `mac` (Metal device 0) to `workstation` (CUDA device 1)

**Files Modified:**
- `bin/.specs/.gherkin/test-001.md` - Updated all examples
- `test-harness/bdd/tests/features/test-001.feature` - Updated all scenarios

**Best Effort:** We changed as many references as we could find, but there may be more scattered throughout the codebase.

### 3. Documentation Updates ✅
**Impact:** Clarified that rbee-keeper is the USER INTERFACE

**Key Insight:** rbee-keeper is NOT a testing tool - it's the CLI UI for llama-orch!

**Files Modified:**
- `README.md` - Updated binary descriptions
- `bin/.specs/COMPONENT_RESPONSIBILITIES_FINAL.md` - Updated component table
- `bin/.specs/CRITICAL_RULES.md` - Updated component table and flow

**Changes:**
```diff
- rbee-keeper: TESTING TOOL - integration tester
+ rbee-keeper: USER INTERFACE - manages queen-rbee, hives, workers, SSH config
```

---

## 🎯 Your Mission: Backend Registry + Lifecycle Management

### Priority 1: Enhance rbee-hive Registry Schema

**Current Problem:** Registry doesn't track what backends/devices each node supports.

**Required Schema Addition:**
```sql
-- Add to beehives table
ALTER TABLE beehives ADD COLUMN backends TEXT; -- JSON array: ["cuda", "metal", "cpu"]
ALTER TABLE beehives ADD COLUMN devices TEXT;  -- JSON object: {"cuda": 2, "metal": 1, "cpu": 1}
```

**Example Data:**
```json
{
  "node_name": "workstation",
  "ssh_host": "workstation.home.arpa",
  "backends": ["cuda", "cpu"],
  "devices": {
    "cuda": 2,    // 2 CUDA devices (device 0, device 1)
    "cpu": 1      // 1 CPU device
  }
}
```

**Files to Modify:**
- `bin/queen-rbee/src/registry.rs` - Update schema
- `bin/queen-rbee/src/http.rs` - Update `/v2/registry/beehives/add` endpoint
- `test-harness/bdd/src/steps/beehive_registry.rs` - Update test data

**Acceptance Criteria:**
- [ ] Registry stores backend capabilities per node
- [ ] Registry stores device counts per backend
- [ ] queen-rbee validates backend/device availability before spawning workers
- [ ] Tests pass with new schema

### Priority 2: Implement Proper Lifecycle Management

**Current Problem:** rbee-keeper doesn't manage queen-rbee/hive/worker lifecycles properly.

**Required Implementation:**

#### A. queen-rbee Lifecycle
```rust
// rbee-keeper should:
// 1. Start queen-rbee if not running
// 2. Connect to existing queen-rbee if already running
// 3. Optionally stop queen-rbee (with cascading shutdown)

rbee-keeper daemon start    // Start queen-rbee
rbee-keeper daemon stop     // Stop queen-rbee (cascades to all hives/workers)
rbee-keeper daemon status   // Check if queen-rbee is running
```

#### B. rbee-hive Lifecycle
```rust
// rbee-keeper should manage hives via queen-rbee:

rbee-keeper hive start --node workstation    // Start hive on remote node
rbee-keeper hive stop --node workstation     // Stop hive (cascades to workers)
rbee-keeper hive status --node workstation   // Check hive status
```

#### C. Worker Lifecycle
```rust
// rbee-keeper should manage workers via queen-rbee:

rbee-keeper worker start --node workstation --model tinyllama --backend cuda --device 1
rbee-keeper worker stop --id worker-abc123
rbee-keeper worker list --node workstation
```

#### D. Cascading Shutdown Principle
**CRITICAL:** When a parent dies, all children die gracefully.

```
queen-rbee dies
    ↓ cascades to
all rbee-hives die
    ↓ cascades to
all workers die
```

**Implementation:**
1. queen-rbee tracks all active hives (SSH connections)
2. When queen-rbee receives SIGTERM, it sends shutdown to all hives
3. Each hive sends shutdown to all its workers
4. Workers gracefully finish current request (with timeout)
5. Everything exits cleanly

### Priority 3: SSH Configuration Management

**Current Problem:** No way to configure SSH for remote machines.

**Required Implementation:**
```rust
// rbee-keeper should manage SSH config:

rbee-keeper config set-ssh --node workstation \
  --host workstation.home.arpa \
  --user vince \
  --key ~/.ssh/id_ed25519

rbee-keeper config list-nodes  // Show all configured nodes
rbee-keeper config remove-node --node workstation
```

**Files to Create/Modify:**
- `bin/rbee-keeper/src/commands/config.rs` - SSH config commands
- `bin/rbee-keeper/src/config.rs` - Config file management (~/.rbee/config.toml)

---

## 📊 Current Test Status

### Passing (32/62)
- ✅ Setup commands (add-node, install, list-nodes, remove-node)
- ✅ Registry operations
- ✅ Some pool preflight checks
- ✅ Some worker preflight checks
- ✅ Some CLI commands

### Failing (30/62)
**Root Causes:**
1. **Missing step definitions** (2 scenarios)
   - "Then the worker receives shutdown command"
   - "And the stream continues until Ctrl+C"

2. **Backend registry not implemented** (estimated 10+ scenarios)
   - Can't validate backend/device availability
   - Can't track which nodes support which backends

3. **Lifecycle management not implemented** (estimated 10+ scenarios)
   - Can't start/stop queen-rbee properly
   - Can't start/stop hives properly
   - Can't start/stop workers properly
   - Cascading shutdown not implemented

4. **SSH config not implemented** (estimated 5+ scenarios)
   - Can't configure SSH for remote machines
   - Can't validate SSH connectivity

---

## 🔍 Key Files to Study

### Registry Schema
- `bin/queen-rbee/src/registry.rs` - Current registry implementation
- `bin/queen-rbee/src/http.rs` - Registry HTTP endpoints

### Lifecycle Management
- `bin/rbee-keeper/src/commands/` - CLI command implementations
- `bin/queen-rbee/src/main.rs` - queen-rbee daemon lifecycle
- `bin/rbee-hive/src/main.rs` - rbee-hive daemon lifecycle

### Test Infrastructure
- `test-harness/bdd/src/steps/global_queen.rs` - Global queen-rbee instance
- `test-harness/bdd/src/steps/beehive_registry.rs` - Registry test steps
- `test-harness/bdd/src/steps/cli_commands.rs` - CLI command test steps

---

## 🚨 Critical Insights

### 1. rbee-keeper is the USER INTERFACE
**This is a fundamental architecture change!**

- ❌ OLD: rbee-keeper is a testing tool
- ✅ NEW: rbee-keeper is the CLI UI for llama-orch

**Implications:**
- rbee-keeper manages queen-rbee lifecycle (start/stop)
- rbee-keeper configures SSH for remote machines
- rbee-keeper manages hives and workers
- Future: Web UI will be added alongside CLI

### 2. Cascading Shutdown is CRITICAL
When queen-rbee dies, everything dies gracefully:
```
queen-rbee SIGTERM
    ↓ sends shutdown to all hives
rbee-hive receives shutdown
    ↓ sends shutdown to all workers
worker receives shutdown
    ↓ finishes current request (with timeout)
    ↓ exits cleanly
```

### 3. Backend Registry is Essential
Without backend/device tracking, we can't:
- Validate that a node supports the requested backend
- Know how many devices are available
- Schedule workers intelligently
- Provide good error messages

---

## 🎯 Recommended Approach

### Phase 1: Backend Registry (Day 1-2)
1. Update schema with backends/devices columns
2. Update `/v2/registry/beehives/add` endpoint
3. Update test data in `beehive_registry.rs`
4. Verify tests pass with new schema

**Expected Impact:** +5 scenarios

### Phase 2: Lifecycle Management (Day 3-5)
1. Implement `rbee-keeper daemon start/stop/status`
2. Implement `rbee-keeper hive start/stop/status`
3. Implement `rbee-keeper worker start/stop/list`
4. Implement cascading shutdown

**Expected Impact:** +10 scenarios

### Phase 3: SSH Configuration (Day 6-7)
1. Implement `rbee-keeper config set-ssh`
2. Implement `rbee-keeper config list-nodes`
3. Implement `rbee-keeper config remove-node`
4. Store config in `~/.rbee/config.toml`

**Expected Impact:** +5 scenarios

### Phase 4: Missing Step Definitions (Day 8)
1. Implement "Then the worker receives shutdown command"
2. Implement "And the stream continues until Ctrl+C"

**Expected Impact:** +2 scenarios

**Total Expected:** 54/62 scenarios passing (87%)

---

## 📝 Testing Strategy

### Unit Tests
```bash
# Test registry schema
cargo test --package queen-rbee --lib registry::tests

# Test lifecycle management
cargo test --package rbee-keeper --lib commands::daemon::tests
```

### Integration Tests
```bash
# Run all BDD tests
cd test-harness/bdd
cargo run --bin bdd-runner

# Run specific feature
LLORCH_BDD_FEATURE_PATH="tests/features/test-001.feature" cargo run --bin bdd-runner
```

### Manual Testing
```bash
# Test queen-rbee lifecycle
rbee-keeper daemon start
rbee-keeper daemon status
rbee-keeper daemon stop

# Test hive lifecycle
rbee-keeper hive start --node workstation
rbee-keeper hive status --node workstation
rbee-keeper hive stop --node workstation

# Test worker lifecycle
rbee-keeper worker start --node workstation --model tinyllama --backend cuda --device 1
rbee-keeper worker list --node workstation
rbee-keeper worker stop --id worker-abc123
```

---

## 🎁 Bonus: Future Web UI

**Note:** This is NOT for TEAM-052 to implement, but good to know:

We're planning a web UI alongside the CLI:
- Same functionality as CLI
- Real-time monitoring dashboard
- Visual worker/hive management
- Log streaming
- Model catalog browser

The architecture supports this because rbee-keeper is now the UI layer!

---

## 📚 Additional Context

### Test-001 Changes
We updated all test-001 scenarios to use:
- **Node:** workstation (was: mac)
- **Backend:** cuda (was: metal)
- **Device:** 1 (was: 0)

**Rationale:** More common setup (CUDA is more widely used than Metal)

### Global queen-rbee Implementation
The global instance is managed by `OnceLock` and started once before all tests:
```rust
// test-harness/bdd/src/steps/global_queen.rs
static GLOBAL_QUEEN: OnceLock<GlobalQueenRbee> = OnceLock::new();

pub async fn start_global_queen_rbee() {
    // Start once, reuse for all scenarios
}
```

This prevents port conflicts and improves test performance.

---

## 🤝 Handoff Checklist

- [x] Global queen-rbee instance implemented
- [x] Port conflicts resolved
- [x] Test-001 updated to workstation + CUDA
- [x] Documentation updated (rbee-keeper is UI)
- [x] Test results documented (32/62 passing)
- [x] Next steps clearly defined
- [x] Handoff document created

---

## 💬 Questions for TEAM-052?

If you have questions, check these resources first:
1. `bin/.specs/COMPONENT_RESPONSIBILITIES_FINAL.md` - Component roles
2. `bin/.specs/CRITICAL_RULES.md` - Lifecycle rules
3. `test-harness/bdd/README.md` - BDD testing guide
4. `test-harness/bdd/src/steps/global_queen.rs` - Global instance implementation

Good luck! 🚀

---

**TEAM-051 signing off.**
