# HANDOFF TO TEAM-054: Fix All Port Issues Once and For All

**From:** TEAM-053  
**Date:** 2025-10-10T20:25:00+02:00  
**Status:** 🔴 CRITICAL - Port confusion across multiple handoffs  
**Priority:** P0 - Must fix before continuing

---

## 🚨 CRITICAL: Port Confusion Discovered

TEAM-053 discovered that **multiple teams made port-related mistakes** that propagated through handoff documents. This handoff consolidates all findings and provides a **complete fix plan**.

**Read these first:**
1. `/home/vince/Projects/llama-orch/test-harness/bdd/MISTAKES_AND_CORRECTIONS.md`
2. `/home/vince/Projects/llama-orch/test-harness/bdd/HISTORICAL_MISTAKES_ANALYSIS.md`

---

## Executive Summary

### The Problem
Multiple teams documented **incorrect ports** for rbee-hive:
- **TEAM-043 and earlier:** Used port 8080 (old architecture)
- **TEAM-047:** Used port 8080 (didn't update after architecture change)
- **TEAM-048:** Used port 8080 (copied from TEAM-047)
- **TEAM-053:** Used port 8090 (made up without checking spec)

### The Truth
**According to the normative spec (`bin/.specs/.gherkin/test-001.md`):**
- **queen-rbee:** port 8080 (control node)
- **rbee-hive:** port 9200 (remote nodes) ← THIS IS CORRECT!
- **llm-worker-rbee:** port 8001+ (workers)

### Current Status
- ✅ **Code is correct** (`bin/queen-rbee/src/http/inference.rs` uses 9200)
- ❌ **Documentation is wrong** (multiple handoffs reference 8080 or 8090)
- ❌ **Tests may fail** if mock servers use wrong ports

---

## 🎯 Your Mission: Fix All Port References

### Phase 1: Update TEAM-053's Documents (Day 1)

#### Task 1.1: Fix TEAM_053_SUMMARY.md
**File:** `test-harness/bdd/TEAM_053_SUMMARY.md`

**Find and replace:**
- Line 60: `8090` → `9200`
- Line 62: `8090` → `9200`
- Line 63: `8090` → `9200`
- Line 70: `8090` → `9200`
- Line 72: `8090` → `9200`
- Line 209: `8090` → `9200`
- Line 273: `8090` → `9200`
- Line 294: `8090` → `9200`

**Add correction note at top:**
```markdown
**CORRECTION (TEAM-054):** This document originally stated rbee-hive uses port 8090.
The correct port is **9200** per the normative spec. All references have been updated.
```

#### Task 1.2: Fix HANDOFF_TO_TEAM_054.md
**File:** `test-harness/bdd/HANDOFF_TO_TEAM_054.md`

**Find and replace:**
- Line 72: `8090` → `9200`
- Line 276: `8090` → `9200`
- Line 445: `8090` → `9200`
- Line 460: `8090` → `9200`

**Add correction note at top:**
```markdown
**CORRECTION (TEAM-054):** This document originally stated rbee-hive uses port 8090.
The correct port is **9200** per the normative spec. All references have been updated.
```

### Phase 2: Update TEAM-047's Documents (Day 1)

#### Task 2.1: Fix HANDOFF_TO_TEAM_047.md
**File:** `test-harness/bdd/HANDOFF_TO_TEAM_047.md`

**Find and replace (for rbee-hive only):**
- Line 152: `http://{}:8080` → `http://{}:9200`
- Line 157: `--addr 0.0.0.0:8080` → `--addr 0.0.0.0:9200`
- Line 161: `http://{}:8080` → `http://{}:9200`
- Line 163: `http://{}:8080` → `http://{}:9200`

**Add correction note at top:**
```markdown
**CORRECTION (TEAM-054):** This document originally stated rbee-hive uses port 8080.
The correct port is **9200** per the normative spec (updated by TEAM-037/TEAM-038).
All rbee-hive references have been updated. queen-rbee remains on port 8080.
```

#### Task 2.2: Fix TEAM_047_SUMMARY.md
**File:** `test-harness/bdd/TEAM_047_SUMMARY.md`

**Find and replace:**
- Line 133: `http://127.0.0.1:8080` → `http://127.0.0.1:9200`

**Add correction note at top:**
```markdown
**CORRECTION (TEAM-054):** This document originally stated rbee-hive uses port 8080.
The correct port is **9200** per the normative spec. Reference updated.
```

### Phase 3: Update TEAM-048's Documents (Day 1)

#### Task 3.1: Fix HANDOFF_TO_TEAM_048.md
**File:** `test-harness/bdd/HANDOFF_TO_TEAM_048.md`

**Find and replace (for rbee-hive only):**
- Line 512: `http://mac.home.arpa:8080` → `http://mac.home.arpa:9200`
- Line 513: `http://mac.home.arpa:8080` → `http://mac.home.arpa:9200`

**Add correction note at top:**
```markdown
**CORRECTION (TEAM-054):** This document originally stated rbee-hive uses port 8080.
The correct port is **9200** per the normative spec. All rbee-hive references have been updated.
queen-rbee remains on port 8080.
```

### Phase 4: Mark Old Documents as Historical (Day 1)

#### Task 4.1: Add Note to TEAM-043 Documents
**Files:**
- `HANDOFF_TO_TEAM_043_FINAL.md`
- `HANDOFF_TO_TEAM_043_COMPLETE.md`

**Add note at top:**
```markdown
**HISTORICAL NOTE (TEAM-054):** This document describes the architecture BEFORE TEAM-037/TEAM-038
introduced queen-rbee orchestration. At that time, rbee-hive used port 8080 and rbee-keeper
connected directly to rbee-hive. This architecture is NO LONGER VALID.

**Current architecture:** queen-rbee (8080) → rbee-hive (9200) → workers (8001+)
**See:** `bin/.specs/.gherkin/test-001.md` for current normative spec.
```

### Phase 5: Create Port Allocation Reference (Day 2)

#### Task 5.1: Create PORT_ALLOCATION.md
**File:** `test-harness/bdd/PORT_ALLOCATION.md` (CREATE NEW)

```markdown
# Port Allocation Reference

**Status:** NORMATIVE  
**Source:** `bin/.specs/.gherkin/test-001.md`  
**Last Updated:** 2025-10-10 (TEAM-054)

---

## Official Port Allocation

| Component | Port | Location | Purpose | Status |
|-----------|------|----------|---------|--------|
| queen-rbee | 8080 | Control node (blep.home.arpa) | Orchestrator HTTP API | ✅ Active |
| rbee-hive | 9200 | Remote nodes (workstation, mac) | Pool manager HTTP API | ✅ Active |
| llm-worker-rbee | 8001+ | Remote nodes | Worker HTTP API (sequential) | ✅ Active |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Control Node (blep.home.arpa)                               │
│                                                             │
│  ┌──────────────┐         ┌──────────────┐                │
│  │ rbee-keeper  │────────>│ queen-rbee   │                │
│  │ (CLI tool)   │         │ port 8080    │                │
│  └──────────────┘         └──────┬───────┘                │
│                                   │                         │
└───────────────────────────────────┼─────────────────────────┘
                                    │ SSH
                                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Remote Node (workstation.home.arpa)                         │
│                                                             │
│  ┌──────────────┐         ┌──────────────┐                │
│  │ rbee-hive    │────────>│ llm-worker   │                │
│  │ port 9200    │         │ port 8001    │                │
│  └──────────────┘         └──────────────┘                │
│                                 │                           │
│                         ┌──────────────┐                   │
│                         │ llm-worker   │                   │
│                         │ port 8002    │                   │
│                         └──────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Historical Context

### Before TEAM-037/TEAM-038 (Old Architecture)
- rbee-hive used port **8080**
- rbee-keeper connected DIRECTLY to rbee-hive
- No queen-rbee orchestrator

### After TEAM-037/TEAM-038 (Current Architecture)
- queen-rbee introduced on port **8080**
- rbee-hive moved to port **9200**
- rbee-keeper connects to queen-rbee
- queen-rbee orchestrates via SSH

**Architecture change date:** 2025-10-10 (around 14:00)

---

## Mock Server Configuration

### For BDD Tests

**Mock queen-rbee:**
```rust
let addr: SocketAddr = "127.0.0.1:8080".parse()?;
```

**Mock rbee-hive:**
```rust
let addr: SocketAddr = "127.0.0.1:9200".parse()?;  // NOT 8080 or 8090!
```

**Mock worker:**
```rust
let addr: SocketAddr = "127.0.0.1:8001".parse()?;
```

---

## Verification Commands

### Check if ports are in use
```bash
# Check queen-rbee
lsof -i :8080

# Check rbee-hive
lsof -i :9200

# Check workers
lsof -i :8001
lsof -i :8002
```

### Test connectivity
```bash
# Test queen-rbee
curl http://localhost:8080/health

# Test rbee-hive (on remote node)
curl http://workstation.home.arpa:9200/v1/health

# Test worker
curl http://workstation.home.arpa:8001/v1/health
```

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Using 8080 for rbee-hive
**Wrong:**
```rust
let rbee_hive_url = "http://127.0.0.1:8080".to_string();  // This is queen-rbee!
```

**Correct:**
```rust
let rbee_hive_url = "http://127.0.0.1:9200".to_string();  // rbee-hive port
```

### ❌ Mistake 2: Using 8090 for rbee-hive
**Wrong:**
```rust
let rbee_hive_url = "http://127.0.0.1:8090".to_string();  // Made up number!
```

**Correct:**
```rust
let rbee_hive_url = "http://127.0.0.1:9200".to_string();  // From spec
```

### ❌ Mistake 3: Copying from old handoffs
**Wrong approach:**
- Read TEAM-043's handoff
- Copy port numbers
- Don't check spec

**Correct approach:**
- Read normative spec (`test-001.md`)
- Verify port numbers
- Cross-reference with this document

---

## References

**Normative Spec:**
- `bin/.specs/.gherkin/test-001.md` (lines 231, 243, 254)

**Architecture Docs:**
- `bin/.specs/LIFECYCLE_CLARIFICATION.md`
- `bin/.specs/ARCHITECTURE_MODES.md`
- `bin/.specs/FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md`

**Mistake Analysis:**
- `test-harness/bdd/MISTAKES_AND_CORRECTIONS.md`
- `test-harness/bdd/HISTORICAL_MISTAKES_ANALYSIS.md`

---

**This is the ONLY correct port allocation. Always verify against this document!**
```

### Phase 6: Verify Code is Correct (Day 2)

#### Task 6.1: Verify queen-rbee Code
**File:** `bin/queen-rbee/src/http/inference.rs`

**Check line 57:**
```rust
"http://127.0.0.1:9200".to_string()  // Should be 9200, not 8080 or 8090
```

**Status:** ✅ Already correct (TEAM-053 fixed this)

#### Task 6.2: Search for Other Port References
```bash
# Search for potential issues
cd /home/vince/Projects/llama-orch

# Find all 8080 references (check if they're for rbee-hive)
grep -rn "8080" bin/queen-rbee/src/ | grep -i hive

# Find all 8090 references (should be none!)
grep -rn "8090" bin/

# Find all 9200 references (should be for rbee-hive)
grep -rn "9200" bin/
```

**Expected results:**
- No 8090 references in code
- 9200 used for rbee-hive connections
- 8080 used for queen-rbee server

### Phase 7: Update Mock Server Implementation (Day 2)

#### Task 7.1: Create Mock rbee-hive on Port 9200
**File:** `test-harness/bdd/src/mock_rbee_hive.rs` (CREATE NEW)

```rust
// TEAM-054: Mock rbee-hive server for BDD tests
// Port: 9200 (per normative spec, NOT 8080 or 8090!)

use axum::{routing::{get, post}, Router, Json};
use std::net::SocketAddr;
use anyhow::Result;

pub async fn start_mock_rbee_hive() -> Result<()> {
    let app = Router::new()
        .route("/v1/health", get(handle_health))
        .route("/v1/workers/spawn", post(handle_spawn_worker))
        .route("/v1/workers/ready", post(handle_worker_ready))
        .route("/v1/workers/list", get(handle_list_workers));
    
    // CRITICAL: Port 9200, not 8080 or 8090!
    let addr: SocketAddr = "127.0.0.1:9200".parse()?;
    tracing::info!("🐝 Starting mock rbee-hive on {} (port 9200 per spec)", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

async fn handle_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "alive",
        "version": "0.1.0-mock"
    }))
}

async fn handle_spawn_worker(Json(req): Json<serde_json::Value>) -> Json<serde_json::Value> {
    tracing::info!("Mock rbee-hive: spawning worker for request: {:?}", req);
    
    Json(serde_json::json!({
        "worker_id": "mock-worker-123",
        "url": "http://127.0.0.1:8001",
        "state": "loading"
    }))
}

async fn handle_worker_ready(Json(req): Json<serde_json::Value>) -> Json<serde_json::Value> {
    tracing::info!("Mock rbee-hive: worker ready callback: {:?}", req);
    
    Json(serde_json::json!({
        "success": true
    }))
}

async fn handle_list_workers() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "workers": []
    }))
}
```

#### Task 7.2: Start Mock Server Before Tests
**File:** `test-harness/bdd/src/main.rs`

```rust
// TEAM-054: Start mock servers before tests
#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    
    // Start global queen-rbee (port 8080)
    crate::steps::global_queen::start_global_queen_rbee().await;
    
    // TEAM-054: Start mock rbee-hive (port 9200, NOT 8080 or 8090!)
    tokio::spawn(async {
        if let Err(e) = crate::mock_rbee_hive::start_mock_rbee_hive().await {
            tracing::error!("Mock rbee-hive failed: {}", e);
        }
    });
    
    // Wait for mock servers to start
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    
    tracing::info!("✅ Mock servers ready:");
    tracing::info!("   - queen-rbee: http://127.0.0.1:8080");
    tracing::info!("   - rbee-hive:  http://127.0.0.1:9200");
    
    // Run tests
    World::cucumber()
        .run("tests/features")
        .await;
}
```

---

## 📋 Complete Checklist

### Documentation Updates
- [ ] Fix `TEAM_053_SUMMARY.md` (8 replacements: 8090 → 9200)
- [ ] Fix `HANDOFF_TO_TEAM_054.md` (4 replacements: 8090 → 9200)
- [ ] Fix `HANDOFF_TO_TEAM_047.md` (4 replacements: 8080 → 9200 for rbee-hive)
- [ ] Fix `TEAM_047_SUMMARY.md` (1 replacement: 8080 → 9200)
- [ ] Fix `HANDOFF_TO_TEAM_048.md` (2 replacements: 8080 → 9200 for rbee-hive)
- [ ] Add correction notes to all updated files
- [ ] Add historical notes to TEAM-043 documents
- [ ] Create `PORT_ALLOCATION.md` reference document

### Code Verification
- [ ] Verify `bin/queen-rbee/src/http/inference.rs` uses 9200 ✅ (already correct)
- [ ] Search for any remaining 8080 references to rbee-hive
- [ ] Search for any 8090 references (should be none!)
- [ ] Verify all 9200 references are for rbee-hive

### Mock Server Implementation
- [ ] Create `test-harness/bdd/src/mock_rbee_hive.rs` on port 9200
- [ ] Update `test-harness/bdd/src/main.rs` to start mock server
- [ ] Add mock server module to `test-harness/bdd/src/lib.rs`
- [ ] Test mock server starts correctly

### Testing
- [ ] Run BDD tests: `cargo run --bin bdd-runner`
- [ ] Verify no port conflicts
- [ ] Check test output for correct port numbers
- [ ] Verify mock rbee-hive receives requests

### Final Verification
- [ ] All documentation references 9200 for rbee-hive
- [ ] All code uses 9200 for rbee-hive
- [ ] Mock servers use correct ports
- [ ] Tests pass with correct ports
- [ ] Create TEAM_054_SUMMARY.md documenting fixes

---

## 🎯 Success Criteria

### Minimum Success (P0)
- [ ] All handoff documents corrected
- [ ] PORT_ALLOCATION.md created
- [ ] Code verified correct
- [ ] No remaining 8080 or 8090 references to rbee-hive

### Target Success (P0 + P1)
- [ ] Mock rbee-hive implemented on port 9200
- [ ] Tests run without port conflicts
- [ ] 48+ scenarios passing (up from 42)

### Stretch Goals
- [ ] All 62 scenarios passing
- [ ] CI check to prevent future port mistakes
- [ ] Automated port validation script

---

## 🚨 Critical Rules

### Rule 1: Always Check the Normative Spec
**Before making ANY architectural decision:**
1. Read `bin/.specs/.gherkin/test-001.md`
2. Search for the specific detail (e.g., port numbers)
3. Verify against multiple sources
4. Document your findings

### Rule 2: Don't Trust Handoffs Blindly
**Handoffs can be wrong!**
- TEAM-043: Wrong (old architecture)
- TEAM-047: Wrong (didn't update)
- TEAM-048: Wrong (copied from TEAM-047)
- TEAM-053: Wrong (made assumptions)

**Always verify against the normative spec!**

### Rule 3: Port Allocation is Sacred
**These ports are NORMATIVE:**
- queen-rbee: 8080
- rbee-hive: 9200
- workers: 8001+

**Never change these without updating the spec!**

### Rule 4: Add Correction Notes
**When fixing mistakes:**
1. Add a correction note at the top of the file
2. Explain what was wrong
3. Reference the normative spec
4. Sign with your team number

---

## 📚 Reference Documents

### Must Read (Priority Order)
1. **`bin/.specs/.gherkin/test-001.md`** - Normative spec (lines 231, 243, 254)
2. **`test-harness/bdd/PORT_ALLOCATION.md`** - Port reference (create this!)
3. **`test-harness/bdd/HISTORICAL_MISTAKES_ANALYSIS.md`** - Mistake timeline
4. **`test-harness/bdd/MISTAKES_AND_CORRECTIONS.md`** - TEAM-053's mistakes

### Architecture References
- `bin/.specs/LIFECYCLE_CLARIFICATION.md`
- `bin/.specs/ARCHITECTURE_MODES.md`
- `bin/.specs/FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md`
- `bin/.specs/CRITICAL_RULES.md`

### Historical Context
- `HANDOFF_TO_TEAM_043_FINAL.md` - Old architecture (before queen-rbee)
- `bin/.specs/COMMIT_6592850_CHANGELOG.md` - TEAM-037/TEAM-038 changes

---

## 🔧 Implementation Guide

### Step-by-Step Process

#### Day 1 Morning: Documentation Fixes
1. Open each handoff document
2. Find all port references
3. Verify if they're for rbee-hive
4. Replace with correct port (9200)
5. Add correction notes
6. Commit changes

#### Day 1 Afternoon: Create Reference
1. Create `PORT_ALLOCATION.md`
2. Document all ports
3. Add architecture diagram
4. Add verification commands
5. Add common mistakes section

#### Day 2 Morning: Code Verification
1. Search codebase for port references
2. Verify all are correct
3. Fix any issues found
4. Run tests to verify

#### Day 2 Afternoon: Mock Server
1. Create `mock_rbee_hive.rs`
2. Implement endpoints
3. Update `main.rs` to start server
4. Test mock server works
5. Run full BDD suite

---

## 🐛 Debugging Tips

### If Tests Still Fail After Fixes

**Check 1: Port conflicts**
```bash
lsof -i :8080  # Should be queen-rbee
lsof -i :9200  # Should be mock rbee-hive
lsof -i :8001  # Should be mock worker (if implemented)
```

**Check 2: Mock server started**
```bash
curl http://localhost:9200/v1/health
# Should return: {"status":"alive","version":"0.1.0-mock"}
```

**Check 3: Code uses correct port**
```bash
grep -rn "rbee_hive_url" bin/queen-rbee/src/
# Should show 9200, not 8080 or 8090
```

**Check 4: Documentation is consistent**
```bash
grep -rn "8090" test-harness/bdd/*.md
# Should return NO results after fixes
```

---

## 📊 Expected Impact

### Before Fixes (Current State)
- ❌ Documentation references wrong ports (8080, 8090)
- ✅ Code is correct (9200)
- ❌ No mock rbee-hive server
- 🟡 42/62 scenarios passing

### After Fixes (Target State)
- ✅ All documentation references correct port (9200)
- ✅ Code verified correct
- ✅ Mock rbee-hive server running
- ✅ 48+ scenarios passing (target: 54+)

---

## 🎓 Lessons for Future Teams

### Lesson 1: Specs Are Normative, Handoffs Are Not
**Always verify against:**
- `bin/.specs/.gherkin/test-001.md`
- `bin/.specs/LIFECYCLE_CLARIFICATION.md`
- `bin/.specs/CRITICAL_RULES.md`

**Never trust blindly:**
- Handoff documents
- Summary documents
- Other teams' assumptions

### Lesson 2: Architecture Changes Need Propagation
**When architecture changes:**
1. Update the normative spec
2. Update all handoffs
3. Mark old handoffs as obsolete
4. Create migration guide

### Lesson 3: Port Allocations Need Documentation
**Create reference documents:**
- PORT_ALLOCATION.md (you're creating this!)
- NETWORK_TOPOLOGY.md (future work)
- SERVICE_REGISTRY.md (future work)

### Lesson 4: Mistakes Propagate Through Handoffs
**Break the chain:**
1. Verify before copying
2. Add correction notes
3. Reference normative specs
4. Don't assume previous teams were correct

---

## 🔄 Handoff to TEAM-055

**After completing this work, create:**
1. `TEAM_054_SUMMARY.md` - What you fixed
2. `HANDOFF_TO_TEAM_055.md` - Next priorities
3. Update `PORT_ALLOCATION.md` if needed

**Include in handoff:**
- All port fixes completed
- Mock server implementation
- Test results (scenarios passing)
- Any remaining issues

---

## 🎯 Final Notes

**This is a CRITICAL fix!** Port confusion has affected multiple teams and caused wasted time. By fixing this systematically, you'll:

1. ✅ Correct all documentation
2. ✅ Create definitive reference
3. ✅ Prevent future mistakes
4. ✅ Unblock test progress

**You've got this, TEAM-054!** 🚀

**Remember:**
- queen-rbee: 8080
- rbee-hive: 9200 ← THIS IS THE ANSWER!
- workers: 8001+

**When in doubt, check the spec!**

---

**TEAM-053 signing off. Good luck, TEAM-054!** 🐝
