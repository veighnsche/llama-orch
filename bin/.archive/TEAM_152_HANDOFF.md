# TEAM-152 Handoff Document

**From:** TEAM-151  
**To:** TEAM-152  
**Date:** 2025-10-20  
**Focus:** Queen Lifecycle - Auto-start queen-rbee when not running

---

## 🎯 Mission

Implement the **queen-lifecycle** functionality so rbee-keeper can automatically start queen-rbee when it's not running.

### Happy Flow Target

From `a_human_wrote_this.md` lines 11-19:

> **"if not then start the queen on port 8500"**  
> **"narration (bee keeper -> stdout): queen is asleep, waking queen."**  
> **"then the bee keeper polls the queen until she gives a healthy sign"**  
> **"narration (bee keeper): queen is awake and healthy."**

---

## ✅ What TEAM-151 Completed

### 1. rbee-keeper Health Check ✅
**Location:** `bin/00_rbee_keeper/src/health_check.rs`

**Function:**
```rust
pub async fn is_queen_healthy(base_url: &str) -> Result<bool>
```

**Returns:**
- `Ok(true)` - Queen is running and healthy
- `Ok(false)` - Queen is not running (connection refused)
- `Err(...)` - Other errors (timeout, etc.)

**Test Command:**
```bash
./target/debug/rbee-keeper test-health
# ❌ queen-rbee is not running (connection refused)
#    Start queen with: queen-rbee --port 8500
```

### 2. queen-rbee Health Endpoint ✅
**Location:** `bin/10_queen_rbee/src/http/health.rs`

**Endpoint:** `GET /health`  
**Port:** 8500 (default)  
**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

### 3. BDD Tests ✅
**Location:** `bin/00_rbee_keeper/bdd/tests/features/queen_health_check.feature`

**Scenarios:**
- ✅ Queen is not running (returns false)
- ✅ Queen is running and healthy (returns true)
- ✅ Custom port support

---

## 🚀 Your Mission: Implement Queen Lifecycle

### Target Crate
**Location:** `bin/05_rbee_keeper_crates/queen-lifecycle/`

**README exists:** `bin/05_rbee_keeper_crates/queen-lifecycle/README.md`

### Required Function

```rust
/// Ensure queen-rbee is running, auto-start if needed
///
/// # Happy Flow
/// 1. Check health using health_check::is_queen_healthy()
/// 2. If healthy → return Ok(())
/// 3. If not running:
///    - Print: "⚠️  queen-rbee not running, starting..."
///    - Spawn queen-rbee process using daemon-lifecycle
///    - Poll health until ready (with timeout)
///    - Print: "✅ queen-rbee is awake and healthy"
///
/// # Arguments
/// * `base_url` - Queen URL (e.g., "http://localhost:8500")
///
/// # Returns
/// * `Ok(())` - Queen is running (was already running or successfully started)
/// * `Err` - Failed to start queen or timeout waiting for health
pub async fn ensure_queen_running(base_url: &str) -> Result<()>
```

---

## 📋 Implementation Steps

### Step 1: Create Crate Structure

```bash
bin/05_rbee_keeper_crates/queen-lifecycle/
├── Cargo.toml
├── README.md (already exists)
├── src/
│   └── lib.rs
└── bdd/
    └── tests/features/queen_lifecycle.feature
```

### Step 2: Dependencies

**Cargo.toml:**
```toml
[dependencies]
# Shared crates (MUST BE IMPLEMENTED FIRST!)
daemon-lifecycle = { path = "../../99_shared_crates/daemon-lifecycle" }

# From parent crate
rbee-keeper = { path = "../.." }  # For health_check module

# Async
tokio = { version = "1", features = ["full"] }

# Error handling
anyhow = "1.0"
```

### Step 3: Implementation

**src/lib.rs:**
```rust
//! Queen lifecycle management
//!
//! Created by: TEAM-152
//! Date: 2025-10-20

use anyhow::Result;
use std::time::Duration;

pub async fn ensure_queen_running(base_url: &str) -> Result<()> {
    // 1. Check if queen is already running
    if rbee_keeper::health_check::is_queen_healthy(base_url).await? {
        return Ok(()); // Already running
    }
    
    // 2. Queen is not running, start it
    println!("⚠️  queen-rbee not running, starting...");
    
    // 3. Find queen-rbee binary
    let queen_binary = find_queen_binary()?;
    
    // 4. Spawn queen process using daemon-lifecycle
    spawn_queen(&queen_binary, 8500).await?;
    
    // 5. Poll health until ready
    poll_until_healthy(base_url, Duration::from_secs(30)).await?;
    
    // 6. Success!
    println!("✅ queen-rbee is awake and healthy");
    Ok(())
}

fn find_queen_binary() -> Result<std::path::PathBuf> {
    // TODO: Find queen-rbee in target/debug/ (dev mode)
    // For now, hardcoded path as per architecture docs
    todo!()
}

async fn spawn_queen(binary_path: &std::path::Path, port: u16) -> Result<()> {
    // TODO: Use daemon-lifecycle crate to spawn queen
    todo!()
}

async fn poll_until_healthy(base_url: &str, timeout: Duration) -> Result<()> {
    // TODO: Retry health check with exponential backoff
    // Use rbee-keeper-polling crate or implement here
    todo!()
}
```

---

## 🚨 CRITICAL: Blocking Dependencies

Before you can implement queen-lifecycle, these shared crates MUST exist:

### 1. `daemon-lifecycle` (HIGHEST PRIORITY)
**Location:** `bin/99_shared_crates/daemon-lifecycle/`

**Purpose:** Spawn and manage daemon processes

**Required API:**
```rust
pub struct DaemonManager;

impl DaemonManager {
    pub async fn spawn(
        binary_path: &Path,
        args: Vec<String>,
    ) -> Result<Child>;
}
```

**Extract from:** `old.rbee-keeper/src/queen_lifecycle.rs` lines 64-132

### 2. `rbee-keeper-polling` (Optional)
**Location:** `bin/05_rbee_keeper_crates/polling/`

**Purpose:** Retry logic with exponential backoff

**Can be simple:**
```rust
pub async fn poll_until<F, Fut>(
    check_fn: F,
    timeout: Duration,
) -> Result<()>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<bool>>,
{
    // Exponential backoff: 100ms, 200ms, 400ms, 800ms, ...
    todo!()
}
```

---

## 📊 Integration Points

### Where to Call `ensure_queen_running()`

**In rbee-keeper's infer command:**

**Location:** `bin/05_rbee_keeper_crates/commands/src/infer.rs` (when migrated)

```rust
pub async fn handle(model: String, prompt: String, ...) -> Result<()> {
    // Step 1: Ensure queen is running
    rbee_keeper_queen_lifecycle::ensure_queen_running("http://localhost:8500").await?;
    
    // Step 2: Submit job to queen
    // ... rest of infer logic
}
```

---

## 🧪 BDD Tests to Create

**Location:** `bin/05_rbee_keeper_crates/queen-lifecycle/bdd/tests/features/queen_lifecycle.feature`

```gherkin
Feature: Queen Lifecycle Management
  As rbee-keeper
  I need to ensure queen-rbee is running
  So I can submit jobs to it

  Scenario: Queen is already running
    Given queen-rbee is running on port 8500
    When I ensure queen is running
    Then it should return immediately without starting a new process
    And I should not see "waking queen" message

  Scenario: Queen is not running (auto-start)
    Given queen-rbee is not running
    When I ensure queen is running
    Then it should start queen-rbee process
    And I should see "queen is asleep, waking queen"
    And it should poll health until ready
    And I should see "queen is awake and healthy"
    And queen should be running on port 8500

  Scenario: Queen startup timeout
    Given queen-rbee binary is not available
    When I ensure queen is running
    Then it should fail with timeout error
    And I should see helpful error message
```

---

## 📝 Files TEAM-151 Created/Modified

### Created Files
1. `bin/00_rbee_keeper/src/health_check.rs` - Health probe function
2. `bin/00_rbee_keeper/bdd/tests/features/queen_health_check.feature` - BDD scenarios
3. `bin/00_rbee_keeper/bdd/src/steps/health_check_steps.rs` - Step definitions
4. `bin/00_rbee_keeper/HEALTH_CHECK_IMPLEMENTATION.md` - Documentation
5. `bin/00_rbee_keeper/MIGRATION_STATUS.md` - CLI migration status
6. `bin/10_queen_rbee/HTTP_FOLDER_WIRING.md` - HTTP wiring docs
7. `bin/10_queen_rbee/HEALTH_API_MIGRATION.md` - Health API docs
8. `bin/10_queen_rbee/CLEANUP_SUMMARY.md` - Cleanup notes

### Modified Files
1. `bin/00_rbee_keeper/src/main.rs` - Added health_check module, test-health command
2. `bin/00_rbee_keeper/Cargo.toml` - Added reqwest dependency
3. `bin/10_queen_rbee/src/main.rs` - Cleaned up, wired http module
4. `bin/10_queen_rbee/src/http/mod.rs` - Commented out non-health modules
5. `bin/10_queen_rbee/src/http/types.rs` - Simplified to HealthResponse only
6. `bin/10_queen_rbee/Cargo.toml` - Added serde, serde_json
7. `bin/15_queen_rbee_crates/health/src/lib.rs` - Migrated health handler
8. `bin/15_queen_rbee_crates/health/Cargo.toml` - Updated dependencies
9. `bin/00_rbee_keeper/bdd/src/steps/world.rs` - Added health check state
10. `bin/00_rbee_keeper/bdd/src/steps/mod.rs` - Added health_check_steps module
11. `bin/00_rbee_keeper/bdd/Cargo.toml` - Added reqwest dependency

---

## 🎯 Success Criteria for TEAM-152

### Minimum Viable Product (MVP)
- [ ] `ensure_queen_running()` function implemented
- [ ] Auto-starts queen when not running
- [ ] Polls health until ready (30s timeout)
- [ ] Prints correct narration messages
- [ ] BDD tests pass

### Stretch Goals
- [ ] Graceful error handling (binary not found, port in use, etc.)
- [ ] Configurable timeout
- [ ] PID file management
- [ ] Log file location
- [ ] Integration with rbee-keeper commands

---

## 📚 Reference Documents

### Architecture
- `bin/a_human_wrote_this.md` - Original happy flow (lines 11-19)
- `bin/a_chatGPT_5_refined_this.md` - Refined flow with narration
- `bin/a_Claude_Sonnet_4_5_refined_this.md` - Code-backed architecture

### Migration Plan
- `bin/MIGRATION_MASTER_PLAN.md` - Full migration plan
- `bin/WORK_UNITS_CHECKLIST.md` - Work unit tracking

### Existing Crate
- `bin/05_rbee_keeper_crates/queen-lifecycle/README.md` - Crate overview

### Old Code (for reference)
- `bin/old.rbee-keeper/src/queen_lifecycle.rs` - Old implementation

---

## 🤝 Handoff Checklist

- [x] Health check function implemented and tested
- [x] Queen health endpoint working on port 8500
- [x] BDD tests created and passing
- [x] Documentation written
- [x] Code signed by TEAM-151
- [x] Handoff document created for TEAM-152

---

## 💡 Tips for TEAM-152

1. **Start with `daemon-lifecycle` crate** - This is the foundation
2. **Keep it simple** - Hardcoded paths for dev mode are OK
3. **Test incrementally** - Test spawn, then health check, then polling
4. **Use existing health_check** - Don't reimplement, just call it
5. **Follow the narration** - Exact messages from happy flow docs
6. **Write BDD tests first** - They'll guide your implementation

---

## 🚀 Ready to Start!

TEAM-152, you have everything you need:
- ✅ Health check working
- ✅ Queen endpoint working  
- ✅ BDD test examples
- ✅ Clear requirements
- ✅ Reference code

**Your mission:** Make queen wake up automatically! 🐝

Good luck, TEAM-152! 🎉

---

**Signed:** TEAM-151  
**Date:** 2025-10-20  
**Status:** Handoff Complete ✅
