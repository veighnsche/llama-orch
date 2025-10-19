# TEAM-130F: rbee-keeper BINARY + CRATES PLAN

**Phase:** Phase 3 Implementation Planning  
**Date:** 2025-10-19  
**Team:** TEAM-130F  
**Status:** 📋 PLAN (Future Architecture)

---

## 🎯 MISSION

Define the **PLANNED** architecture for rbee-keeper after Phase 3 consolidation work is complete.

This is NOT current state - this is the TARGET architecture.

---

## 📊 BINARY OVERVIEW (PLANNED)

**Name:** `rbee-keeper`  
**Type:** CLI binary (thin client)  
**Role:** User-facing command-line interface  
**Architecture:** Thin client - ALL logic in queen-rbee

### Key Principles

1. **NO SSH** - All communication via HTTP to queen-rbee
2. **NO orchestration logic** - Just formats requests and displays responses
3. **Auto-starts queen-rbee** - If not running, starts it automatically
4. **Stateless** - No local state, no databases, no registries

### Current vs Planned LOC

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Total LOC** | 1,252 | ~985 | **-267 LOC** |
| **Files** | 13 | 10 | -3 files |
| **Shared crate deps** | 1 | 4 | +3 deps |

**Removals:**
- ❌ `src/ssh.rs` (14 LOC) - Architectural violation
- ❌ `src/commands/hive.rs` (84 LOC) - Bypasses queen
- ❌ `src/pool_client.rs` (115 LOC) - Legacy
- ❌ `src/queen_lifecycle.rs` (75 LOC) - Use daemon-lifecycle
- ❌ SSH usage in `commands/logs.rs` (24 LOC)
- ❌ Manual HTTP client code (~30 LOC) - Use rbee-http-client

**Additions:**
- ✅ Use `daemon-lifecycle` crate (replace queen_lifecycle.rs)
- ✅ Use `rbee-http-client` crate (replace manual reqwest)
- ✅ Use `rbee-types` crate (for BeehiveNode)

---

## 📦 PLANNED CRATE STRUCTURE

### Crate 1: `rbee-keeper-config`

**Location:** `bin/rbee-keeper/src/config.rs`  
**LOC:** ~44 (no change)  
**Purpose:** Configuration management

```rust
pub struct Config {
    pub queen_url: String,
    pub timeout_secs: u64,
}

impl Config {
    pub fn from_env() -> Self;
    pub fn default() -> Self;
}
```

**Dependencies:**
- None (pure Rust)

**Status:** ✅ Keep as-is

---

### Crate 2: `rbee-keeper-commands`

**Location:** `bin/rbee-keeper/src/commands/`  
**LOC:** ~600 (reduced from ~750)  
**Purpose:** Command handlers

**Files:**
```
commands/
├─ mod.rs           (6 LOC)
├─ infer.rs         (120 LOC) - Inference requests
├─ setup.rs         (280 LOC) - Node management
├─ workers.rs       (150 LOC) - Worker operations
├─ logs.rs          (24 LOC) - REVISED: via queen HTTP, no SSH
└─ install.rs       (98 LOC) - Installation
```

**Changes:**

**REMOVE:**
- ❌ `commands/hive.rs` (84 LOC) - Delete entirely

**REVISE:**
- ⚠️ `commands/logs.rs` - Remove SSH, use queen HTTP endpoint

**Example (logs.rs BEFORE):**
```rust
// WRONG - Direct SSH
use crate::ssh;
ssh::execute_remote_command_streaming(host, "journalctl -u rbee-hive")?;
```

**Example (logs.rs AFTER):**
```rust
// CORRECT - Via queen-rbee HTTP
use rbee_http_client::RbeeHttpClient;

let client = RbeeHttpClient::with_base_url("http://localhost:8080");
let logs: LogsResponse = client
    .get_json(&format!("/v2/logs?node={}&follow={}", node, follow))
    .await?;

// Stream logs to stdout
for line in logs.lines {
    println!("{}", line);
}
```

**Dependencies:**
- `rbee-http-client` (NEW - shared crate)
- `rbee-types` (NEW - for BeehiveNode, request/response types)
- `input-validation` (existing)
- `anyhow`, `colored`, `serde`

**Status:** ⚠️ Needs revision (remove SSH, use shared crates)

---

### Crate 3: ~~rbee-keeper-pool-client~~ **DELETE**

**Location:** `bin/rbee-keeper/src/pool_client.rs`  
**LOC:** ~115  
**Status:** ❌ **DELETE**

**Why DELETE:**
- Legacy pool-manager code
- Creates confusion and drift
- Not actually compatible
- No users depend on it

**Action:** `rm bin/rbee-keeper/src/pool_client.rs`

---

### Crate 4: `rbee-keeper-cli`

**Location:** `bin/rbee-keeper/src/cli.rs`  
**LOC:** ~180 (no change)  
**Purpose:** CLI argument parsing (clap)

```rust
#[derive(Parser)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

pub enum Commands {
    Infer(InferArgs),
    Setup(SetupAction),
    Workers(WorkersAction),
    Logs(LogsArgs),
    Install(InstallArgs),
}
```

**Status:** ✅ Keep as-is

---

## 🔗 SHARED CRATE DEPENDENCIES (PLANNED)

### New Dependencies (Phase 3)

1. **daemon-lifecycle** (NEW)
   - Replaces: `src/queen_lifecycle.rs` (75 LOC)
   - Usage: Auto-start queen-rbee if not running
   - Savings: ~60 LOC

2. **rbee-http-client** (NEW)
   - Replaces: Manual `reqwest::Client::new()` calls (6 occurrences)
   - Usage: All HTTP requests to queen-rbee
   - Savings: ~30 LOC

3. **rbee-types** (NEW)
   - Provides: `BeehiveNode`, `WorkerState`, request/response types
   - Usage: Type-safe communication with queen-rbee
   - Savings: ~30 LOC (avoid type duplication)

### Existing Dependencies (Keep)

4. **input-validation**
   - Current usage: 2 call sites (node name validation)
   - Status: ✅ Keep

5. **auth-min**
   - Current usage: None (rbee-keeper is CLI, not HTTP server)
   - Status: ❌ Not needed

---

## 📋 BINARY STRUCTURE (PLANNED)

```
bin/rbee-keeper/
├─ src/
│  ├─ main.rs                    (~12 LOC) - Entry point
│  ├─ cli.rs                     (~180 LOC) - Clap parsing
│  ├─ config.rs                  (~44 LOC) - Config
│  └─ commands/
│     ├─ mod.rs                  (6 LOC)
│     ├─ infer.rs                (~120 LOC)
│     ├─ setup.rs                (~280 LOC)
│     ├─ workers.rs              (~150 LOC)
│     ├─ logs.rs                 (~24 LOC) - REVISED: no SSH
│     └─ install.rs              (~98 LOC)
├─ Cargo.toml
└─ README.md
```

**Removed Files:**
- ❌ `src/ssh.rs` (14 LOC) - Architectural violation
- ❌ `src/queen_lifecycle.rs` (75 LOC) - Replaced by daemon-lifecycle
- ❌ `src/commands/hive.rs` (84 LOC) - Bypasses queen
- ❌ `src/pool_client.rs` (115 LOC) - Legacy

**Total Files:** 10 (down from 13)

---

## 🎯 COMMAND MAPPING (PLANNED)

### Commands That Stay

| Command | Current | Planned | Change |
|---------|---------|---------|--------|
| `rbee infer` | ✅ Via queen HTTP | ✅ Via queen HTTP | Use rbee-http-client |
| `rbee setup add-node` | ✅ Via queen HTTP | ✅ Via queen HTTP | Use rbee-types |
| `rbee setup list-nodes` | ✅ Via queen HTTP | ✅ Via queen HTTP | Use rbee-types |
| `rbee setup remove-node` | ✅ Via queen HTTP | ✅ Via queen HTTP | Use rbee-types |
| `rbee workers list` | ✅ Via queen HTTP | ✅ Via queen HTTP | Use rbee-http-client |
| `rbee workers health` | ✅ Via queen HTTP | ✅ Via queen HTTP | Use rbee-http-client |
| `rbee workers shutdown` | ✅ Via queen HTTP | ✅ Via queen HTTP | Use rbee-http-client |
| `rbee install` | ✅ Local files | ✅ Local files | No change |

### Commands That Change

| Command | Current (WRONG) | Planned (CORRECT) |
|---------|-----------------|-------------------|
| `rbee logs` | ❌ Direct SSH | ✅ Via queen HTTP `/v2/logs` |

### Commands That Are REMOVED

| Command | Current (WRONG) | Planned |
|---------|-----------------|---------|
| `rbee hive models` | ❌ Direct SSH | ❌ REMOVED (use `rbee models`) |
| `rbee hive workers` | ❌ Direct SSH | ❌ REMOVED (use `rbee workers`) |
| `rbee hive status` | ❌ Direct SSH | ❌ REMOVED (use `rbee status`) |

**Why removed:** These bypassed queen-rbee orchestration. All hive operations now go through queen.

---

## 🔧 IMPLEMENTATION PLAN

### Phase 1: Remove Violations (Day 1)

**Delete files:**
```bash
rm bin/rbee-keeper/src/ssh.rs
rm bin/rbee-keeper/src/commands/hive.rs
rm bin/rbee-keeper/src/pool_client.rs
```

**Update Cargo.toml:**
```toml
# Remove (if present)
# ssh2 = "..."

# Keep existing
input-validation = { path = "../shared-crates/input-validation" }
```

**Verify:**
```bash
cargo check --bin rbee-keeper
# Should compile without ssh.rs and hive.rs
```

---

### Phase 2: Integrate Shared Crates (Day 2)

**Update Cargo.toml:**
```toml
[dependencies]
# NEW: Phase 3 shared crates
daemon-lifecycle = { path = "../shared-crates/daemon-lifecycle" }
rbee-http-client = { path = "../shared-crates/rbee-http-client" }
rbee-types = { path = "../shared-crates/rbee-types" }

# Existing
input-validation = { path = "../shared-crates/input-validation" }
anyhow = "1.0"
clap = { version = "4.5", features = ["derive"] }
colored = "2.1"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1.40", features = ["full"] }
```

**Replace queen_lifecycle.rs:**
```rust
// OLD: src/queen_lifecycle.rs (75 LOC)
pub async fn ensure_queen_rbee_running(client: &reqwest::Client, queen_url: &str) -> Result<()> {
    // ... 75 lines of lifecycle logic
}

// NEW: Use daemon-lifecycle crate
use daemon_lifecycle::DaemonLifecycle;

pub async fn ensure_queen_rbee_running(queen_url: &str) -> Result<()> {
    let lifecycle = DaemonLifecycle::with_health_check(
        "queen-rbee",
        queen_url,
        vec![
            "--port".to_string(),
            "8080".to_string(),
            "--database".to_string(),
            "/tmp/queen-rbee-ephemeral.db".to_string(),
        ],
    );
    
    lifecycle.ensure_running().await?;
    Ok(())
}
```

**Replace manual HTTP clients:**
```rust
// OLD: Manual reqwest
let client = reqwest::Client::new();
let response = client.post(&url).json(&request).send().await?;
let result: AddNodeResponse = response.json().await?;

// NEW: Use rbee-http-client
use rbee_http_client::RbeeHttpClient;
use rbee_types::{AddNodeRequest, AddNodeResponse};

let client = RbeeHttpClient::with_base_url("http://localhost:8080");
let result: AddNodeResponse = client
    .post_json("/v2/registry/beehives/add", &request)
    .await?;
```

---

### Phase 3: Revise logs.rs (Day 2)

**OLD (WRONG - Direct SSH):**
```rust
// commands/logs.rs (24 LOC with SSH)
use crate::ssh;

pub async fn handle(node: String, follow: bool) -> Result<()> {
    let command = if follow {
        "journalctl -u rbee-hive -f"
    } else {
        "journalctl -u rbee-hive -n 100"
    };
    
    ssh::execute_remote_command_streaming(&node, command)?;
    Ok(())
}
```

**NEW (CORRECT - Via queen HTTP):**
```rust
// commands/logs.rs (24 LOC without SSH)
use rbee_http_client::RbeeHttpClient;
use rbee_types::LogsResponse;

pub async fn handle(node: String, follow: bool) -> Result<()> {
    let client = RbeeHttpClient::with_base_url("http://localhost:8080");
    
    // Queen-rbee handles SSH to node
    let logs: LogsResponse = client
        .get_json(&format!("/v2/logs?node={}&follow={}", node, follow))
        .await?;
    
    // Display logs
    for line in logs.lines {
        println!("{}", line);
    }
    
    Ok(())
}
```

---

### Phase 4: Testing (Day 3)

**Unit tests:**
```bash
cargo test --bin rbee-keeper
```

**Integration tests:**
```bash
# Test auto-start queen-rbee
rbee infer "test prompt"  # Should auto-start queen if not running

# Test node management
rbee setup add-node test-node --host localhost --user vince
rbee setup list-nodes
rbee setup remove-node test-node

# Test worker operations
rbee workers list
rbee workers health

# Test logs (via queen, not SSH)
rbee logs --node test-node
```

**Verify no SSH:**
```bash
# Should NOT find any SSH usage
grep -r "ssh::" bin/rbee-keeper/src/
grep -r "Command::new(\"ssh\")" bin/rbee-keeper/src/
# Both should return no results
```

---

## ✅ ACCEPTANCE CRITERIA

### Must Have

1. ✅ No SSH code in rbee-keeper (0 occurrences)
2. ✅ No `commands/hive.rs` (deleted)
3. ✅ Uses `daemon-lifecycle` for queen startup
4. ✅ Uses `rbee-http-client` for all HTTP requests
5. ✅ Uses `rbee-types` for shared types
6. ✅ All commands route through queen-rbee
7. ✅ `rbee logs` works via queen HTTP (no direct SSH)
8. ✅ All tests pass
9. ✅ Binary compiles without warnings

### Should Have

10. ✅ Consistent error handling (via rbee-http-client)
11. ✅ Type-safe communication (via rbee-types)
12. ✅ Reduced LOC (~1,100 from 1,252)

### Nice to Have

13. ⚠️ Better CLI help messages
14. ⚠️ Progress indicators for long operations
15. ⚠️ Colored output improvements

---

## 📊 METRICS (PLANNED)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Total LOC** | 1,252 | ~985 | **-267 LOC** |
| **Files** | 13 | 10 | -3 files |
| **Crates** | 5 | 4 | -1 crate |
| **SSH usage** | 3 files | 0 files | **-3 files** |
| **HTTP clients** | Manual (6×) | Shared crate | Consistent |
| **Type definitions** | Local | Shared | Type-safe |

**LOC Breakdown:**
- Remove ssh.rs: -14 LOC
- Remove hive.rs: -84 LOC
- Remove pool_client.rs: -115 LOC
- Remove queen_lifecycle.rs: -75 LOC (replaced by crate usage)
- Remove manual HTTP: -30 LOC
- Add shared crate usage: +51 LOC
- **Net savings: -267 LOC**

---

## 🎯 DEPENDENCIES (PLANNED)

### Cargo.toml (After Phase 3)

```toml
[package]
name = "rbee-keeper"
version = "0.1.0"
edition = "2021"
license = "GPL-3.0-or-later"

[[bin]]
name = "rbee"
path = "src/main.rs"

[dependencies]
# Phase 3 NEW: Shared crates
daemon-lifecycle = { path = "../shared-crates/daemon-lifecycle" }
rbee-http-client = { path = "../shared-crates/rbee-http-client" }
rbee-types = { path = "../shared-crates/rbee-types" }

# Existing: Shared crates
input-validation = { path = "../shared-crates/input-validation" }

# External dependencies
anyhow = "1.0"
clap = { version = "4.5", features = ["derive"] }
colored = "2.1"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1.40", features = ["full"] }
chrono = "0.4"

[dev-dependencies]
tempfile = "3.8"
```

**Removed:**
- ❌ `ssh2` (if it was there)
- ❌ Any SSH-related dependencies

**Added:**
- ✅ `daemon-lifecycle` (queen startup)
- ✅ `rbee-http-client` (HTTP wrapper)
- ✅ `rbee-types` (shared types)

---

## 🚀 MIGRATION CHECKLIST

### Pre-Migration

- [ ] Read TEAM_130E_CRITICAL_CORRECTIONS.md
- [ ] Verify daemon-lifecycle crate exists
- [ ] Verify rbee-http-client crate exists
- [ ] Verify rbee-types crate exists
- [ ] Backup current rbee-keeper code

### Migration Steps

- [ ] Delete `src/ssh.rs`
- [ ] Delete `src/commands/hive.rs`
- [ ] Delete `src/queen_lifecycle.rs`
- [ ] Update Cargo.toml (add shared crates)
- [ ] Update `commands/logs.rs` (remove SSH)
- [ ] Update all commands (use rbee-http-client)
- [ ] Update all commands (use rbee-types)
- [ ] Update main.rs (use daemon-lifecycle)
- [ ] Run `cargo check`
- [ ] Run `cargo test`
- [ ] Manual testing (all commands)

### Post-Migration

- [ ] Verify no SSH code remains (`grep -r ssh`)
- [ ] Verify all tests pass
- [ ] Update README.md
- [ ] Update CHANGELOG.md
- [ ] Tag release: `v0.2.0-phase3`

---

## 📝 NOTES

### Why This Architecture?

1. **Thin client principle** - rbee-keeper has no business logic
2. **Single source of truth** - Queen-rbee makes all decisions
3. **Type safety** - Shared types prevent schema drift
4. **Consistency** - Shared HTTP client ensures uniform error handling
5. **Maintainability** - Less code, fewer bugs

### pool_client.rs is DELETED

**Status:** ❌ DELETED

Legacy pool-manager code has been removed. No backward compatibility needed - the code was never actually compatible.

---

**Status:** 📋 PLAN COMPLETE  
**Team:** TEAM-130F  
**Next:** queen-rbee plan  
**LOC Impact:** -152 LOC (1,252 → 1,100)
