# TEAM-130C: ARCHITECTURAL VIOLATIONS SUMMARY

**Date:** 2025-10-19  
**Status:** 🔴 CRITICAL  
**Impact:** ALL Phase 2 PART1 documents need correction

---

## 🚨 ROOT CAUSE: AI CODER DRIFT

**Problem:** 100+ documents by different AI teams created code that doesn't match specifications.

**Teams documented what EXISTS, not what SHOULD exist.**

---

## 🔴 THREE CRITICAL VIOLATIONS

### Violation #1: rbee-keeper has SSH ❌ (CONFIRMED)

**Current (WRONG):**
```
rbee-keeper (CLI)
    └─ ssh-client crate (14 LOC) - src/ssh.rs
        └─ Direct SSH to rbee-hive (bypasses queen!)
            └─ Commands: rbee hive models|workers|status
```

**Evidence:**
```rust
// src/ssh.rs (14 LOC)
pub fn execute_remote_command_streaming(host: &str, command: &str) -> Result<()> {
    Command::new("ssh").arg(host).arg(command).status()?;
}

// src/commands/hive.rs (84 LOC)
fn handle_models(action: ModelsAction, host: &str) -> Result<()> {
    let command = format!("rbee-hive models {}", ...);
    ssh::execute_remote_command_streaming(host, &command)?; // ❌ Bypasses queen!
}
```

**Commands that violate:**
- `rbee hive models download llama-7b --host gpu-0` (bypasses queen)
- `rbee hive models list --host gpu-0` (bypasses queen)
- `rbee hive worker spawn --host gpu-0` (bypasses queen)
- `rbee hive worker list --host gpu-0` (bypasses queen)
- `rbee hive git pull --host gpu-0` (bypasses queen)
- `rbee hive status --host gpu-0` (bypasses queen)

**Correct Architecture:**
```
rbee-keeper (CLI - NO SSH!)
    └─ HTTP ONLY to queen-rbee
        └─ queen-rbee has SSH for hive management
```

**Files to Delete:**
- `bin/rbee-keeper/src/ssh.rs` (14 LOC)
- `bin/rbee-keeper/src/commands/hive.rs` (84 LOC)
- Remove SSH from `bin/rbee-keeper/src/commands/logs.rs` (24 LOC)

**Commands to Replace:**
- `rbee hive models` → `rbee models` (via queen `/v2/models/*`)
- `rbee hive worker` → `rbee workers` (via queen `/v2/workers/*`)
- `rbee hive status` → `rbee status` (via queen `/v2/status`)
- `rbee logs --node X` → via queen `/v2/logs?node=X`

**Why this is critical:**
1. Duplicated SSH logic (queen will also need SSH)
2. Bypasses orchestration (queen doesn't track these operations)
3. Security risk (multiple SSH entry points)
4. Maintenance burden (changes in 2 places)

---

### Violation #2: rbee-hive has CLI ❌ (CONFIRMED)

**Current (WRONG):**
```
rbee-hive
    ├─ cli crate (719 LOC) - src/cli.rs + commands/
    │   ├─ commands/daemon.rs (348 LOC) - daemon mode
    │   ├─ commands/models.rs (118 LOC) - model commands
    │   ├─ commands/workers.rs (105 LOC) - worker commands
    │   ├─ commands/status.rs (74 LOC) - status command
    │   └─ cli.rs (68 LOC) - clap parsing
    └─ main.rs parses CLI args with subcommands
```

**Evidence:**
```rust
// src/cli.rs (68 LOC)
#[derive(Parser)]
pub enum Commands {
    Daemon(DaemonArgs),   // ✅ Keep
    Models(ModelsArgs),   // ❌ Remove
    Workers(WorkersArgs), // ❌ Remove
    Status(StatusArgs),   // ❌ Remove
}

// main.rs
match cli.command {
    Commands::Daemon(args) => daemon::handle(args).await?,
    Commands::Models(args) => models::handle(args).await?, // ❌ Never planned!
    Commands::Workers(args) => workers::handle(args).await?, // ❌ Never planned!
    Commands::Status(args) => status::handle(args).await?, // ❌ Never planned!
}
```

**Why CLI was never planned:**
- rbee-hive is controlled via HTTP by queen-rbee
- queen-rbee sends commands to hive HTTP API
- No human should interact with rbee-hive directly

**Correct Architecture:**
```
rbee-hive (DAEMON ONLY - NO CLI!)
    ├─ HTTP API ONLY (no CLI commands)
    └─ main.rs (~50 LOC - just start daemon)
```

**Files to Delete:**
- `bin/rbee-hive/src/commands/models.rs` (118 LOC)
- `bin/rbee-hive/src/commands/workers.rs` (105 LOC)
- `bin/rbee-hive/src/commands/status.rs` (74 LOC)
- Remove CLI parsing from `bin/rbee-hive/src/cli.rs` (keep only port/config)
- Remove CLI subcommands from `bin/rbee-hive/src/main.rs`

**Binary Becomes:**
```rust
// main.rs (~50 LOC)
#[derive(Parser)]
struct Args {
    #[arg(short, long, default_value = "8080")]
    port: u16,
    
    #[arg(short, long)]
    config: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    start_daemon(args.port, args.config).await
}
```

**All functionality moved to HTTP API:**
- Models: GET/POST `/v1/models/*`
- Workers: GET/POST `/v1/workers/*`
- Status: GET `/v1/health`, `/v1/status`

---

### Violation #3: queen-rbee is 76% MISSING ❌

**Current (2,015 LOC):**
- ✅ Beehive registry (200 LOC)
- ✅ Worker registry (153 LOC)
- ✅ HTTP server basic (897 LOC)
- ✅ Shutdown hives (158 LOC)
- ✅ SSH client for shutdown (76 LOC)

**Missing (8,300 LOC):**
- ❌ **START hives** (local + remote) - 800 LOC
- ❌ **Scheduler** (worker selection) - 1,200 LOC
- ❌ **Admission control** - 400 LOC
- ❌ **Job queue** (SQLite) - 500 LOC
- ❌ **Request router** - 600 LOC
- ❌ **Model provisioner** (coordination) - 500 LOC
- ❌ **Eviction policies** - 400 LOC
- ❌ **Retry & backoff** - 300 LOC
- ❌ **SSE relay** - 400 LOC
- ❌ **Health monitor** (hives) - 500 LOC
- ❌ **Metrics collector** - 300 LOC
- ❌ **Load balancing** - 400 LOC

**Completion:** 24% (2,015 / 10,315 LOC)

---

## 📊 IMPACT ON TEAM-130C DOCUMENTS

### Documents Affected:
1. ❌ `TEAM_130C_rbee-keeper_PART1` - ssh-client crate is WRONG
2. ❌ `TEAM_130C_rbee-hive_PART1` - cli crate is WRONG
3. ❌ `TEAM_130C_queen-rbee_PART1` - Missing 15 crates (~8,300 LOC)
4. ✅ `TEAM_130C_llm-worker_PART1` - Correct (no violations)

### Correction Required:
- **rbee-keeper:** Remove ssh-client (14 LOC), remove hive commands (84+24 LOC)
- **rbee-hive:** Remove cli crate (719 LOC), HTTP API only
- **queen-rbee:** Add 15 missing crates (~8,300 LOC)

---

## 🎯 CORRECTED SYSTEM ARCHITECTURE

```
┌────────────────────────────────────────────────────────────────┐
│ rbee-keeper (CLI - Thin Client)                                │
│  - NO SSH                                                      │
│  - HTTP to queen-rbee ONLY                                     │
│  - Commands: infer, workers, models, setup, install            │
│  - 5 crates: config, pool-client, queen-lifecycle, commands   │
│  - Total: 1,065 LOC (no SSH!)                                 │
└──────────────────────┬─────────────────────────────────────────┘
                       │ HTTP
                       ↓
┌────────────────────────────────────────────────────────────────┐
│ queen-rbee (Orchestrator - THE BRAIN)                          │
│  - Makes ALL intelligent decisions                             │
│  - 15 crates:                                                  │
│    1. registry (hive + worker)                                 │
│    2. hive-lifecycle (START/STOP local + SSH remote) ← NEW!   │
│    3. admission (quota, rate limit) ← NEW!                     │
│    4. queue (job queue SQLite) ← NEW!                          │
│    5. scheduler (worker selection + Rhai M2) ← NEW!            │
│    6. router (request routing) ← NEW!                          │
│    7. provisioner (model coordination) ← NEW!                  │
│    8. eviction (LRU policies) ← NEW!                           │
│    9. retry (backoff logic) ← NEW!                             │
│    10. sse-relay (token streaming) ← NEW!                      │
│    11. http-server (API endpoints)                             │
│    12. remote (SSH client) ← EXPAND from current               │
│    13. monitor (hive health) ← NEW!                            │
│    14. metrics (collection) ← NEW!                             │
│    15. auth (authentication)                                   │
│  - Total: ~10,315 LOC (2,015 exists + 8,300 missing)           │
└──────────────────────┬─────────────────────────────────────────┘
                       │ HTTP + SSH
                       ↓
┌────────────────────────────────────────────────────────────────┐
│ rbee-hive (Daemon - NO CLI!)                                   │
│  - HTTP API ONLY (no CLI commands)                             │
│  - 9 crates (removed cli crate):                               │
│    1. registry (worker state)                                  │
│    2. http-server (API endpoints)                              │
│    3. http-middleware (auth)                                   │
│    4. provisioner (model download)                             │
│    5. monitor (worker health)                                  │
│    6. resources (limits)                                       │
│    7. shutdown (graceful)                                      │
│    8. metrics (prometheus)                                     │
│    9. restart (policy)                                         │
│  - Binary: ~50 LOC (just start daemon)                         │
│  - Total: ~3,365 LOC (removed 719 LOC cli)                     │
└──────────────────────┬─────────────────────────────────────────┘
                       │ spawns
                       ↓
┌────────────────────────────────────────────────────────────────┐
│ llm-worker-rbee (Worker - Correct!)                            │
│  - 6 crates (no changes)                                       │
│  - inference-base stays in binary (NOT reusable)               │
│  - Total: 5,026 LOC                                            │
└────────────────────────────────────────────────────────────────┘
```

---

## 📋 CORRECTED LOC COUNTS

| Binary | Current LOC | Violations | Add Missing | Corrected LOC | Change |
|--------|-------------|------------|-------------|---------------|--------|
| rbee-keeper | 1,252 | Remove SSH (14) + hive (84) + logs SSH (24) | +300 (models, workers expand) | 1,430 | +178 LOC |
| rbee-hive | 4,184 | Remove CLI (297: models 118 + workers 105 + status 74) | None | 3,887 | -297 LOC |
| queen-rbee | 2,015 | None | +8,300 (15 missing crates) | 10,315 | +8,300 LOC |
| llm-worker | 5,026 | None | None | 5,026 | No change |
| **TOTAL** | **12,477** | **-419 LOC** | **+8,600 LOC** | **20,658** | **+8,181 LOC** |

**System Growth:** 65% larger than documented
- Violations removed: -419 LOC
- Missing functionality added: +8,600 LOC
- Net growth: +8,181 LOC (primarily queen-rbee)

---

## 🔧 CORRECTION ACTIONS

### Immediate (Phase 2):
1. ✅ Create `QUEEN_RBEE_COMPLETE_RESPONSIBILITIES.md` (DONE)
2. ✅ Create `ARCHITECTURAL_VIOLATIONS_SUMMARY.md` (THIS DOC)
3. ❌ Update `rbee-keeper_PART1` (remove SSH crate)
4. ❌ Update `rbee-hive_PART1` (remove CLI crate)
5. ❌ Update `queen-rbee_PART1` (add 15 missing crates)
6. ❌ Update `CROSS_BINARY_ANALYSIS` (reflect violations)

### Phase 3 (Library Analysis):
- Document why SSH is in queen-rbee (hive management)
- Document why NO SSH in rbee-keeper (thin client)
- Document why NO CLI in rbee-hive (daemon only)

### Phase 4 (Migration):
- Migration plan to REMOVE violations
- Migration plan to ADD missing queen functionality

---

## 🎯 KEY PRINCIPLES (CORRECTED)

### 1. rbee-keeper = Thin Client
- **ONLY** HTTP to queen-rbee
- **NO** SSH (never)
- **NO** direct hive communication
- **NO** orchestration logic
- Commands: infer, workers, models, setup, install
- Auto-starts queen-rbee if not running

### 2. queen-rbee = THE BRAIN
- Makes **ALL** intelligent decisions
- Orchestrates hives via HTTP + SSH
- Manages hive lifecycle (start/stop/monitor)
- Scheduler, admission, queue, router
- SSE relay to clients

### 3. rbee-hive = Dumb Daemon
- **NO** CLI (daemon only)
- **ONLY** HTTP API
- Executes queen commands
- Reports state upward
- Manages local workers

### 4. llm-worker = Executor
- Loads model
- Executes inference
- Streams tokens
- No decisions

---

**Status:** 🔴 CRITICAL - Major architectural violations discovered  
**Next:** Correct all PART1 documents with proper architecture  
**Impact:** Entire decomposition strategy needs revision
