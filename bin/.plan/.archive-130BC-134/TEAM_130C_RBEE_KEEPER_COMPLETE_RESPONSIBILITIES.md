# TEAM-130C: rbee-keeper COMPLETE RESPONSIBILITIES

**Date:** 2025-10-19  
**Status:** 🟡 PARTIALLY CORRECT - Has violations  
**Role:** User Interface (Thin Client)

---

## 🎯 CORE PRINCIPLE

**rbee-keeper is the USER INTERFACE to queen-rbee**

```
User → rbee-keeper (CLI) → queen-rbee (HTTP) → Everything else
```

**NO SSH**, **NO direct hive communication**, **NO orchestration**

---

## 📋 CORRECT RESPONSIBILITIES

### 1. **queen-rbee Lifecycle Management** ✅ CORRECT

**What EXISTS (132 LOC - queen_lifecycle.rs):**

```rust
pub async fn ensure_queen_rbee_running(
    client: &reqwest::Client, 
    queen_url: &str
) -> Result<()> {
    // 1. Check if queen-rbee is running (GET /health)
    // 2. If not running, spawn queen-rbee daemon
    // 3. Wait for health check (30s timeout)
    // 4. Detach process (keep running in background)
}
```

**When to use:**
- ✅ `rbee infer` - Needs queen for orchestration
- ✅ `rbee setup add-node` - Registers in queen registry
- ✅ `rbee setup list-nodes` - Queries queen registry
- ✅ `rbee models download` - Should go through queen
- ✅ `rbee workers list` - Should query queen's worker registry

**When NOT to use:**
- ❌ `rbee install` - Local file operations only
- ❌ `rbee --version` - No queen needed

**Ephemeral Mode:**
- Starts queen with temp database (`/tmp/queen-rbee-ephemeral.db`)
- Queen shuts down when keeper exits (cascading shutdown)
- Use case: Quick one-off inference commands

**Daemon Mode (future):**
- Queen runs persistently
- Multiple keeper sessions can connect
- Use case: Long-running orchestration

---

### 2. **User Interface (CLI)** ✅ CORRECT

**All commands submit to queen-rbee HTTP API:**

**A. Inference:**
```bash
rbee infer --node gpu-0 --model llama-7b --prompt "Hello"
  ↓
POST http://localhost:8080/v2/tasks
  {
    "node": "gpu-0",
    "model": "llama-7b", 
    "prompt": "Hello",
    ...
  }
```

**B. Node Management:**
```bash
rbee setup add-node gpu-node-1 --ssh-host 192.168.1.100
  ↓
POST http://localhost:8080/v2/registry/beehives/add
  {
    "node_name": "gpu-node-1",
    "ssh_host": "192.168.1.100",
    ...
  }

rbee setup list-nodes
  ↓
GET http://localhost:8080/v2/registry/beehives/list

rbee setup remove-node gpu-node-1
  ↓
POST http://localhost:8080/v2/registry/beehives/remove
```

**C. Model Management (CORRECTED - via queen):**
```bash
rbee models download llama-7b --node gpu-0
  ↓
POST http://localhost:8080/v2/models/download
  {
    "node": "gpu-0",
    "model_ref": "llama-7b"
  }
  ↓ (queen coordinates)
queen-rbee → finds hive for node
           → SSH to hive (queen has SSH, not keeper!)
           → triggers hive model provisioner
           → waits for completion
           → returns to keeper

rbee models list --node gpu-0
  ↓
GET http://localhost:8080/v2/models/list?node=gpu-0
  ↓ (queen queries)
queen-rbee → finds hive for node
           → HTTP to hive /v1/models/list
           → returns to keeper
```

**D. Worker Management (CORRECTED - via queen):**
```bash
rbee workers list
  ↓
GET http://localhost:8080/v2/workers/list
  ↓ (queen has worker registry)
Returns all workers across ALL hives

rbee workers list --node gpu-0
  ↓
GET http://localhost:8080/v2/workers/list?node=gpu-0
  ↓ (queen filters by node)

rbee workers spawn --node gpu-0 --model llama-7b
  ↓
POST http://localhost:8080/v2/workers/spawn
  {
    "node": "gpu-0",
    "model_ref": "llama-7b"
  }
  ↓ (queen orchestrates)
queen-rbee → finds hive for node
           → HTTP to hive /v1/workers/spawn
           → waits for worker ready
           → registers in queen worker registry
           → returns to keeper
```

**E. Installation:**
```bash
rbee install --system
  ↓ (LOCAL file operations - NO queen, NO SSH)
1. Copy binaries to /usr/local/bin/
2. Create systemd services
3. Configure paths
```

---

## 🔴 VIOLATIONS IN CURRENT CODE

### Violation: Direct SSH to Hives ❌

**File:** `bin/rbee-keeper/src/commands/hive.rs` (84 LOC)  
**File:** `bin/rbee-keeper/src/ssh.rs` (14 LOC)

**Current (WRONG):**
```rust
// commands/hive.rs
pub fn handle(action: HiveAction) -> Result<()> {
    match action {
        HiveAction::Models { action, host } => {
            // ❌ Direct SSH to hive, bypassing queen!
            ssh::execute_remote_command_streaming(host, &command)?;
        }
        HiveAction::Worker { action, host } => {
            // ❌ Direct SSH to hive, bypassing queen!
            ssh::execute_remote_command_streaming(host, &command)?;
        }
        // ...
    }
}

// ssh.rs
pub fn execute_remote_command_streaming(host: &str, command: &str) -> Result<()> {
    // ❌ Uses system ssh binary directly!
    Command::new("ssh").arg(host).arg(command).status()?;
}
```

**Commands that violate:**
```bash
# ❌ These ALL bypass queen-rbee via direct SSH:
rbee hive models download llama-7b --host gpu-0
rbee hive models list --host gpu-0
rbee hive worker spawn cuda llama-7b --host gpu-0
rbee hive worker list --host gpu-0
rbee hive git pull --host gpu-0
rbee hive status --host gpu-0
```

**Why this is wrong:**
1. **Duplicated logic** - Same SSH code as queen-rbee will need
2. **Bypass orchestration** - Queen doesn't know about these operations
3. **No centralized tracking** - Queen's registries are out of sync
4. **Security risk** - Multiple SSH entry points
5. **Maintenance burden** - Changes must be made in 2 places

---

## ✅ CORRECTED ARCHITECTURE

### Lifecycle Chain:

```
┌─────────────────────────────────────────────────────┐
│ User types: rbee infer --node gpu-0 --model llama   │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│ rbee-keeper (CLI)                                   │
│  1. ensure_queen_rbee_running()                     │
│     - Check queen health                            │
│     - If not running: spawn queen daemon            │
│     - Wait for ready                                │
│  2. POST http://localhost:8080/v2/tasks             │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP
                   ▼
┌─────────────────────────────────────────────────────┐
│ queen-rbee (Orchestrator Daemon)                    │
│  1. Receive inference request                       │
│  2. Check worker registry (do we have worker?)      │
│  3. If no worker:                                   │
│     - Select hive for model                         │
│     - ensure_hive_running(hive_id)                  │
│       - Check hive health                           │
│       - If not running: start hive (SSH/local)      │
│       - Wait for ready                              │
│     - Request hive to spawn worker                  │
│     - Wait for worker ready                         │
│  4. Route request to worker                         │
│  5. Relay SSE stream back to keeper                 │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP or SSH (queen has SSH!)
                   ▼
┌─────────────────────────────────────────────────────┐
│ rbee-hive (Daemon - NO CLI!)                        │
│  1. Receive worker spawn request (HTTP)             │
│  2. Check if model available locally                │
│  3. If not: download model (provision)              │
│  4. Spawn worker process                            │
│  5. Worker calls back /ready                        │
│  6. Report worker ready to queen                    │
└──────────────────┬──────────────────────────────────┘
                   │ spawns
                   ▼
┌─────────────────────────────────────────────────────┐
│ llm-worker-rbee (Worker)                            │
│  1. Load model into VRAM                            │
│  2. Call back to hive /ready                        │
│  3. Wait for inference requests                     │
│  4. Generate tokens                                 │
└─────────────────────────────────────────────────────┘
```

---

## 📦 CORRECTED CRATE STRUCTURE

Based on CORRECT responsibilities:

| # | Crate | LOC | Purpose | Status |
|---|-------|-----|---------|--------|
| 1 | config | 44 | Config loading | ✅ Keep |
| 2 | ~~ssh-client~~ | ~~14~~ | ~~SSH wrapper~~ | ❌ REMOVE |
| 3 | pool-client | 115 | HTTP client for direct pool mode | ⚠️ Keep (rarely used) |
| 4 | queen-lifecycle | 75 | Auto-start queen daemon | ✅ Keep |
| 5 | commands | 817 → 625 | CLI commands | ✅ Keep (remove hive.rs 84 LOC + logs.rs SSH 24 LOC) |

**Corrected Total:** 859 LOC in libraries + 187 LOC binary

**Removed:**
- ssh-client crate (14 LOC)
- commands/hive.rs (84 LOC) - all commands go via queen
- commands/logs.rs SSH code (24 LOC) - logs via queen API

**Added commands (via queen):**
- `rbee models download` → queen `/v2/models/download`
- `rbee models list` → queen `/v2/models/list`
- `rbee workers spawn` → queen `/v2/workers/spawn`
- `rbee workers list` → queen `/v2/workers/list`
- `rbee workers stop` → queen `/v2/workers/shutdown`
- `rbee logs` → queen `/v2/logs?node=X`

---

## 📋 COMMAND BREAKDOWN (CORRECTED)

### Commands that use queen-lifecycle:

**infer** (186 LOC)
```rust
pub async fn handle(node, model, prompt, ...) -> Result<()> {
    ensure_queen_rbee_running(&client, "http://localhost:8080").await?;
    
    let response = client
        .post("http://localhost:8080/v2/tasks")
        .json(&request)
        .send().await?;
    
    // Stream SSE tokens
}
```

**setup** (222 LOC)
```rust
pub async fn handle_add_node(...) -> Result<()> {
    ensure_queen_rbee_running(&client, "http://localhost:8080").await?;
    
    client
        .post("http://localhost:8080/v2/registry/beehives/add")
        .json(&request)
        .send().await?;
}
```

**models** (NEW - ~150 LOC)
```rust
pub async fn handle_models(action: ModelsAction) -> Result<()> {
    ensure_queen_rbee_running(&client, "http://localhost:8080").await?;
    
    match action {
        ModelsAction::Download { node, model } => {
            client
                .post("http://localhost:8080/v2/models/download")
                .json(&json!({ "node": node, "model_ref": model }))
                .send().await?;
        }
        ModelsAction::List { node } => {
            client
                .get(&format!("http://localhost:8080/v2/models/list?node={}", node))
                .send().await?;
        }
    }
}
```

**workers** (NEW - ~150 LOC - replaces current workers.rs 197 LOC)
```rust
pub async fn handle_workers(action: WorkersAction) -> Result<()> {
    ensure_queen_rbee_running(&client, "http://localhost:8080").await?;
    
    match action {
        WorkersAction::List { node } => {
            let url = if let Some(n) = node {
                format!("http://localhost:8080/v2/workers/list?node={}", n)
            } else {
                "http://localhost:8080/v2/workers/list".to_string()
            };
            client.get(&url).send().await?;
        }
        WorkersAction::Spawn { node, model } => {
            client
                .post("http://localhost:8080/v2/workers/spawn")
                .json(&json!({ "node": node, "model_ref": model }))
                .send().await?;
        }
    }
}
```

**install** (98 LOC - NO queen needed)
```rust
pub fn handle_install(system: bool) -> Result<()> {
    // Local file operations only
    // Copy binaries to /usr/local/bin/ or ~/.local/bin/
    // Create systemd services if system=true
}
```

### Commands to REMOVE:

**❌ hive.rs (84 LOC) - ALL commands bypass queen**
- `rbee hive models` → Should be `rbee models` via queen
- `rbee hive worker` → Should be `rbee workers` via queen
- `rbee hive git` → NOT needed (queen manages installation)
- `rbee hive status` → Should be `rbee status` via queen

**❌ logs.rs SSH code (24 LOC) - bypasses queen**
- `rbee logs --node X` → Should be queen `/v2/logs?node=X`

---

## 🎯 CORRECTED LOC COUNTS

| Component | Current | Remove | Add | Corrected |
|-----------|---------|--------|-----|-----------|
| **Libraries** |
| config | 44 | - | - | 44 |
| ssh-client | 14 | -14 | - | 0 ❌ |
| pool-client | 115 | - | - | 115 |
| queen-lifecycle | 75 | - | - | 75 |
| commands | 817 | -108 | +300 | 1,009 |
| **Subtotal** | 1,065 | -122 | +300 | 1,243 |
| **Binary** |
| main.rs + cli.rs | 187 | - | - | 187 |
| **TOTAL** | 1,252 | -122 | +300 | 1,430 |

**Changes:**
- Remove ssh-client crate: -14 LOC
- Remove hive.rs: -84 LOC
- Remove logs.rs SSH: -24 LOC
- Add models.rs (new): +150 LOC
- Expand workers.rs: +150 LOC

**Net:** +178 LOC (more functionality, all via queen)

---

## 🔑 KEY PRINCIPLES

### 1. Thin Client Pattern
```
rbee-keeper is JUST a CLI wrapper around queen-rbee HTTP API
```

### 2. No SSH
```
Only queen-rbee has SSH for managing remote hives
rbee-keeper NEVER uses SSH
```

### 3. No Orchestration
```
rbee-keeper submits requests
queen-rbee makes ALL decisions
```

### 4. Lifecycle Management
```
rbee-keeper → queen lifecycle (start/stop queen daemon)
queen-rbee → hive lifecycle (start/stop hives via SSH)
rbee-hive → worker lifecycle (spawn/stop workers locally)
```

### 5. Cascading Shutdown (Ephemeral Mode)
```
User Ctrl+C rbee keeper
  ↓
keeper exits
  ↓
queen detects keeper disconnect
  ↓
queen shuts down all hives (SSH)
  ↓
hives shut down all workers
  ↓
Complete cleanup
```

---

## 📊 COMPARISON: CURRENT vs CORRECTED

| Feature | Current | Corrected |
|---------|---------|-----------|
| **SSH** | ✅ Has (ssh.rs 14 LOC) | ❌ Removed |
| **Direct hive commands** | ✅ Has (hive.rs 84 LOC) | ❌ Removed |
| **queen lifecycle** | ✅ Correct | ✅ Keep |
| **infer via queen** | ✅ Correct | ✅ Keep |
| **setup via queen** | ✅ Correct | ✅ Keep |
| **models via queen** | ❌ Missing | ✅ Add (150 LOC) |
| **workers via queen** | ⚠️ Partial (missing spawn) | ✅ Expand (150 LOC) |
| **Total LOC** | 1,252 | 1,430 |

**Completion:** ~87% correct (some violations exist)

---

## 🚀 MIGRATION PLAN

### Phase 1: Remove Violations
1. Delete `src/ssh.rs` (14 LOC)
2. Delete `src/commands/hive.rs` (84 LOC)
3. Remove SSH from `src/commands/logs.rs` (24 LOC)
4. Update `src/cli.rs` to remove hive subcommands

### Phase 2: Add Missing Commands
1. Create `src/commands/models.rs` (150 LOC)
   - `rbee models download --node X MODEL`
   - `rbee models list --node X`
2. Expand `src/commands/workers.rs` (add 150 LOC)
   - Add `rbee workers spawn --node X --model Y`
3. Update `src/commands/logs.rs` to use queen API (not SSH)

### Phase 3: Update Queen API
1. Add queen endpoints:
   - `POST /v2/models/download`
   - `GET /v2/models/list`
   - `POST /v2/workers/spawn`
   - `GET /v2/logs`
2. Queen coordinates with hives via SSH/HTTP

---

**Status:** 🟡 PARTIALLY CORRECT - Has violations (SSH, direct hive access)  
**Violations:** ssh.rs (14 LOC), hive.rs (84 LOC), logs.rs SSH (24 LOC)  
**Missing:** models commands (150 LOC), workers spawn (in workers.rs)  
**Next:** Remove violations, add missing functionality via queen API
