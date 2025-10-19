# TEAM-130D: rbee-keeper - PART 1: METRICS & CRATES (CORRECTED)

**Binary:** `bin/rbee-keeper` (CLI tool `rbee`)  
**Phase:** Phase 2, Day 7-8 (REWRITE)  
**Date:** 2025-10-19  
**Team:** 130D (Architectural Corrections Applied)

---

## 🎯 EXECUTIVE SUMMARY

**Current:** CLI remote control tool (1,252 LOC code-only, 13 files)  
**Violations:** SSH (122 LOC), direct hive access  
**Corrected:** Thin client with 4 crates (1,430 LOC)  
**Risk:** LOW (remove violations, add queen API calls)  
**Timeline:** 3 days (24 hours)

**TEAM-130D Corrections Applied:**
- ✅ Removed SSH crate (14 LOC violation)
- ✅ Removed hive.rs commands (84 LOC violation)
- ✅ Removed logs.rs SSH (24 LOC violation)
- ✅ Added models commands via queen (150 LOC)
- ✅ Expanded workers commands via queen (150 LOC)
- ✅ ALL commands go through queen-rbee HTTP API

**Reference:** `TEAM_130C_RBEE_KEEPER_COMPLETE_RESPONSIBILITIES.md`

---

## 📊 GROUND TRUTH METRICS

```bash
$ cloc bin/rbee-keeper/src --quiet
Files: 13 | Code: 1,252 | Comments: 300 | Blanks: 214
Total Lines: 1,766
```

**TEAM-130D Analysis:**
- **Violations:** 122 LOC (ssh.rs 14 + hive.rs 84 + logs.rs SSH 24)
- **Keep:** 1,130 LOC (correct architecture)
- **Add:** 300 LOC (models + workers expansion)
- **Corrected Total:** 1,430 LOC

**Files with Violations:**
1. ssh.rs - 14 LOC ❌ REMOVE (bypasses queen)
2. commands/hive.rs - 84 LOC ❌ REMOVE (direct SSH to hives)
3. commands/logs.rs - 24 LOC ❌ REMOVE SSH code (use queen API)

**Files to Keep:**
1. queen_lifecycle.rs - 132 LOC ✅ CORRECT (keeper→queen lifecycle)
2. commands/infer.rs - 186 LOC ✅ CORRECT (via queen)
3. commands/setup.rs - 222 LOC ✅ CORRECT (via queen)
4. commands/install.rs - 98 LOC ✅ CORRECT (local only)
5. commands/workers.rs - 197 LOC ⚠️ EXPAND (add spawn)

---

## 🏗️ 4 CORRECTED CRATES (No Violations)

| # | Crate | LOC | Purpose | Status |
|---|-------|-----|---------|--------|
| 1 | config | 44 | Configuration loading | ✅ Keep |
| 2 | ~~ssh-client~~ | ~~14~~ | ~~SSH wrapper~~ | ❌ REMOVE |
| 3 | queen-lifecycle | 75 | Auto-start queen daemon | ✅ Keep |
| 4 | commands | 1,009 | All CLI commands (via queen) | ✅ Expand |

**Total:** 1,128 LOC in libraries + 187 LOC binary + 115 LOC pool-client (rarely used)

**Removed Violations:**
- ssh-client crate: -14 LOC
- hive.rs: -84 LOC (direct SSH)
- logs.rs SSH: -24 LOC

**Added Functionality:**
- models.rs: +150 LOC (via queen `/v2/models/*`)
- workers.rs expansion: +150 LOC (add spawn via queen)

---

## 📦 CRATE SPECIFICATIONS

### CRATE 1: rbee-keeper-config (44 LOC) ✅ KEEP

**Purpose:** Configuration file management  
**Files:** config.rs (44)

**API:**
```rust
pub struct Config {
    pub pool: PoolConfig,
    pub paths: PathsConfig,
    pub remote: Option<RemoteConfig>, // ⚠️ Will be removed (no SSH!)
}

pub struct PoolConfig {
    pub default_url: String,
    pub api_key: String,
}

impl Config {
    pub fn load() -> Result<Self>;
}
```

**Config Paths:**
1. `$RBEE_CONFIG` environment variable
2. `~/.config/rbee/config.toml`
3. `/etc/rbee/config.toml`

**Dependencies:** serde, toml, dirs, anyhow

**TEAM-130D Note:** `RemoteConfig` will be deprecated (no SSH in keeper)

---

### CRATE 2: ~~rbee-keeper-ssh-client~~ ❌ REMOVE (VIOLATION)

**Status:** DELETED - Violates architecture

**Why Remove:**
- rbee-keeper NEVER uses SSH
- Only queen-rbee has SSH (for hive management)
- Duplicates logic that queen needs
- Bypasses orchestration layer

**Migration:**
```bash
# OLD (WRONG):
rbee hive models list --host gpu-0
  ↓ direct SSH to hive (bypasses queen!)

# NEW (CORRECT):
rbee models list --node gpu-0
  ↓ HTTP to queen
  ↓ queen queries hive via HTTP/SSH
```

---

### CRATE 3: rbee-keeper-queen-lifecycle (75 LOC) ✅ KEEP

**Purpose:** Auto-start queen-rbee daemon if not running  
**Files:** queen_lifecycle.rs (75) - moved from src/

**API:**
```rust
pub async fn ensure_queen_rbee_running(
    client: &reqwest::Client,
    queen_url: &str,
) -> Result<()> {
    // 1. Check if queen is running (GET /health)
    if health_check_ok(queen_url) {
        return Ok(());
    }
    
    // 2. Find queen-rbee binary
    let queen_binary = find_binary("queen-rbee")?;
    
    // 3. Spawn queen daemon
    let child = Command::new(&queen_binary)
        .arg("--port").arg("8080")
        .arg("--database").arg("/tmp/queen-rbee-ephemeral.db")
        .spawn()?;
    
    // 4. Wait for ready (30s)
    wait_for_health(queen_url, Duration::from_secs(30)).await?;
    
    // 5. Detach (keep running)
    std::mem::forget(child);
    
    Ok(())
}
```

**When to Use:**
- ✅ `rbee infer` - Needs queen for orchestration
- ✅ `rbee setup` - Registers nodes in queen
- ✅ `rbee models` - Downloads via queen coordination
- ✅ `rbee workers` - Spawns via queen orchestration
- ❌ `rbee install` - Local files, no queen needed

**Modes:**

**Ephemeral (Current):**
- Temp DB (`/tmp/queen-rbee-ephemeral.db`)
- Auto-starts for single command
- Cascading shutdown on exit

**Daemon (Future):**
- Persistent DB (`~/.rbee/queen.db`)
- Long-running instance
- Multiple keeper sessions

**Dependencies:** tokio, reqwest, colored, anyhow

---

### CRATE 4: rbee-keeper-commands (1,009 LOC) ✅ EXPANDED

**Purpose:** All CLI command implementations (via queen HTTP API)  
**Files:** commands/*.rs

**LOC Breakdown:**
```
KEEP (correct):
  infer.rs        186 LOC    Inference via queen
  setup.rs        222 LOC    Node registry via queen
  install.rs       98 LOC    Local installation
  
REMOVE (violations):
  hive.rs          84 LOC    ❌ Direct SSH to hives
  logs.rs SSH      24 LOC    ❌ Partial (remove SSH part)

EXPAND:
  workers.rs      197→347    +150 LOC (add spawn)
  
ADD NEW:
  models.rs         0→150    +150 LOC (via queen)
  logs.rs (new)     0→50     +50 LOC (via queen API, not SSH)
  
TOTAL:           703→1,009   +306 LOC net
```

---

#### A. infer.rs (186 LOC) ✅ CORRECT

**Command:** `rbee infer --node gpu-0 --model llama-7b --prompt "Hello"`

```rust
pub async fn handle(
    node: String,
    model: String,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
) -> Result<()> {
    let client = reqwest::Client::new();
    let queen_url = "http://localhost:8080";
    
    // Auto-start queen if needed
    ensure_queen_rbee_running(&client, queen_url).await?;
    
    // Submit to queen
    let response = client
        .post(format!("{}/v2/tasks", queen_url))
        .json(&json!({
            "node": node,
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }))
        .send().await?;
    
    // Stream SSE tokens
    stream_sse_output(response).await?;
    
    Ok(())
}
```

**Flow:**
1. keeper→queen (HTTP)
2. queen→hive (HTTP/SSH)
3. hive→worker (spawn)
4. worker→tokens (SSE)
5. queen→keeper (SSE relay)

---

#### B. setup.rs (222 LOC) ✅ CORRECT

**Commands:**
- `rbee setup add-node gpu-node-1 --ssh-host 192.168.1.100`
- `rbee setup list-nodes`
- `rbee setup remove-node gpu-node-1`

```rust
async fn handle_add_node(...) -> Result<()> {
    ensure_queen_rbee_running(&client, queen_url).await?;
    
    // Register in queen's hive registry
    client
        .post(format!("{}/v2/registry/beehives/add", queen_url))
        .json(&AddNodeRequest { ... })
        .send().await?;
    
    Ok(())
}
```

**Queen handles:**
- SSH connectivity test
- SQLite hive registry storage
- Hive lifecycle management

---

#### C. models.rs (NEW - 150 LOC) ✅ ADD

**Commands:**
- `rbee models download llama-7b --node gpu-0`
- `rbee models list --node gpu-0`
- `rbee models catalog`

```rust
pub async fn handle(action: ModelsAction) -> Result<()> {
    let client = reqwest::Client::new();
    let queen_url = "http://localhost:8080";
    
    ensure_queen_rbee_running(&client, queen_url).await?;
    
    match action {
        ModelsAction::Download { node, model } => {
            // Queen coordinates download
            client
                .post(format!("{}/v2/models/download", queen_url))
                .json(&json!({ "node": node, "model_ref": model }))
                .send().await?;
            
            println!("Model download initiated via queen-rbee");
        }
        ModelsAction::List { node } => {
            // Queen queries hive
            let url = format!("{}/v2/models/list?node={}", queen_url, node);
            let response = client.get(&url).send().await?;
            let models: Vec<ModelInfo> = response.json().await?;
            
            for model in models {
                println!("{}: {} ({})", model.id, model.name, model.size);
            }
        }
        ModelsAction::Catalog => {
            // Queen's unified catalog across all hives
            let response = client.get(format!("{}/v2/models/catalog", queen_url)).await?;
            // Display catalog
        }
    }
    
    Ok(())
}
```

**Why via queen:**
- Centralized model catalog (across all hives)
- Coordinated downloads (avoid duplicates)
- Consistent API

---

#### D. workers.rs (197→347 LOC) ✅ EXPAND

**Current (197 LOC):**
- `rbee workers list` ✅ CORRECT (via queen)
- `rbee workers health` ✅ CORRECT (via queen)
- `rbee workers shutdown` ✅ CORRECT (via queen)

**Missing (add 150 LOC):**
- `rbee workers spawn --node gpu-0 --model llama-7b` ❌ MISSING

```rust
pub async fn handle(action: WorkersAction) -> Result<()> {
    let client = reqwest::Client::new();
    let queen_url = "http://localhost:8080";
    
    ensure_queen_rbee_running(&client, queen_url).await?;
    
    match action {
        WorkersAction::List { node } => {
            // Queen's worker registry (across all hives)
            let url = if let Some(n) = node {
                format!("{}/v2/workers/list?node={}", queen_url, n)
            } else {
                format!("{}/v2/workers/list", queen_url)
            };
            
            let response = client.get(&url).send().await?;
            let workers: Vec<WorkerInfo> = response.json().await?;
            
            for worker in workers {
                println!("{}: {} on {} ({})", 
                    worker.id, worker.model_ref, worker.node, worker.state);
            }
        }
        WorkersAction::Spawn { node, model } => {
            // NEW: Spawn via queen orchestration
            client
                .post(format!("{}/v2/workers/spawn", queen_url))
                .json(&json!({
                    "node": node,
                    "model_ref": model,
                }))
                .send().await?;
            
            println!("Worker spawn initiated via queen-rbee");
        }
        WorkersAction::Shutdown { worker_id } => {
            client
                .post(format!("{}/v2/workers/shutdown", queen_url))
                .json(&json!({ "worker_id": worker_id }))
                .send().await?;
        }
    }
    
    Ok(())
}
```

---

#### E. install.rs (98 LOC) ✅ CORRECT (No queen needed)

**Command:** `rbee install --system`

```rust
pub fn handle(system: bool) -> Result<()> {
    // Local file operations only - NO queen needed
    
    if system {
        // Install to /usr/local/bin/
        copy_binaries("/usr/local/bin/")?;
        create_systemd_services()?;
    } else {
        // Install to ~/.local/bin/
        copy_binaries(&format!("{}/.local/bin", home_dir()))?;
    }
    
    Ok(())
}
```

---

#### F. logs.rs (NEW - 50 LOC) ✅ ADD (via queen, not SSH)

**Command:** `rbee logs --node gpu-0 --follow`

**OLD (WRONG - 24 LOC SSH):**
```rust
// ❌ Direct SSH to hive (violation!)
ssh::execute_remote_command_streaming(host, "journalctl -u rbee-hive -f")?;
```

**NEW (CORRECT - 50 LOC via queen):**
```rust
pub async fn handle(node: String, follow: bool) -> Result<()> {
    let client = reqwest::Client::new();
    let queen_url = "http://localhost:8080";
    
    ensure_queen_rbee_running(&client, queen_url).await?;
    
    // Queen queries hive for logs
    let url = format!("{}/v2/logs?node={}&follow={}", queen_url, node, follow);
    let response = client.get(&url).send().await?;
    
    // Stream logs back to terminal
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        print!("{}", String::from_utf8_lossy(&chunk?));
    }
    
    Ok(())
}
```

---

#### G. ~~hive.rs~~ (84 LOC) ❌ REMOVE (VIOLATION)

**Status:** DELETED - All commands violate architecture

**Violated Commands:**
```bash
# ❌ These ALL bypass queen via direct SSH:
rbee hive models download llama-7b --host gpu-0
rbee hive models list --host gpu-0
rbee hive worker spawn --host gpu-0
rbee hive git pull --host gpu-0
rbee hive status --host gpu-0
```

**Replacement:**
```bash
# ✅ Use these instead (via queen):
rbee models download llama-7b --node gpu-0
rbee models list --node gpu-0
rbee workers spawn --node gpu-0 --model llama-7b
# (git operations handled by queen hive lifecycle)
rbee status --node gpu-0
```

---

## 📊 DEPENDENCY GRAPH (CORRECTED)

```
Layer 0 (Standalone):
- config (44 LOC)
- queen-lifecycle (75 LOC)

Layer 1 (Commands - Uses Layer 0):
- commands (1,009 LOC) → uses config, queen-lifecycle

Binary (187 LOC):
- main.rs (12 LOC)
- cli.rs (175 LOC) → stays in binary (clap)
  → uses commands crate

(Optional - rarely used):
- pool-client (115 LOC) → for direct pool mode
```

**No SSH crate ✅**  
**No circular dependencies ✅**

---

## 🔗 CORRECTED ARCHITECTURE

### User Interface Pattern:

```
User Command
    ↓
rbee-keeper (CLI)
    ├─ ensure_queen_rbee_running() (auto-start if needed)
    ├─ HTTP POST/GET to queen-rbee
    └─ Stream response back to user
```

**NO SSH**, **NO direct hive access**, **ALL via queen**

### Example Flow:

```
$ rbee models download llama-7b --node gpu-0

rbee-keeper:
  1. ensure_queen_rbee_running()
  2. POST http://localhost:8080/v2/models/download
     {"node": "gpu-0", "model_ref": "llama-7b"}
  3. Wait for response
  4. Display result

queen-rbee (orchestration):
  1. Receive download request
  2. Find hive for node "gpu-0"
  3. ensure_hive_running("gpu-0") (SSH if network mode)
  4. POST http://hive-gpu-0:8080/v1/models/provision
  5. Wait for download complete
  6. Return to keeper

rbee-hive (execution):
  1. Receive provision request
  2. Download model from HuggingFace
  3. Store in local cache
  4. Return success
```

---

## 📋 COMPARISON: TEAM-130C vs TEAM-130D

| Component | 130C (Violations) | 130D (Corrected) | Change |
|-----------|-------------------|------------------|--------|
| **Crates** | 5 (inc. ssh-client) | 4 (no SSH) | -1 crate |
| **ssh-client** | 14 LOC | REMOVED | -14 LOC |
| **commands/hive.rs** | 84 LOC | REMOVED | -84 LOC |
| **commands/logs.rs** | 24 LOC SSH | 50 LOC via queen | +26 LOC |
| **commands/models.rs** | MISSING | 150 LOC | +150 LOC |
| **commands/workers.rs** | 197 LOC | 347 LOC | +150 LOC |
| **Total LOC** | 1,252 | 1,430 | +178 LOC |

**Net Result:** +178 LOC, -122 LOC violations, +300 LOC new functionality

---

## ✅ TEAM-130D CORRECTIONS APPLIED

**Violations Removed:**
1. ✅ Deleted ssh.rs (14 LOC)
2. ✅ Deleted hive.rs (84 LOC)
3. ✅ Removed SSH from logs.rs (24 LOC)

**Functionality Added:**
1. ✅ Created models.rs (150 LOC via queen)
2. ✅ Expanded workers.rs (+150 LOC, added spawn)
3. ✅ Rewrote logs.rs (50 LOC via queen API)

**Architecture Verified:**
- ✅ NO SSH at keeper level
- ✅ ALL commands via queen HTTP API
- ✅ queen-lifecycle correct (keeper→queen)
- ✅ Thin client pattern maintained

---

**Status:** TEAM-130D Complete - Architecture Violations Fixed  
**Next:** rbee-hive PART1 (remove CLI violation)  
**Reference:** `TEAM_130C_RBEE_KEEPER_COMPLETE_RESPONSIBILITIES.md`
