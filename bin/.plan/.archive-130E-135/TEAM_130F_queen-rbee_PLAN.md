# TEAM-130F: queen-rbee BINARY + CRATES PLAN

**Phase:** Phase 3 Implementation Planning  
**Date:** 2025-10-19  
**Team:** TEAM-130F  
**Status:** 📋 PLAN (Future Architecture)

---

## 🎯 MISSION

Define **PLANNED** architecture for queen-rbee after Phase 3 consolidation.

**Key Changes:**
- ✅ Add hive lifecycle management (NEW - was missing!)
- ✅ Use shared crates (daemon-lifecycle, rbee-types, rbee-http-client, rbee-ssh-client)
- ✅ Remove local SSH implementation
- ✅ Type-safe communication
- ✅ Remove unused dependencies (secrets-management)

---

## 📊 METRICS (PLANNED)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Total LOC** | 2,015 | ~2,100 | **+85 LOC** |
| **Files** | 17 | 19 | +2 files |
| **Shared crate deps** | 5 | 7 | +2 deps |

**LOC Breakdown:**
- Add hive-lifecycle: +300 LOC (NEW - critical missing functionality)
- Remove ssh.rs: -76 LOC (use rbee-ssh-client)
- Add shared crate usage: +50 LOC
- Refactor HTTP calls: -189 LOC (use rbee-http-client)
- **Net change: +85 LOC**

**Why LOC increases:** Adding critical missing hive lifecycle functionality

---

## 📦 INTERNAL CRATES (Within Binary)

### 1. beehive-registry (~200 LOC)
**Location:** `src/beehive_registry.rs`  
**Purpose:** SQLite persistent storage of beehive nodes  
**Why NOT shared:** Queen-specific (orchestration context)

**Changes:**
- Use `rbee-types::BeehiveNode` (remove local definition)

**Dependencies:** `rbee-types`, `rusqlite`

---

### 2. worker-registry (~210 LOC)
**Location:** `src/worker_registry.rs`  
**Purpose:** In-memory worker registry for routing/load balancing  
**Why NOT shared:** Queen-specific (routing context)

```rust
pub struct WorkerInfo {
    // Core identity
    pub id: String,
    pub url: String,
    pub model_ref: String,
    pub backend: String,
    pub device: u32,
    
    // State (uses shared enum)
    pub state: rbee_types::WorkerState,
    
    // Routing-specific (NOT in rbee-hive's WorkerInfo)
    pub node_name: String,
    pub slots_total: u32,
    pub slots_available: u32,
    pub vram_bytes: Option<u64>,
}
```

**CRITICAL:** Different from rbee-hive's WorkerInfo (see TEAM-130E corrections)

**Changes:**
- Use `rbee-types::WorkerState` enum only
- Keep WorkerInfo local (routing-specific)

**Dependencies:** `rbee-types` (WorkerState only)

---

### 3. hive-lifecycle (~300 LOC) **NEW!**
**Location:** `src/hive_lifecycle.rs` (NEW FILE)  
**Purpose:** Start/stop local and remote hives  
**Why NOT shared:** Queen-specific (orchestration logic)

```rust
pub struct HiveLifecycleManager {
    lifecycle: DaemonLifecycle,  // Uses shared crate
    ssh_client: RbeeSshClient,   // Uses shared crate
    http_client: RbeeHttpClient, // Uses shared crate
    registry: Arc<BeehiveRegistry>,
}

impl HiveLifecycleManager {
    pub async fn start_local_hive(&self, port: u16) -> Result<String> {
        // Use daemon-lifecycle for local spawn
    }
    
    pub async fn start_network_hive(&self, node: &BeehiveNode) -> Result<String> {
        // Use rbee-ssh-client for remote spawn
    }
    
    pub async fn stop_hive(&self, hive_id: &str) -> Result<()> {
        // Graceful shutdown
    }
}
```

**Dependencies:** `daemon-lifecycle`, `rbee-ssh-client`, `rbee-http-client`, `rbee-types`

---

### 4. http-server (~900 LOC)
**Location:** `src/http/`  
**Purpose:** Axum HTTP server with orchestration endpoints  
**Why NOT shared:** Queen-specific API

**Changes:**
- Use `rbee-http-client` for hive communication
- Use `rbee-types` for request/response types

**Dependencies:** `axum`, `rbee-http-client`, `rbee-types`, `auth-min`

---

### 5. preflight (~150 LOC)
**Location:** `src/preflight/`  
**Purpose:** Preflight checks before operations  
**Why NOT shared:** Queen-specific checks

**Changes:**
- Use `rbee-ssh-client` for SSH checks
- Use `rbee-http-client` for HTTP checks

**Dependencies:** `rbee-ssh-client`, `rbee-http-client`

---

## 🔗 DEPENDENCIES (PLANNED)

```toml
[package]
name = "queen-rbee"
version = "0.1.0"
edition = "2021"
license = "GPL-3.0-or-later"

[lib]
name = "queen_rbee"
path = "src/lib.rs"

[[bin]]
name = "queen-rbee"
path = "src/main.rs"

[dependencies]
# Phase 3 NEW: Shared crates
daemon-lifecycle = { path = "../shared-crates/daemon-lifecycle" }
rbee-http-client = { path = "../shared-crates/rbee-http-client" }
rbee-ssh-client = { path = "../shared-crates/rbee-ssh-client" }
rbee-types = { path = "../shared-crates/rbee-types" }

# Existing: Shared crates
auth-min = { path = "../shared-crates/auth-min" }
input-validation = { path = "../shared-crates/input-validation" }
audit-logging = { path = "../shared-crates/audit-logging" }
deadline-propagation = { path = "../shared-crates/deadline-propagation" }

# REMOVED: secrets-management (unused)

# HTTP server
axum = "0.8"
tower = "0.5"
tower-http = { version = "0.6", features = ["trace", "cors"] }

# Async runtime
tokio = { version = "1", features = ["rt-multi-thread", "macros", "sync", "time", "signal", "fs", "process"] }
futures = "0.3"

# Database (for beehive registry)
rusqlite = { version = "0.32", features = ["bundled"] }
dirs = "5.0"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# CLI
clap = { version = "4.5", features = ["derive"] }

# Error handling
anyhow = "1.0"
thiserror = "1.0"

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }

# UUID generation
uuid = { version = "1.0", features = ["v4", "serde"] }

# Time handling
chrono = { version = "0.4", features = ["serde"] }

[dev-dependencies]
tempfile = "3.0"
wiremock = "0.6"
```

---

## 📋 BINARY STRUCTURE (PLANNED)

```
bin/queen-rbee/
├─ src/
│  ├─ main.rs                    (~50 LOC) - Entry point
│  ├─ lib.rs                     (~30 LOC) - Library exports
│  ├─ beehive_registry.rs        (~200 LOC) - Beehive registry (SQLite)
│  ├─ worker_registry.rs         (~210 LOC) - Worker registry (routing context)
│  ├─ hive_lifecycle.rs          (~300 LOC) - NEW: Hive lifecycle manager
│  ├─ http/
│  │  ├─ mod.rs                  (~20 LOC)
│  │  ├─ routes.rs               (~100 LOC) - Route definitions
│  │  ├─ health.rs               (~40 LOC) - Health endpoint
│  │  ├─ beehives.rs             (~200 LOC) - Beehive endpoints
│  │  ├─ workers.rs              (~150 LOC) - Worker endpoints
│  │  ├─ inference.rs            (~200 LOC) - Inference endpoints
│  │  ├─ types.rs                (~100 LOC) - HTTP types (reduced)
│  │  └─ middleware/
│  │     ├─ mod.rs               (~5 LOC)
│  │     └─ auth.rs              (~50 LOC) - Auth middleware
│  ├─ preflight/
│  │  ├─ mod.rs                  (~10 LOC)
│  │  ├─ ssh.rs                  (~70 LOC) - SSH preflight
│  │  └─ rbee_hive.rs            (~70 LOC) - Hive preflight
├─ Cargo.toml
└─ README.md
```

**Removed Files:**
- ❌ `ssh.rs` (76 LOC) - Use rbee-ssh-client

**New Files:**
- ✅ `hive_lifecycle.rs` (300 LOC) - NEW

**Total Files:** 19 (was 17)

---

## 🎯 NEW API ENDPOINTS

### Hive Lifecycle (NEW!)
- `POST /v2/hives/start` - Start local or remote hive
- `POST /v2/hives/stop` - Stop hive
- `GET /v2/hives/status` - Check hive health

### Logs (NEW!)
- `GET /v2/logs?node=X&follow=true` - Stream logs via SSH

---

## 🔧 IMPLEMENTATION PLAN

### Day 1: Integrate Types & Remove Unused

**Update Cargo.toml:**
```toml
[dependencies]
# NEW: Phase 3 shared crates
rbee-types = { path = "../shared-crates/rbee-types" }

# REMOVE
# secrets-management = { path = "../shared-crates/secrets-management" }
```

**Update beehive_registry.rs:**
```rust
// BEFORE
pub struct BeehiveNode {
    pub node_name: String,
    pub ssh_host: String,
    // ... 12 fields
}

// AFTER
use rbee_types::BeehiveNode;
// Use shared type, remove local definition
```

**Update worker_registry.rs:**
```rust
// BEFORE
pub enum WorkerState {
    Loading,
    Idle,
    Busy,
}

// AFTER
use rbee_types::WorkerState;
// Use shared enum, keep WorkerInfo local
```

---

### Day 2-3: Create Hive Lifecycle

**Create hive_lifecycle.rs:**
```rust
use daemon_lifecycle::DaemonLifecycle;
use rbee_ssh_client::SshClient;
use rbee_http_client::RbeeHttpClient;
use rbee_types::BeehiveNode;

pub struct HiveLifecycleManager {
    registry: Arc<BeehiveRegistry>,
    http_client: RbeeHttpClient,
}

impl HiveLifecycleManager {
    pub async fn start_local_hive(&self, port: u16) -> Result<String> {
        let hive_id = format!("local-{}", port);
        let health_url = format!("http://localhost:{}/v1/health", port);
        
        // Use daemon-lifecycle
        let lifecycle = DaemonLifecycle::with_health_check(
            "rbee-hive",
            &health_url,
            vec!["--port".to_string(), port.to_string()],
        );
        
        lifecycle.ensure_running().await?;
        
        // Update registry
        self.registry.update_status(&hive_id, "online", Some(now())).await?;
        
        Ok(hive_id)
    }
    
    pub async fn start_network_hive(&self, node: &BeehiveNode) -> Result<String> {
        // Use rbee-ssh-client
        let ssh = SshClient::new(&node.ssh_host, &node.ssh_user)
            .with_port(node.ssh_port)
            .with_key(node.ssh_key_path.as_ref().unwrap());
        
        // Check if already running
        let check_result = ssh.exec("pgrep rbee-hive").await?;
        if check_result.success {
            return Ok(node.node_name.clone());
        }
        
        // Start via SSH
        ssh.exec_detached(&format!(
            "nohup rbee-hive --port {} > /dev/null 2>&1 &",
            9200
        )).await?;
        
        // Wait for health check
        let health_url = format!("http://{}:9200/v1/health", node.ssh_host);
        for _ in 0..30 {
            if self.http_client.health_check(&health_url).await {
                break;
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        
        // Update registry
        self.registry.update_status(&node.node_name, "online", Some(now())).await?;
        
        Ok(node.node_name.clone())
    }
    
    pub async fn stop_hive(&self, hive_id: &str) -> Result<()> {
        let node = self.registry.get(hive_id).await?;
        
        // Graceful shutdown via HTTP
        let shutdown_url = format!("http://{}:{}/v1/admin/shutdown", 
            node.ssh_host, 9200);
        self.http_client.post(&shutdown_url).await.ok();
        
        // Update registry
        self.registry.update_status(hive_id, "offline", Some(now())).await?;
        
        Ok(())
    }
}
```

**Add HTTP endpoints:**
```rust
// http/beehives.rs
pub async fn start_hive_handler(
    State(state): State<AppState>,
    Json(request): Json<StartHiveRequest>,
) -> Result<Json<StartHiveResponse>, AppError> {
    let hive_id = if request.mode == "local" {
        state.hive_lifecycle.start_local_hive(request.port).await?
    } else {
        let node = state.beehive_registry.get(&request.node_name).await?;
        state.hive_lifecycle.start_network_hive(&node).await?
    };
    
    Ok(Json(StartHiveResponse { hive_id }))
}
```

---

### Day 4: Integrate HTTP Client

**Update Cargo.toml:**
```toml
[dependencies]
rbee-http-client = { path = "../shared-crates/rbee-http-client" }
```

**Replace manual HTTP clients:**
```rust
// BEFORE (worker_registry.rs)
let client = reqwest::Client::new();
let response = client
    .post(format!("{}/v1/admin/shutdown", worker.url))
    .send().await?;

// AFTER
use rbee_http_client::RbeeHttpClient;

let client = RbeeHttpClient::new();
client.post(&format!("{}/v1/admin/shutdown", worker.url)).await?;
```

**Update all HTTP call sites** (9 occurrences):
- `worker_registry.rs`: Worker shutdown
- `http/inference.rs`: Forward to hive
- `preflight/rbee_hive.rs`: Health checks

---

### Day 5: Integrate SSH Client

**Update Cargo.toml:**
```toml
[dependencies]
daemon-lifecycle = { path = "../shared-crates/daemon-lifecycle" }
rbee-ssh-client = { path = "../shared-crates/rbee-ssh-client" }
```

**Delete ssh.rs:**
```bash
rm bin/queen-rbee/src/ssh.rs
```

**Update preflight/ssh.rs:**
```rust
// BEFORE
use crate::ssh;
ssh::test_ssh_connection(host, port, user, key_path).await?;

// AFTER
use rbee_ssh_client::SshClient;

let ssh = SshClient::new(host, user)
    .with_port(port)
    .with_key(key_path);
ssh.test_connection().await?;
```

**Add /v2/logs endpoint:**
```rust
// http/beehives.rs
pub async fn logs_handler(
    State(state): State<AppState>,
    Query(params): Query<LogsQuery>,
) -> Result<impl IntoResponse, AppError> {
    let node = state.beehive_registry.get(&params.node).await?;
    
    let ssh = SshClient::new(&node.ssh_host, &node.ssh_user)
        .with_port(node.ssh_port)
        .with_key(node.ssh_key_path.as_ref().unwrap());
    
    let result = ssh.exec("journalctl -u rbee-hive -n 100").await?;
    
    Ok(Json(LogsResponse {
        lines: result.stdout.lines().map(|s| s.to_string()).collect(),
    }))
}
```

---

### Day 6: Testing

**Unit tests:**
```bash
cargo test --bin queen-rbee
```

**Integration tests:**
```bash
# Test hive lifecycle
curl -X POST http://localhost:8080/v2/hives/start \
  -H "Content-Type: application/json" \
  -d '{"mode":"local","port":9200}'

# Test logs
curl http://localhost:8080/v2/logs?node=test-node

# Test worker routing
curl -X POST http://localhost:8080/v2/tasks \
  -H "Content-Type: application/json" \
  -d '{"prompt":"test"}'
```

**Verify no local SSH:**
```bash
grep -r "ssh::" bin/queen-rbee/src/
# Should return no results (except in tests)
```

---

## ✅ ACCEPTANCE CRITERIA

1. ✅ Hive lifecycle implemented (start/stop local + remote)
2. ✅ Uses `rbee-types` for BeehiveNode and WorkerState
3. ✅ Uses `rbee-ssh-client` (no local SSH code)
4. ✅ Uses `rbee-http-client` for hive communication
5. ✅ Uses `daemon-lifecycle` for local hive startup
6. ✅ All tests pass
7. ✅ `/v2/hives/start` works for local and network modes
8. ✅ `/v2/logs` endpoint works via SSH

---

## 📝 CRITICAL NOTES

### WorkerInfo is NOT Shared!
- queen-rbee WorkerInfo: Routing context (needs node_name, slots_available)
- rbee-hive WorkerInfo: Lifecycle context (needs pid, restart_count, heartbeat)
- **Only share WorkerState enum** - NOT the full WorkerInfo struct

### Why LOC Increases is Good
- Adding MISSING critical functionality (hive lifecycle)
- Identified as architectural gap by TEAM-130C
- Better to have complete functionality than save LOC

### SSH is Required
- SSH is MANDATORY for network mode (remote hives)
- SSH is NOT needed for local mode only
- rbee-ssh-client consolidation avoids duplication

---

**Status:** 📋 PLAN COMPLETE  
**LOC Impact:** +85 LOC (2,015 → 2,100)  
**Critical Addition:** Hive lifecycle management (300 LOC)
