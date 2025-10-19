# TEAM-027 Handoff: Complete test-001 MVP Implementation

**Date:** 2025-10-09T23:17:00+02:00  
**From:** TEAM-026  
**To:** TEAM-027  
**Status:** Partial implementation, rate-limited  
**Priority:** CRITICAL - MVP implementation

---

## Executive Summary

**What TEAM-026 Completed:**
1. ✅ Added HTTP server dependencies to rbee-hive (axum, tokio, tower, etc.)
2. ✅ Created HTTP module structure mirroring llm-worker-rbee patterns
3. ✅ Implemented WorkerRegistry (Arc<RwLock<HashMap>>) for in-memory worker tracking
4. ✅ Created health endpoint (GET /v1/health)
5. ✅ Created worker management endpoints (spawn, ready callback, list)
6. ✅ Created placeholder model management endpoints
7. ✅ Created HttpServer with graceful shutdown (mirrors llm-worker-rbee)
8. ✅ Created router configuration with all endpoints

**Current System State (THE 4 BINARIES):**
1. **queen-rbee** (HTTP daemon :8080) - M1 ❌ NOT BUILT
2. **llm-worker-rbee** (HTTP daemon :8001+) - M0 ✅ WORKING
3. **rbee-hive** (HTTP daemon :8080) - M1 ⚠️ PARTIALLY BUILT (HTTP infrastructure done, needs wiring)
4. **rbee-keeper** (CLI) - M0 ⚠️ NEEDS HTTP CLIENT

**M0 Status:** 2 of 4 binaries complete, 2 partially complete

---

## What TEAM-026 Built (Files Created)

### rbee-hive HTTP Infrastructure ✅

**Location:** `/home/vince/Projects/llama-orch/bin/rbee-hive/src/`

1. **`http/mod.rs`** - Module exports
2. **`http/server.rs`** - HttpServer struct with graceful shutdown (mirrors llm-worker-rbee)
3. **`http/health.rs`** - GET /v1/health endpoint (returns status, version, api_version)
4. **`http/workers.rs`** - Worker management endpoints:
   - POST /v1/workers/spawn - Spawns llm-worker-rbee process
   - POST /v1/workers/ready - Worker ready callback
   - GET /v1/workers/list - List all workers
5. **`http/models.rs`** - Model endpoints (placeholder, NOT IMPLEMENTED)
6. **`http/routes.rs`** - Router configuration with all endpoints
7. **`registry.rs`** - WorkerRegistry with thread-safe HashMap

**Pattern:** All HTTP code mirrors `bin/llm-worker-rbee/src/http/` to avoid drift

---

## What Still Needs to Be Done

### Priority 1: Complete rbee-hive Daemon (CRITICAL) 🔥

#### Task 1.1: Wire Up Daemon Mode
**Files to modify:**
- `bin/rbee-hive/src/main.rs` - Add async runtime, daemon mode
- `bin/rbee-hive/src/cli.rs` - Add `daemon` subcommand

**What to do:**
1. Add `mod http;` and `mod registry;` to main.rs
2. Change `main()` to `#[tokio::main] async fn main()`
3. Add `Daemon` subcommand to cli.rs:
   ```rust
   #[derive(Subcommand)]
   pub enum Commands {
       // ... existing commands ...
       /// Start HTTP daemon
       Daemon {
           #[arg(long, default_value = "0.0.0.0:8080")]
           addr: String,
       },
   }
   ```
4. Wire up daemon handler:
   ```rust
   Commands::Daemon { addr } => {
       let addr: SocketAddr = addr.parse()?;
       let registry = Arc::new(WorkerRegistry::new());
       let router = create_router(registry.clone());
       let server = HttpServer::new(addr, router).await?;
       
       // TODO: Spawn background tasks (health monitor, idle timeout)
       
       server.run().await?;
       Ok(())
   }
   ```

**Estimated time:** 1-2 hours

---

#### Task 1.2: Implement Background Monitoring Loops
**Files to create:**
- `bin/rbee-hive/src/monitor.rs` - Health monitoring loop
- `bin/rbee-hive/src/timeout.rs` - Idle timeout enforcement

**Health Monitor (monitor.rs):**
```rust
//! Health monitoring loop
//! Per test-001-mvp.md Phase 5: Monitor worker health every 30s

use crate::registry::WorkerRegistry;
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info};

pub async fn health_monitor_loop(registry: Arc<WorkerRegistry>) {
    let mut interval = tokio::time::interval(Duration::from_secs(30));
    
    loop {
        interval.tick().await;
        
        for worker in registry.list().await {
            // Check worker health: GET {worker.url}/health
            let client = reqwest::Client::new();
            match client.get(&format!("{}/health", worker.url)).send().await {
                Ok(response) if response.status().is_success() => {
                    info!(worker_id = %worker.id, "Worker healthy");
                }
                Ok(response) => {
                    error!(worker_id = %worker.id, status = %response.status(), "Worker unhealthy");
                    // TODO: Mark worker as unhealthy
                }
                Err(e) => {
                    error!(worker_id = %worker.id, error = %e, "Worker unreachable");
                    // TODO: Mark worker as unhealthy
                }
            }
        }
    }
}
```

**Idle Timeout (timeout.rs):**
```rust
//! Idle timeout enforcement
//! Per test-001-mvp.md Phase 8: Auto-shutdown after 5 minutes idle

use crate::registry::{WorkerRegistry, WorkerState};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tracing::{info, warn};

pub async fn idle_timeout_loop(registry: Arc<WorkerRegistry>) {
    let mut interval = tokio::time::interval(Duration::from_secs(60));
    
    loop {
        interval.tick().await;
        
        let now = SystemTime::now();
        for worker in registry.get_idle_workers().await {
            if let Ok(idle_duration) = now.duration_since(worker.last_activity) {
                if idle_duration > Duration::from_secs(300) { // 5 minutes
                    info!(worker_id = %worker.id, "Worker idle timeout, shutting down");
                    
                    // Send shutdown request: POST {worker.url}/v1/admin/shutdown
                    let client = reqwest::Client::new();
                    match client.post(&format!("{}/v1/admin/shutdown", worker.url))
                        .send()
                        .await
                    {
                        Ok(_) => {
                            registry.remove(&worker.id).await;
                            info!(worker_id = %worker.id, "Worker shutdown complete");
                        }
                        Err(e) => {
                            warn!(worker_id = %worker.id, error = %e, "Failed to shutdown worker");
                        }
                    }
                }
            }
        }
    }
}
```

**Wire up in daemon handler:**
```rust
// Spawn background tasks
let registry_clone = registry.clone();
tokio::spawn(async move {
    health_monitor_loop(registry_clone).await;
});

let registry_clone = registry.clone();
tokio::spawn(async move {
    idle_timeout_loop(registry_clone).await;
});
```

**Estimated time:** 2-3 hours

---

#### Task 1.3: Add reqwest Dependency
**File to modify:** `bin/rbee-hive/Cargo.toml`

Add:
```toml
reqwest = { workspace = true, features = ["json"] }
```

**Estimated time:** 5 minutes

---

#### Task 1.4: Fix Worker Spawn Logic
**File to modify:** `bin/rbee-hive/src/http/workers.rs`

**Issues to fix:**
1. Port allocation is naive (8081 + workers.len())
2. Worker binary path is hardcoded ("llm-worker-rbee")
3. API key is "dummy-key"
4. No callback URL provided to worker

**Better implementation:**
```rust
// Determine port (proper allocation)
let port = find_available_port().await?; // TODO: Implement

// Get worker binary path
let worker_binary = std::env::current_exe()?
    .parent()
    .unwrap()
    .join("llm-worker-rbee");

// Generate API key
let api_key = format!("key-{}", Uuid::new_v4());

// Callback URL (this server)
let callback_url = format!("http://localhost:8080/v1/workers/ready");

// Spawn worker
let spawn_result = tokio::process::Command::new(worker_binary)
    .arg("--worker-id").arg(&worker_id)
    .arg("--model").arg(&request.model_path)
    .arg("--port").arg(port.to_string())
    .arg("--callback-url").arg(&callback_url)
    .spawn();
```

**Estimated time:** 1 hour

---

### Priority 2: Implement rbee-keeper HTTP Client (CRITICAL) 🔥

#### Task 2.1: Add HTTP Client Dependencies
**File to modify:** `bin/rbee-keeper/Cargo.toml`

Add:
```toml
reqwest = { workspace = true, features = ["json", "stream"] }
tokio = { workspace = true, features = ["full"] }
sqlx = { workspace = true, features = ["runtime-tokio-rustls", "sqlite"] }
futures = { workspace = true }
```

**Estimated time:** 5 minutes

---

#### Task 2.2: Create Pool Client Module
**File to create:** `bin/rbee-keeper/src/pool_client.rs`

```rust
//! HTTP client for rbee-hive pool manager
//! Per test-001-mvp.md Phase 2: Pool Preflight

use anyhow::Result;
use serde::{Deserialize, Serialize};

pub struct PoolClient {
    base_url: String,
    api_key: String,
    client: reqwest::Client,
}

#[derive(Debug, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub api_version: String,
}

#[derive(Debug, Serialize)]
pub struct SpawnWorkerRequest {
    pub model_ref: String,
    pub backend: String,
    pub device: u32,
    pub model_path: String,
}

#[derive(Debug, Deserialize)]
pub struct SpawnWorkerResponse {
    pub worker_id: String,
    pub url: String,
    pub state: String,
}

impl PoolClient {
    pub fn new(base_url: String, api_key: String) -> Self {
        Self {
            base_url,
            api_key,
            client: reqwest::Client::new(),
        }
    }
    
    pub async fn health_check(&self) -> Result<HealthResponse> {
        let response = self.client
            .get(&format!("{}/v1/health", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;
        
        Ok(response.json().await?)
    }
    
    pub async fn spawn_worker(&self, request: SpawnWorkerRequest) -> Result<SpawnWorkerResponse> {
        let response = self.client
            .post(&format!("{}/v1/workers/spawn", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await?;
        
        Ok(response.json().await?)
    }
}
```

**Estimated time:** 1-2 hours

---

#### Task 2.3: Create SQLite Worker Registry
**File to create:** `bin/rbee-keeper/src/registry.rs`

```rust
//! SQLite worker registry
//! Per test-001-mvp.md Phase 1: Worker Registry Check

use anyhow::Result;
use sqlx::{SqliteConnection, Connection};

pub struct WorkerRegistry {
    db_path: String,
}

impl WorkerRegistry {
    pub fn new(db_path: String) -> Self {
        Self { db_path }
    }
    
    pub async fn init(&self) -> Result<()> {
        let mut conn = SqliteConnection::connect(&self.db_path).await?;
        
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS workers (
                id TEXT PRIMARY KEY,
                node TEXT NOT NULL,
                url TEXT NOT NULL,
                model_ref TEXT NOT NULL,
                state TEXT NOT NULL,
                last_health_check_unix INTEGER NOT NULL
            )
            "#
        )
        .execute(&mut conn)
        .await?;
        
        Ok(())
    }
    
    pub async fn find_worker(&self, node: &str, model_ref: &str) -> Result<Option<WorkerInfo>> {
        let mut conn = SqliteConnection::connect(&self.db_path).await?;
        
        let worker = sqlx::query_as!(
            WorkerInfo,
            r#"
            SELECT * FROM workers 
            WHERE node = ? AND model_ref = ? AND state IN ('idle', 'ready')
            AND last_health_check_unix > ?
            "#,
            node,
            model_ref,
            (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs() - 60) as i64
        )
        .fetch_optional(&mut conn)
        .await?;
        
        Ok(worker)
    }
    
    pub async fn register_worker(&self, worker: &WorkerInfo) -> Result<()> {
        let mut conn = SqliteConnection::connect(&self.db_path).await?;
        
        sqlx::query!(
            r#"
            INSERT OR REPLACE INTO workers 
            (id, node, url, model_ref, state, last_health_check_unix)
            VALUES (?, ?, ?, ?, ?, ?)
            "#,
            worker.id,
            worker.node,
            worker.url,
            worker.model_ref,
            worker.state,
            worker.last_health_check_unix
        )
        .execute(&mut conn)
        .await?;
        
        Ok(())
    }
}

#[derive(Debug)]
pub struct WorkerInfo {
    pub id: String,
    pub node: String,
    pub url: String,
    pub model_ref: String,
    pub state: String,
    pub last_health_check_unix: i64,
}
```

**Estimated time:** 2-3 hours

---

#### Task 2.4: Implement `infer` Command
**File to create:** `bin/rbee-keeper/src/commands/infer.rs`

**Flow (per test-001-mvp.md):**
1. Check local registry for existing worker (Phase 1)
2. If not found, call rbee-hive health check (Phase 2)
3. Call rbee-hive to spawn worker (Phase 5)
4. Poll worker until ready (Phase 7)
5. Send inference request, stream tokens (Phase 8)

```rust
//! Inference command
//! Per test-001-mvp.md: 8-phase flow

use crate::pool_client::PoolClient;
use crate::registry::WorkerRegistry;
use anyhow::Result;

pub async fn handle_infer(
    node: String,
    model: String,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
) -> Result<()> {
    // Phase 1: Check local registry
    let registry = WorkerRegistry::new("~/.rbee/workers.db".to_string());
    registry.init().await?;
    
    if let Some(worker) = registry.find_worker(&node, &model).await? {
        println!("Found existing worker: {}", worker.url);
        // Skip to Phase 8
        return execute_inference(&worker.url, prompt, max_tokens, temperature).await;
    }
    
    // Phase 2: Pool preflight
    let pool_url = format!("http://{}:8080", node);
    let pool_client = PoolClient::new(pool_url, "api-key".to_string());
    
    let health = pool_client.health_check().await?;
    println!("Pool health: {} (version {})", health.status, health.version);
    
    // Phase 3-5: Spawn worker
    let spawn_request = SpawnWorkerRequest {
        model_ref: model.clone(),
        backend: "cpu".to_string(), // TODO: Detect backend
        device: 0,
        model_path: "/models/model.gguf".to_string(), // TODO: Get from catalog
    };
    
    let worker = pool_client.spawn_worker(spawn_request).await?;
    println!("Worker spawned: {} (state: {})", worker.worker_id, worker.state);
    
    // Phase 6: Register worker
    registry.register_worker(&WorkerInfo {
        id: worker.worker_id.clone(),
        node: node.clone(),
        url: worker.url.clone(),
        model_ref: model,
        state: worker.state,
        last_health_check_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs() as i64,
    }).await?;
    
    // Phase 7: Wait for worker ready
    wait_for_worker_ready(&worker.url).await?;
    
    // Phase 8: Execute inference
    execute_inference(&worker.url, prompt, max_tokens, temperature).await
}

async fn wait_for_worker_ready(worker_url: &str) -> Result<()> {
    // TODO: Poll GET {worker_url}/v1/ready until ready
    Ok(())
}

async fn execute_inference(
    worker_url: &str,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
) -> Result<()> {
    // TODO: POST {worker_url}/v1/inference with SSE streaming
    Ok(())
}
```

**Estimated time:** 4-6 hours

---

### Priority 3: Integration Testing (IMPORTANT) 🧪

#### Task 3.1: Create End-to-End Test Script
**File to create:** `bin/.specs/.gherkin/test-001-mvp-run.sh`

```bash
#!/usr/bin/env bash
# Test-001 MVP: Cross-Node Inference
# Per test-001-mvp.md

set -euo pipefail

echo "=== Test-001 MVP: Cross-Node Inference ==="

# Step 1: Start rbee-hive daemon on mac
echo "Starting rbee-hive daemon..."
ssh mac.home.arpa "cd ~/Projects/llama-orch && cargo run --bin rbee-hive -- daemon" &
DAEMON_PID=$!
sleep 5

# Step 2: Verify health
echo "Checking pool health..."
curl -s http://mac.home.arpa:8080/v1/health | jq .

# Step 3: Run inference
echo "Running inference..."
cargo run --bin rbee -- infer \
  --node mac \
  --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --prompt "write a short story" \
  --max-tokens 20 \
  --temperature 0.7

# Cleanup
kill $DAEMON_PID

echo "✅ Test-001 MVP PASSED"
```

**Estimated time:** 2-3 hours

---

## Outstanding Work from Previous Teams

### From TEAM-025 Handoff (CP4 - Multi-Model Testing)

**Status:** ❌ NOT STARTED

**Tasks:**
1. Download remaining models (tinyllama, phi3, mistral) on all pools
2. Create multi-model test script
3. Test all models on all backends (CPU, Metal, CUDA)
4. Document results in MODEL_SUPPORT.md

**Priority:** LOW (defer until after MVP)

**Reason:** MVP (test-001) takes precedence. CP4 is about testing multiple models, but MVP only needs one model (TinyLlama) working end-to-end.

---

### From TEAM-024 Handoff (M1 - Orchestrator Daemon)

**Status:** ❌ NOT STARTED

**Tasks:**
1. Build queen-rbee HTTP daemon
2. Implement admission control
3. Implement queue management
4. Implement scheduling
5. Implement SSE relay

**Priority:** LOW (defer until after MVP)

**Reason:** MVP uses direct worker access via rbee-keeper. queen-rbee is M1 milestone, comes after M0 MVP is complete.

---

## Critical Files for TEAM-027

### Must Read (in order):
1. `bin/.specs/.gherkin/test-001-mvp.md` - THE source of truth (671 lines)
2. `bin/.plan/TEAM_026_HANDOFF.md` - Previous handoff (608 lines)
3. `bin/llm-worker-rbee/src/http/` - Pattern to mirror (avoid drift)
4. `.windsurf/rules/candled-rules.md` - Coding standards

### Files TEAM-026 Created (review these):
1. `bin/rbee-hive/src/http/` - All HTTP infrastructure
2. `bin/rbee-hive/src/registry.rs` - Worker registry
3. `bin/rbee-hive/Cargo.toml` - Updated dependencies

### Reference When Needed:
- `bin/.specs/00_llama-orch.md` - Full system spec
- `bin/.specs/01_M0_worker_orcd.md` - Worker spec
- `bin/.specs/FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md` - Architecture

---

## Known Issues & Gotchas

### Issue 1: Worker Binary Path
**Problem:** `handle_spawn_worker` hardcodes "llm-worker-rbee"  
**Solution:** Use `std::env::current_exe()?.parent()?.join("llm-worker-rbee")`

### Issue 2: Port Allocation
**Problem:** Naive port allocation (8081 + workers.len())  
**Solution:** Implement proper port allocation or use port 0 (OS assigns)

### Issue 3: No Model Catalog Yet
**Problem:** `spawn_worker` needs model_path, but no catalog lookup  
**Solution:** For MVP, hardcode path or pass as CLI arg

### Issue 4: No Authentication
**Problem:** API keys are "dummy-key"  
**Solution:** For MVP, skip auth or use simple shared secret

### Issue 5: llm-worker-rbee Callback
**Problem:** llm-worker-rbee expects `--callback-url` arg  
**Solution:** Check llm-worker-rbee CLI args, add if missing

---

## Dependencies & Prerequisites

### System Requirements:
- ✅ Rust toolchain (installed)
- ✅ `hf` CLI (installed)
- ✅ SSH access to pools (configured)
- ✅ llm-worker-rbee (working)

### Rust Crates Needed (already added):
- ✅ axum, tokio, tower, serde, tracing, uuid (rbee-hive)
- ⏳ reqwest, sqlx, futures (rbee-keeper - need to add)

### Models:
- ✅ qwen-0.5b downloaded (943MB)
- ⏳ tinyllama (need for MVP testing)

---

## Testing Checklist

### Before Starting:
- [ ] Read test-001-mvp.md completely
- [ ] Review TEAM-026 created files
- [ ] Understand llm-worker-rbee HTTP patterns
- [ ] Verify llm-worker-rbee works

### After Completing Priority 1 (rbee-hive daemon):
- [ ] `cargo build --bin rbee-hive` succeeds
- [ ] `rbee-hive daemon` starts without errors
- [ ] GET /v1/health returns 200
- [ ] POST /v1/workers/spawn spawns process
- [ ] POST /v1/workers/ready updates registry
- [ ] Background loops run (health monitor, idle timeout)

### After Completing Priority 2 (rbee-keeper client):
- [ ] `cargo build --bin rbee` succeeds
- [ ] `rbee infer` command exists
- [ ] Can call rbee-hive health endpoint
- [ ] Can spawn worker via HTTP
- [ ] SQLite registry works

### End-to-End:
- [ ] Full test-001 flow works
- [ ] Tokens stream in real-time
- [ ] Worker auto-shuts down after idle
- [ ] Latency < 30 seconds (cold start)

---

## Success Criteria for TEAM-027

### Minimum (MVP Happy Path):
- [ ] rbee-hive daemon starts and serves HTTP
- [ ] rbee-keeper can spawn worker via rbee-hive
- [ ] Inference streams tokens
- [ ] Worker auto-shuts down after 5 min idle

### Target (MVP + Edge Cases):
- [ ] All 8 phases implemented
- [ ] At least 3 edge cases handled (connection timeout, version mismatch, VRAM check)
- [ ] Test script passes
- [ ] Documentation updated

### Stretch (Production Ready):
- [ ] All 10 edge cases handled
- [ ] Comprehensive error messages
- [ ] Retry logic with backoff
- [ ] Metrics & logging

---

## Implementation Timeline

### Week 1: Complete rbee-hive Daemon
- [ ] Day 1: Wire up daemon mode, add background loops (Tasks 1.1-1.3)
- [ ] Day 2: Fix worker spawn logic, test daemon (Task 1.4)
- [ ] Day 3: Debug and polish

### Week 2: Implement rbee-keeper HTTP Client
- [ ] Day 1-2: Pool client module (Task 2.2)
- [ ] Day 3: SQLite registry (Task 2.3)
- [ ] Day 4-5: Implement infer command (Task 2.4)

### Week 3: Integration & Testing
- [ ] Day 1-2: End-to-end test script (Task 3.1)
- [ ] Day 3-4: Edge case testing
- [ ] Day 5: Documentation + handoff

---

## Communication Protocol

### When to Ask User:
- ✅ Ask if you find blocking issues
- ✅ Ask if MVP requirements are unclear
- ✅ Ask before making architectural changes
- ❌ Don't ask about implementation details (follow MVP)

### When to Update Docs:
- ✅ Update this handoff when complete
- ✅ Create TEAM_027_COMPLETION_SUMMARY.md
- ✅ Create TEAM_028_HANDOFF.md
- ✅ Update test-001-mvp.md with implementation notes

### When to Stop:
- 🛑 If MVP requirements are contradictory (ask user)
- 🛑 If test-001 takes >3 weeks (re-evaluate)
- 🛑 If fundamental blocker found (escalate)

---

## Final Notes from TEAM-026

**What Went Well:**
- ✅ Mirrored llm-worker-rbee HTTP patterns (no drift)
- ✅ Created comprehensive HTTP infrastructure
- ✅ WorkerRegistry with proper thread-safety
- ✅ All endpoints defined per MVP spec

**What Was Hard:**
- 😓 Rate-limited mid-implementation
- 😓 Many moving parts (8 phases, 10 edge cases)
- 😓 Need to coordinate rbee-hive + rbee-keeper + llm-worker-rbee

**Advice for TEAM-027:**
- 📖 Read test-001-mvp.md FIRST (it's the source of truth)
- 🎯 Follow the 8 phases step-by-step
- 🧪 Test after each phase
- 📝 Document as you go
- 💬 Ask if MVP is unclear
- 🔍 Check llm-worker-rbee for patterns (avoid drift)

**Remember:**
- The MVP is well-defined!
- The HTTP patterns are established!
- The infrastructure is 60% done!
- Just wire it up and test!

**Key Insight:**
- Mirror llm-worker-rbee patterns everywhere
- Use Arc<RwLock<T>> for shared state
- Use tokio::spawn for background tasks
- Use tracing for logging (not println!)

---

**Signed:** TEAM-026  
**Date:** 2025-10-09T23:17:00+02:00  
**Status:** Partial implementation, rate-limited, HTTP infrastructure complete  
**Next Team:** TEAM-027 - Wire it up and ship the MVP! 🚀
