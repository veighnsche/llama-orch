# CORRECTION: TEAM-269 to TEAM-272 Architecture Fix

**Date:** Oct 23, 2025  
**Status:** 🔴 ARCHITECTURE ISSUES IDENTIFIED  
**Action Required:** Revise plans before implementation

---

## 🚨 Critical Issues Identified

### Issue 1: Model Provisioner Should Be in Model Catalog
**Current Plan:** Separate `model-provisioner` crate  
**Problem:** Provisioner is tightly coupled to catalog, creates unnecessary separation  
**Solution:** Consolidate into `model-catalog` with vendor-specific sections

### Issue 2: Worker Registry in Wrong Place
**Current Plan:** Worker registry in hive  
**Problem:** Hive just executes operations, doesn't track workers. Queen tracks workers via heartbeats.  
**Solution:** Remove worker registry from hive, move worker tracking to queen

### Issue 3: Missing Worker Provisioner Concept
**Current Plan:** No worker provisioner  
**Problem:** Workers need to be downloaded/built just like models (infinite vendors possible)  
**Solution:** Create worker provisioner pattern similar to model provisioner

### Issue 4: Weak Worker Contract
**Current Plan:** No formal worker contract  
**Problem:** Need clear interface for infinite worker implementations  
**Solution:** Define robust worker contract (like hive contract)

---

## 📋 Corrected Architecture

### Separation of Concerns

```
┌─────────────────────────────────────────────────────────────┐
│ QUEEN (Orchestrator)                                        │
│ - Tracks workers via heartbeats                             │
│ - Routes inference requests to workers                      │
│ - Manages hive lifecycle (start/stop/install)               │
│ - Worker registry (who's alive, what they serve)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP (operations)
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ HIVE (Executor)                                             │
│ - Executes operations (spawn/delete/list)                   │
│ - NO worker tracking (stateless executor)                   │
│ - NO worker registry                                        │
│ - Reports back via SSE                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Process spawn
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ WORKER (Inference Engine)                                   │
│ - Sends heartbeat to queen (not hive!)                      │
│ - Serves inference requests                                 │
│ - Implements worker contract                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Corrected Implementation Plan

### TEAM-269: Model Catalog Enhancement (Revised)

**Mission:** Add model provisioning to model-catalog crate with vendor support.

**Changes from Original:**
- ❌ NO separate `model-provisioner` crate
- ✅ Add provisioning to `model-catalog`
- ✅ Vendor-specific sections (HuggingFace first)
- ✅ Extensible for future vendors

**File Structure:**
```
bin/25_rbee_hive_crates/model-catalog/
├── src/
│   ├── lib.rs
│   ├── catalog.rs          (existing)
│   ├── types.rs            (existing)
│   ├── provisioner/        (NEW)
│   │   ├── mod.rs          ← Main provisioner
│   │   ├── huggingface.rs  ← HF-specific implementation
│   │   └── traits.rs       ← Vendor trait for future extensions
└── Cargo.toml
```

**Implementation:**

```rust
// src/provisioner/traits.rs
// TEAM-269: Vendor trait for extensibility
pub trait ModelVendor {
    async fn download_model(&self, model_id: &str, dest: &Path) -> Result<u64>;
    fn supports_model(&self, model_id: &str) -> bool;
}

// src/provisioner/huggingface.rs
// TEAM-269: HuggingFace implementation (first vendor)
pub struct HuggingFaceVendor {
    // HF-specific config
}

impl ModelVendor for HuggingFaceVendor {
    async fn download_model(&self, model_id: &str, dest: &Path) -> Result<u64> {
        // HF Hub API implementation
        // For v0.1.0: placeholder
    }
    
    fn supports_model(&self, model_id: &str) -> bool {
        // Check if model_id looks like HF format
        model_id.contains('/') // e.g., "meta-llama/Llama-2-7b"
    }
}

// src/provisioner/mod.rs
// TEAM-269: Main provisioner with vendor routing
pub struct ModelProvisioner {
    catalog: Arc<ModelCatalog>,
    vendors: Vec<Box<dyn ModelVendor>>,
}

impl ModelProvisioner {
    pub fn new(catalog: Arc<ModelCatalog>) -> Self {
        let vendors: Vec<Box<dyn ModelVendor>> = vec![
            Box::new(HuggingFaceVendor::default()), // First vendor
            // Future: Box::new(OllamaVendor::default()),
            // Future: Box::new(LocalVendor::default()),
        ];
        
        Self { catalog, vendors }
    }
    
    pub async fn download_model(&self, job_id: &str, model_id: &str) -> Result<String> {
        // Find vendor that supports this model
        let vendor = self.vendors.iter()
            .find(|v| v.supports_model(model_id))
            .ok_or_else(|| anyhow!("No vendor supports model '{}'", model_id))?;
        
        // Download using vendor
        let model_path = self.catalog.model_path(model_id);
        let size = vendor.download_model(model_id, &model_path).await?;
        
        // Register in catalog
        let model = ModelEntry::new(model_id.to_string(), model_id.to_string(), model_path, size);
        self.catalog.add(model)?;
        
        Ok(model_id.to_string())
    }
}
```

**Deliverables:**
1. ✅ Vendor trait for extensibility
2. ✅ HuggingFace vendor implementation (first)
3. ✅ ModelProvisioner in model-catalog
4. ✅ ModelDownload operation wired up
5. ✅ Clear extension points for future vendors

---

### TEAM-270: Worker Contract Definition (Revised)

**Mission:** Define robust worker contract, NOT implement worker registry in hive.

**Changes from Original:**
- ❌ NO worker registry in hive
- ❌ NO worker-lifecycle crate in hive
- ✅ Define worker contract (like hive contract)
- ✅ Document worker interface
- ✅ Create worker contract types

**File Structure:**
```
contracts/
├── worker-contract/        (NEW)
│   ├── src/
│   │   ├── lib.rs
│   │   ├── types.rs        ← WorkerInfo, WorkerStatus
│   │   ├── heartbeat.rs    ← Heartbeat protocol
│   │   └── api.rs          ← Worker HTTP API spec
│   └── Cargo.toml
└── openapi/
    └── worker-api.yaml     (NEW - OpenAPI spec)
```

**Worker Contract:**

```rust
// contracts/worker-contract/src/types.rs
// TEAM-270: Worker contract types

/// Worker information sent in heartbeat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Unique worker ID
    pub id: String,
    
    /// Model being served
    pub model_id: String,
    
    /// Device (e.g., "CPU-0", "GPU-0")
    pub device: String,
    
    /// HTTP port
    pub port: u16,
    
    /// Current status
    pub status: WorkerStatus,
    
    /// Worker implementation (e.g., "llama-cpp", "vllm", "ollama")
    pub implementation: String,
    
    /// Worker version
    pub version: String,
}

/// Worker status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WorkerStatus {
    Starting,
    Ready,
    Busy,
    Stopped,
}

/// Heartbeat message sent from worker to queen
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHeartbeat {
    pub worker: WorkerInfo,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}
```

**Worker HTTP API Contract:**

```rust
// contracts/worker-contract/src/api.rs
// TEAM-270: Worker HTTP API specification

/// Worker must implement these endpoints:

/// GET /health
/// Returns: 200 OK if worker is alive

/// POST /v1/infer
/// Body: InferRequest
/// Returns: InferResponse (streaming or non-streaming)

/// GET /info
/// Returns: WorkerInfo

/// POST /heartbeat (internal - called by worker itself)
/// Sends heartbeat to queen
```

**Deliverables:**
1. ✅ Worker contract types
2. ✅ Heartbeat protocol
3. ✅ Worker HTTP API spec
4. ✅ OpenAPI documentation
5. ✅ Extension points for multiple implementations

---

### TEAM-271: Worker Spawn Operation (Revised)

**Mission:** Implement WorkerSpawn operation in hive (stateless execution only).

**Changes from Original:**
- ❌ NO worker registry in hive
- ❌ NO worker tracking in hive
- ✅ Just spawn process and return
- ✅ Worker sends heartbeat to queen (not hive)
- ✅ Hive is stateless executor

**File Structure:**
```
bin/20_rbee_hive/
└── src/
    ├── job_router.rs       ← WorkerSpawn operation
    └── spawn.rs            (NEW - spawn logic)
```

**Implementation:**

```rust
// bin/20_rbee_hive/src/spawn.rs
// TEAM-271: Worker spawning (stateless)

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::process::Command;

const NARRATE: NarrationFactory = NarrationFactory::new("hv-spawn");

/// Spawn a worker process (stateless - just spawn and return)
pub async fn spawn_worker(
    job_id: &str,
    worker_id: &str,
    model_id: &str,
    device: &str,
    queen_url: &str, // Where to send heartbeat
) -> Result<SpawnResult> {
    NARRATE
        .action("spawn_start")
        .job_id(job_id)
        .context(worker_id)
        .context(model_id)
        .human("🚀 Spawning worker '{}' with model '{}'")
        .emit();
    
    // Find available port
    let port = find_available_port().await?;
    
    // Find worker binary
    let worker_binary = find_worker_binary()?;
    
    // Spawn worker process
    let mut child = Command::new(&worker_binary)
        .arg("--worker-id").arg(worker_id)
        .arg("--model").arg(model_id)
        .arg("--device").arg(device)
        .arg("--port").arg(port.to_string())
        .arg("--queen-url").arg(queen_url) // Worker sends heartbeat here
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    
    let pid = child.id().ok_or_else(|| anyhow!("Failed to get PID"))?;
    
    NARRATE
        .action("spawn_complete")
        .job_id(job_id)
        .context(&pid.to_string())
        .context(&port.to_string())
        .human("✅ Worker spawned: PID {}, port {}")
        .emit();
    
    // Return spawn info (hive doesn't track it)
    Ok(SpawnResult {
        worker_id: worker_id.to_string(),
        pid,
        port,
    })
}

pub struct SpawnResult {
    pub worker_id: String,
    pub pid: u32,
    pub port: u16,
}
```

**Key Changes:**
- Hive spawns worker and returns immediately
- Worker sends heartbeat to queen (via `--queen-url` arg)
- Hive doesn't track worker state
- Queen receives heartbeat and tracks worker

**Deliverables:**
1. ✅ spawn_worker() function (stateless)
2. ✅ Port allocation
3. ✅ Worker binary resolution
4. ✅ WorkerSpawn operation wired up
5. ✅ Worker configured to send heartbeat to queen

---

### TEAM-272: Worker Operations (Revised)

**Mission:** Implement WorkerList, WorkerGet, WorkerDelete operations (query queen, execute on hive).

**Changes from Original:**
- ❌ NO worker registry in hive
- ✅ WorkerList queries queen's registry
- ✅ WorkerDelete kills process by PID
- ✅ Hive is stateless executor

**Implementation:**

```rust
// bin/20_rbee_hive/src/job_router.rs
// TEAM-272: Worker operations

Operation::WorkerList { hive_id } => {
    // TEAM-272: List workers
    // NOTE: This should actually query queen's registry
    // For now, return empty list (hive doesn't track workers)
    
    NARRATE
        .action("worker_list_start")
        .job_id(&job_id)
        .context(&hive_id)
        .human("📋 Listing workers on hive '{}'")
        .emit();
    
    // TODO: Query queen's registry via HTTP
    // let workers = query_queen_workers(&hive_id).await?;
    
    NARRATE
        .action("worker_list_empty")
        .job_id(&job_id)
        .human("⚠️  Hive doesn't track workers. Query queen's registry instead.")
        .emit();
}

Operation::WorkerDelete { hive_id, id } => {
    // TEAM-272: Delete worker
    // This is valid - hive can kill process by PID
    // But needs to get PID from queen first
    
    NARRATE
        .action("worker_delete_start")
        .job_id(&job_id)
        .context(&id)
        .human("🗑️  Deleting worker '{}'")
        .emit();
    
    // TODO: Get worker info from queen
    // let worker_info = query_queen_worker(&id).await?;
    
    // Kill process by PID
    // kill_process(worker_info.pid).await?;
    
    NARRATE
        .action("worker_delete_complete")
        .job_id(&job_id)
        .human("✅ Worker process killed")
        .emit();
}
```

**Deliverables:**
1. ✅ WorkerList (queries queen)
2. ✅ WorkerGet (queries queen)
3. ✅ WorkerDelete (kills process)
4. ✅ Document queen integration needed

---

## 🎯 Corrected Phase Summary

### TEAM-269: Model Catalog Enhancement
**Effort:** 24-32 hours  
**Focus:** Add provisioning to model-catalog with vendor support (HF first)  
**Key:** Vendor trait for extensibility

### TEAM-270: Worker Contract Definition
**Effort:** 16-20 hours (reduced - no registry implementation)  
**Focus:** Define worker contract, types, API spec  
**Key:** Clear interface for infinite worker implementations

### TEAM-271: Worker Spawn (Stateless)
**Effort:** 20-24 hours (reduced - no tracking)  
**Focus:** Spawn worker process, return immediately  
**Key:** Worker sends heartbeat to queen, not hive

### TEAM-272: Worker Operations (Queen Integration)
**Effort:** 16-20 hours (reduced - query queen)  
**Focus:** Operations that query queen's registry  
**Key:** Hive is stateless executor

**Total:** 76-96 hours (reduced from 100-128 hours)

---

## 📝 Implementation Order

1. **TEAM-269:** Model catalog provisioning with vendor support
2. **TEAM-270:** Worker contract definition
3. **TEAM-271:** Worker spawn (stateless)
4. **TEAM-272:** Worker operations (queen integration)
5. **TEAM-273:** Queen worker registry (NEW - tracks heartbeats)

---

## 🚨 Critical Architectural Principles

### 1. Separation of Concerns
- **Queen:** Orchestrator, tracks state, routes requests
- **Hive:** Stateless executor, runs operations, reports back
- **Worker:** Inference engine, sends heartbeat to queen

### 2. Heartbeat Flow
```
Worker → Queen (heartbeat every 5s)
Queen → Tracks worker state
Queen → Routes inference to worker
```

### 3. Hive is Stateless
- Hive spawns workers but doesn't track them
- Hive can kill workers by PID (if given PID)
- Hive doesn't know which workers are alive

### 4. Vendor Extensibility
- Models: HuggingFace, Ollama, Local, etc.
- Workers: llama-cpp, vllm, ollama, custom, etc.
- Clear trait/contract for each

---

## 📚 Next Steps

1. **Review this document** with team
2. **Update TEAM-269 guide** with vendor approach
3. **Update TEAM-270 guide** with contract focus
4. **Update TEAM-271 guide** with stateless approach
5. **Update TEAM-272 guide** with queen integration
6. **Create TEAM-273 guide** for queen worker registry
7. **Update START_HERE** document with corrected plan

---

## ✅ Validation Checklist

Before implementing, verify:
- [ ] Model provisioner is in model-catalog
- [ ] Vendor trait exists for extensibility
- [ ] HuggingFace is first vendor (sectioned)
- [ ] Worker contract is defined
- [ ] Worker sends heartbeat to queen (not hive)
- [ ] Hive is stateless (no worker tracking)
- [ ] Queen tracks workers (not hive)
- [ ] Clear extension points for future vendors/workers

---

**This correction ensures proper separation of concerns and extensibility!**

**Do NOT implement TEAM-269 to TEAM-272 until these corrections are applied!**
