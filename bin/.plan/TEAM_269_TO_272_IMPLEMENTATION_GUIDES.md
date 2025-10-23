# TEAM-269 to TEAM-272: Implementation Guides

**Phases:** 3-6 of 9  
**Teams:** 269 (Model Provisioner), 270 (Worker Registry), 271 (Worker Spawn), 272 (Worker Management)

**Status:** ✅ Individual guides created

---

## 📋 Individual Team Guides

Each team now has a dedicated implementation guide:

1. **TEAM_269_MODEL_PROVISIONER.md** - Model downloading (24-32h)
2. **TEAM_270_WORKER_REGISTRY.md** - Worker registry & types (20-24h)
3. **TEAM_271_WORKER_LIFECYCLE_SPAWN.md** - Worker spawning (32-40h) ⚠️ Most complex
4. **TEAM_272_WORKER_LIFECYCLE_MGMT.md** - Worker management (24-32h)

**Total Effort:** 100-128 hours for phases 3-6

---

## Quick Reference

Below is a consolidated overview. **See individual guides for full details.**

---

## TEAM-269: Model Provisioner (Phase 3)

**Estimated Effort:** 24-32 hours  
**Prerequisites:** TEAM-268 complete

### Mission

Implement model downloading from HuggingFace Hub. Add ModelDownload operation to job_router.

### Key Deliverables

1. **download_model() function** - Downloads model files from HuggingFace
2. **Progress tracking** - Updates ModelCatalog status during download
3. **File management** - Stores models in ~/.cache/rbee/models/
4. **ModelDownload operation** - Wired up in job_router.rs

### Implementation Outline

**File:** `bin/25_rbee_hive_crates/model-provisioner/src/lib.rs`

```rust
// TEAM-269: Model provisioner implementation
use anyhow::{anyhow, Result};
use rbee_hive_model_catalog::{ModelCatalog, ModelEntry, ModelStatus};
use std::path::PathBuf;
use observability_narration_core::NarrationFactory;

const NARRATE: NarrationFactory = NarrationFactory::new("model-prov");

pub struct ModelProvisioner {
    catalog: Arc<ModelCatalog>,
    cache_dir: PathBuf,
}

impl ModelProvisioner {
    pub fn new(catalog: Arc<ModelCatalog>) -> Result<Self> {
        // Use the same directory as ModelCatalog
        // Linux/Mac: ~/.cache/rbee/models/
        // Windows: %LOCALAPPDATA%\rbee\models\
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow!("Cannot determine cache directory"))?
            .join("rbee")
            .join("models");
        
        std::fs::create_dir_all(&cache_dir)?;
        
        Ok(Self { catalog, cache_dir })
    }
    
    pub async fn download_model(
        &self,
        job_id: &str,
        model_id: &str,
    ) -> Result<String> {
        NARRATE
            .action("download_start")
            .job_id(job_id)
            .context(model_id)
            .human("📥 Starting download: {}")
            .emit();
        
        // Check if already exists
        if self.catalog.contains(model_id) {
            return Err(anyhow!("Model '{}' already exists", model_id));
        }
        
        // Create model entry with Downloading status
        let model_path = self.cache_dir.join(model_id);
        let mut model = ModelEntry::new(
            model_id.to_string(),
            model_id.to_string(),
            model_path.clone(),
            0, // Size unknown until download
        );
        model.status = ModelStatus::Downloading { progress: 0.0 };
        
        // Add to catalog (creates directory and metadata.yaml)
        self.catalog.add(model)?;
        
        // TODO: Actual HuggingFace download
        // For now, create placeholder directory
        tokio::fs::create_dir_all(&model_path).await?;
        
        NARRATE
            .action("download_progress")
            .job_id(job_id)
            .context("50")
            .human("Progress: {}%")
            .emit();
        
        // Update status to Ready
        self.catalog.update_status(model_id, ModelStatus::Ready)?;
        
        NARRATE
            .action("download_complete")
            .job_id(job_id)
            .context(model_id)
            .human("✅ Download complete: {}")
            .emit();
        
        Ok(model_id.to_string())
    }
    
    pub async fn delete_model_files(&self, model_id: &str) -> Result<()> {
        let model = self.catalog.get(model_id)?;
        
        if model.path.exists() {
            tokio::fs::remove_dir_all(&model.path).await?;
        }
        
        Ok(())
    }
}
```

**Wire up in job_router.rs:**

```rust
Operation::ModelDownload { hive_id, model } => {
    // TEAM-269: Implemented model download
    NARRATE
        .action("model_download_start")
        .job_id(&job_id)
        .context(&hive_id)
        .context(&model)
        .human("📥 Downloading model '{}' on hive '{}'")
        .emit();

    match state.model_provisioner.download_model(&job_id, &model).await {
        Ok(model_id) => {
            NARRATE
                .action("model_download_complete")
                .job_id(&job_id)
                .context(&model_id)
                .human("✅ Model downloaded: {}")
                .emit();
        }
        Err(e) => {
            NARRATE
                .action("model_download_error")
                .job_id(&job_id)
                .context(&model)
                .human("❌ Download failed: {}")
                .emit();
            return Err(e);
        }
    }
}
```

### Acceptance Criteria

- [ ] ModelProvisioner struct implemented
- [ ] download_model() function working
- [ ] Progress tracking via ModelCatalog status updates
- [ ] ModelDownload operation wired up
- [ ] Files stored in ~/.cache/rbee/models/
- [ ] Narration events emitted
- [ ] `cargo check --bin rbee-hive` passes

---

## TEAM-270: Worker Registry (Phase 4)

**Estimated Effort:** 20-24 hours  
**Prerequisites:** TEAM-269 complete

### Mission

Implement worker registry for tracking running worker processes. Similar to model-catalog but for workers.

### Key Deliverables

1. **Worker types** - WorkerEntry, WorkerStatus
2. **WorkerRegistry** - Arc<Mutex<HashMap>> storage
3. **CRUD operations** - add, get, remove, list
4. **Unit tests**

### Implementation Outline

**File:** `bin/25_rbee_hive_crates/worker-lifecycle/src/registry.rs`

```rust
// TEAM-270: Worker registry
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use anyhow::{anyhow, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerEntry {
    pub id: String,
    pub model_id: String,
    pub device: String,
    pub pid: u32,
    pub port: u16,
    pub status: WorkerStatus,
    pub started_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WorkerStatus {
    Starting,
    Ready,
    Busy,
    Stopped,
    Failed { error: String },
}

#[derive(Clone)]
pub struct WorkerRegistry {
    workers: Arc<Mutex<HashMap<String, WorkerEntry>>>,
}

impl WorkerRegistry {
    pub fn new() -> Self {
        Self {
            workers: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    pub fn register(&self, worker: WorkerEntry) -> Result<()> {
        let mut workers = self.workers.lock().unwrap();
        
        if workers.contains_key(&worker.id) {
            return Err(anyhow!("Worker '{}' already exists", worker.id));
        }
        
        workers.insert(worker.id.clone(), worker);
        Ok(())
    }
    
    pub fn get(&self, id: &str) -> Result<WorkerEntry> {
        let workers = self.workers.lock().unwrap();
        workers
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow!("Worker '{}' not found", id))
    }
    
    pub fn remove(&self, id: &str) -> Result<WorkerEntry> {
        let mut workers = self.workers.lock().unwrap();
        workers
            .remove(id)
            .ok_or_else(|| anyhow!("Worker '{}' not found", id))
    }
    
    pub fn list(&self) -> Vec<WorkerEntry> {
        let workers = self.workers.lock().unwrap();
        workers.values().cloned().collect()
    }
    
    pub fn update_status(&self, id: &str, status: WorkerStatus) -> Result<()> {
        let mut workers = self.workers.lock().unwrap();
        let worker = workers
            .get_mut(id)
            .ok_or_else(|| anyhow!("Worker '{}' not found", id))?;
        worker.status = status;
        Ok(())
    }
}
```

### Acceptance Criteria

- [ ] WorkerEntry struct defined
- [ ] WorkerStatus enum defined
- [ ] WorkerRegistry implemented
- [ ] CRUD operations working
- [ ] Unit tests passing
- [ ] `cargo check --package rbee-hive-worker-lifecycle` passes

---

## TEAM-271: Worker Lifecycle - Spawn (Phase 5)

**Estimated Effort:** 32-40 hours (MOST COMPLEX PHASE)  
**Prerequisites:** TEAM-270 complete

### Mission

Implement worker spawning. This is the most complex phase - spawning actual worker processes, managing ports, and registering them.

### Key Deliverables

1. **spawn_worker() function** - Spawns worker process
2. **Port allocation** - Finds available port for worker
3. **Process management** - Tracks worker PID
4. **WorkerSpawn operation** - Wired up in job_router.rs

### Implementation Outline

**File:** `bin/25_rbee_hive_crates/worker-lifecycle/src/spawn.rs`

```rust
// TEAM-271: Worker spawning
use crate::registry::{WorkerEntry, WorkerRegistry, WorkerStatus};
use anyhow::{anyhow, Result};
use observability_narration_core::NarrationFactory;
use std::process::Stdio;
use tokio::process::Command;

const NARRATE: NarrationFactory = NarrationFactory::new("worker-lc");

pub struct WorkerSpawner {
    registry: Arc<WorkerRegistry>,
    worker_binary: PathBuf,
}

impl WorkerSpawner {
    pub fn new(registry: Arc<WorkerRegistry>) -> Result<Self> {
        // Worker binary location strategy:
        // 1. Development: ./target/debug/llama-worker or ./target/release/llama-worker
        // 2. Production: System PATH or ~/.local/bin/llama-worker
        // For now, use target directory for easier rebuilds during development
        
        let worker_binary = if cfg!(debug_assertions) {
            PathBuf::from("./target/debug/llama-worker")
        } else {
            // Try multiple locations
            let candidates = vec![
                PathBuf::from("./target/release/llama-worker"),
                PathBuf::from("/usr/local/bin/llama-worker"),
                dirs::home_dir()
                    .map(|h| h.join(".local/bin/llama-worker"))
                    .unwrap_or_else(|| PathBuf::from("llama-worker")),
            ];
            
            candidates
                .into_iter()
                .find(|p| p.exists())
                .unwrap_or_else(|| PathBuf::from("llama-worker"))
        };
        
        Ok(Self {
            registry,
            worker_binary,
        })
    }
    
    pub async fn spawn_worker(
        &self,
        job_id: &str,
        worker_id: &str,
        model_id: &str,
        device: &str,
    ) -> Result<String> {
        NARRATE
            .action("spawn_start")
            .job_id(job_id)
            .context(worker_id)
            .context(model_id)
            .context(device)
            .human("🚀 Spawning worker '{}' with model '{}' on device {}")
            .emit();
        
        // Find available port
        let port = self.find_available_port().await?;
        
        NARRATE
            .action("spawn_port")
            .job_id(job_id)
            .context(&port.to_string())
            .human("Allocated port: {}")
            .emit();
        
        // Spawn worker process
        let mut child = Command::new(&self.worker_binary)
            .arg("--model")
            .arg(model_id)
            .arg("--device")
            .arg(device)
            .arg("--port")
            .arg(port.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        
        let pid = child.id().ok_or_else(|| anyhow!("Failed to get worker PID"))?;
        
        NARRATE
            .action("spawn_process")
            .job_id(job_id)
            .context(&pid.to_string())
            .human("Worker process started: PID {}")
            .emit();
        
        // Register worker
        let worker = WorkerEntry {
            id: worker_id.to_string(),
            model_id: model_id.to_string(),
            device: device.to_string(),
            pid,
            port,
            status: WorkerStatus::Starting,
            started_at: chrono::Utc::now(),
        };
        
        self.registry.register(worker)?;
        
        // TODO: Wait for worker to be ready (health check)
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        
        self.registry.update_status(worker_id, WorkerStatus::Ready)?;
        
        NARRATE
            .action("spawn_complete")
            .job_id(job_id)
            .context(worker_id)
            .human("✅ Worker ready: {}")
            .emit();
        
        Ok(worker_id.to_string())
    }
    
    async fn find_available_port(&self) -> Result<u16> {
        // Simple port allocation: start at 9100 and find first available
        for port in 9100..9200 {
            if tokio::net::TcpListener::bind(("127.0.0.1", port)).await.is_ok() {
                return Ok(port);
            }
        }
        Err(anyhow!("No available ports in range 9100-9200"))
    }
}
```

**Wire up in job_router.rs:**

```rust
Operation::WorkerSpawn { hive_id, model, worker, device } => {
    // TEAM-271: Implemented worker spawn
    NARRATE
        .action("worker_spawn_start")
        .job_id(&job_id)
        .context(&hive_id)
        .context(&model)
        .context(&worker)
        .context(&device.to_string())
        .human("🚀 Spawning worker '{}' with model '{}' on device {}")
        .emit();

    match state.worker_spawner.spawn_worker(&job_id, &worker, &model, &device).await {
        Ok(worker_id) => {
            NARRATE
                .action("worker_spawn_complete")
                .job_id(&job_id)
                .context(&worker_id)
                .human("✅ Worker spawned: {}")
                .emit();
        }
        Err(e) => {
            NARRATE
                .action("worker_spawn_error")
                .job_id(&job_id)
                .context(&worker)
                .human("❌ Spawn failed: {}")
                .emit();
            return Err(e);
        }
    }
}
```

### Acceptance Criteria

- [ ] WorkerSpawner struct implemented
- [ ] spawn_worker() function working
- [ ] Port allocation working
- [ ] Process spawning working
- [ ] Worker registration working
- [ ] WorkerSpawn operation wired up
- [ ] Narration events emitted
- [ ] `cargo check --bin rbee-hive` passes

### Known Issues

⚠️ **Worker binary may not exist yet** - Document this limitation. Use placeholder or mock for now.

---

## TEAM-272: Worker Lifecycle - Management (Phase 6)

**Estimated Effort:** 24-32 hours  
**Prerequisites:** TEAM-271 complete

### Mission

Implement worker management operations: WorkerList, WorkerGet, WorkerDelete.

### Key Deliverables

1. **WorkerList operation** - List all workers
2. **WorkerGet operation** - Get worker details
3. **WorkerDelete operation** - Stop and remove worker
4. **Process cleanup** - Kill worker process on delete

### Implementation Outline

**Wire up in job_router.rs:**

```rust
Operation::WorkerList { hive_id } => {
    // TEAM-272: Implemented worker list
    NARRATE
        .action("worker_list_start")
        .job_id(&job_id)
        .context(&hive_id)
        .human("📋 Listing workers on hive '{}'")
        .emit();

    let workers = state.worker_registry.list();
    
    NARRATE
        .action("worker_list_result")
        .job_id(&job_id)
        .context(workers.len().to_string())
        .human("Found {} worker(s)")
        .emit();
    
    for worker in &workers {
        NARRATE
            .action("worker_list_entry")
            .job_id(&job_id)
            .context(&worker.id)
            .context(&worker.model_id)
            .context(&worker.device)
            .context(&worker.port.to_string())
            .human("  {} | {} | {} | port {}")
            .emit();
    }
}

Operation::WorkerGet { hive_id, id } => {
    // TEAM-272: Implemented worker get
    NARRATE
        .action("worker_get_start")
        .job_id(&job_id)
        .context(&hive_id)
        .context(&id)
        .human("🔍 Getting worker '{}' on hive '{}'")
        .emit();

    match state.worker_registry.get(&id) {
        Ok(worker) => {
            let json = serde_json::to_string_pretty(&worker)
                .unwrap_or_else(|_| "Failed to serialize".to_string());
            
            NARRATE
                .action("worker_get_details")
                .job_id(&job_id)
                .human(&json)
                .emit();
        }
        Err(e) => {
            NARRATE
                .action("worker_get_error")
                .job_id(&job_id)
                .context(&id)
                .human("❌ Worker '{}' not found: {}")
                .emit();
            return Err(e);
        }
    }
}

Operation::WorkerDelete { hive_id, id } => {
    // TEAM-272: Implemented worker delete
    NARRATE
        .action("worker_delete_start")
        .job_id(&job_id)
        .context(&hive_id)
        .context(&id)
        .human("🗑️  Deleting worker '{}' on hive '{}'")
        .emit();

    match state.worker_registry.get(&id) {
        Ok(worker) => {
            // Kill process
            NARRATE
                .action("worker_delete_kill")
                .job_id(&job_id)
                .context(&worker.pid.to_string())
                .human("Killing process PID {}")
                .emit();
            
            // TODO: Actually kill the process
            // use nix::sys::signal::{kill, Signal};
            // kill(Pid::from_raw(worker.pid as i32), Signal::SIGTERM)?;
            
            // Remove from registry
            state.worker_registry.remove(&id)?;
            
            NARRATE
                .action("worker_delete_complete")
                .job_id(&job_id)
                .context(&id)
                .human("✅ Worker deleted: {}")
                .emit();
        }
        Err(e) => {
            NARRATE
                .action("worker_delete_error")
                .job_id(&job_id)
                .context(&id)
                .human("❌ Delete failed: {}")
                .emit();
            return Err(e);
        }
    }
}
```

### Acceptance Criteria

- [ ] WorkerList operation implemented
- [ ] WorkerGet operation implemented
- [ ] WorkerDelete operation implemented
- [ ] Process cleanup working (or documented as TODO)
- [ ] All operations emit narration
- [ ] `cargo check --bin rbee-hive` passes

---

## Summary

**TEAM-269:** Model downloading (24-32h)  
**TEAM-270:** Worker registry (20-24h)  
**TEAM-271:** Worker spawning (32-40h) ← Most complex  
**TEAM-272:** Worker management (24-32h)

**Total:** 100-128 hours for phases 3-6

**Next:** TEAM-273 integrates everything in job_router.rs
