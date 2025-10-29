# TEAM-271: Worker Lifecycle - Spawn

**Phase:** 5 of 9  
**Estimated Effort:** 32-40 hours ⚠️ **MOST COMPLEX PHASE**  
**Prerequisites:** TEAM-270 complete  
**Blocks:** TEAM-272 (Worker Management)

---

## 🎯 Mission

Implement worker spawning - the most complex phase. Spawn actual worker processes, manage ports, track PIDs, and register workers.

**Deliverables:**
1. ✅ WorkerSpawner struct
2. ✅ spawn_worker() function
3. ✅ Port allocation logic
4. ✅ Process management
5. ✅ WorkerSpawn operation wired up
6. ✅ Narration events
7. ✅ Unit tests

---

## 📁 Files to Create/Modify

```
bin/25_rbee_hive_crates/worker-lifecycle/
├── src/
│   ├── lib.rs          ← Export spawn module
│   ├── spawn.rs        ← Implement WorkerSpawner
│   └── ports.rs        ← Port allocation (optional separate module)
└── Cargo.toml          ← Add dependencies

bin/20_rbee_hive/
├── src/
│   ├── job_router.rs   ← Wire up WorkerSpawn operation
│   └── main.rs         ← Initialize WorkerSpawner
└── Cargo.toml          ← Add dependencies
```

---

## 🏗️ Implementation Guide

### Step 1: Add Dependencies (worker-lifecycle/Cargo.toml)

```toml
[dependencies]
anyhow = "1.0"
chrono = { version = "0.4", features = ["serde"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { workspace = true, features = ["full", "process"] }
dirs = "5.0"

# Narration
observability-narration-core = { path = "../../99_shared_crates/narration-core" }

[dev-dependencies]
tempfile = "3.8"
```

### Step 2: Implement WorkerSpawner (spawn.rs)

```rust
// TEAM-271: Worker spawning implementation
use crate::registry::{WorkerEntry, WorkerRegistry, WorkerStatus};
use anyhow::{anyhow, Result};
use observability_narration_core::NarrationFactory;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use tokio::process::Command;

const NARRATE: NarrationFactory = NarrationFactory::new("worker-lc");

/// Worker spawner for creating and managing worker processes
pub struct WorkerSpawner {
    registry: Arc<WorkerRegistry>,
    worker_binary: PathBuf,
}

impl WorkerSpawner {
    /// Create a new worker spawner
    ///
    /// # Worker Binary Location Strategy
    ///
    /// 1. **Development:** `./target/debug/llama-worker` or `./target/release/llama-worker`
    /// 2. **Production:** System PATH or `~/.local/bin/llama-worker`
    ///
    /// For v0.1.0, we prioritize development builds for easier iteration.
    pub fn new(registry: Arc<WorkerRegistry>) -> Result<Self> {
        let worker_binary = Self::find_worker_binary()?;

        Ok(Self {
            registry,
            worker_binary,
        })
    }

    /// Find the worker binary
    fn find_worker_binary() -> Result<PathBuf> {
        // Development mode: prefer target directory
        if cfg!(debug_assertions) {
            let debug_path = PathBuf::from("./target/debug/llama-worker");
            if debug_path.exists() {
                return Ok(debug_path);
            }
        }

        // Try multiple locations
        let candidates = vec![
            PathBuf::from("./target/release/llama-worker"),
            PathBuf::from("/usr/local/bin/llama-worker"),
            dirs::home_dir()
                .map(|h| h.join(".local/bin/llama-worker"))
                .unwrap_or_else(|| PathBuf::from("llama-worker")),
            PathBuf::from("llama-worker"), // Last resort: system PATH
        ];

        for candidate in candidates {
            if candidate.exists() {
                return Ok(candidate);
            }
        }

        // If no binary found, return the last candidate (system PATH)
        // This will fail at spawn time with a clear error
        Ok(PathBuf::from("llama-worker"))
    }

    /// Spawn a new worker process
    ///
    /// # Arguments
    /// * `job_id` - Job ID for narration routing
    /// * `worker_id` - Unique worker identifier
    /// * `model_id` - Model to load
    /// * `device` - Device to use (e.g., "CPU-0", "GPU-0")
    ///
    /// # Returns
    /// Worker ID on success
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

        // Check if worker already exists
        if self.registry.contains(worker_id) {
            return Err(anyhow!("Worker '{}' already exists", worker_id));
        }

        // Find available port
        let port = self.find_available_port().await?;

        NARRATE
            .action("spawn_port")
            .job_id(job_id)
            .context(&port.to_string())
            .human("Allocated port: {}")
            .emit();

        // Spawn worker process
        NARRATE
            .action("spawn_binary")
            .job_id(job_id)
            .context(&self.worker_binary.display().to_string())
            .human("Executing: {}")
            .emit();

        let mut child = Command::new(&self.worker_binary)
            .arg("--model")
            .arg(model_id)
            .arg("--device")
            .arg(device)
            .arg("--port")
            .arg(port.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                anyhow!(
                    "Failed to spawn worker binary '{}': {}. \
                     Make sure the worker binary exists and is executable.",
                    self.worker_binary.display(),
                    e
                )
            })?;

        let pid = child
            .id()
            .ok_or_else(|| anyhow!("Failed to get worker PID"))?;

        NARRATE
            .action("spawn_process")
            .job_id(job_id)
            .context(&pid.to_string())
            .human("Worker process started: PID {}")
            .emit();

        // Register worker
        let worker = WorkerEntry::new(
            worker_id.to_string(),
            model_id.to_string(),
            device.to_string(),
            pid,
            port,
        );

        self.registry.register(worker)?;

        NARRATE
            .action("spawn_registered")
            .job_id(job_id)
            .context(worker_id)
            .human("Worker registered in registry")
            .emit();

        // TODO: Wait for worker to be ready (health check)
        // For v0.1.0, use simple delay
        NARRATE
            .action("spawn_waiting")
            .job_id(job_id)
            .human("Waiting for worker to be ready...")
            .emit();

        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        // Update status to Ready
        self.registry
            .update_status(worker_id, WorkerStatus::Ready)?;

        NARRATE
            .action("spawn_complete")
            .job_id(job_id)
            .context(worker_id)
            .human("✅ Worker ready: {}")
            .emit();

        Ok(worker_id.to_string())
    }

    /// Find an available port for a worker
    ///
    /// Tries ports in range 9100-9200.
    async fn find_available_port(&self) -> Result<u16> {
        for port in 9100..9200 {
            // Try to bind to the port
            if let Ok(listener) = tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
                // Port is available, drop the listener and return
                drop(listener);
                return Ok(port);
            }
        }
        Err(anyhow!("No available ports in range 9100-9200"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_find_available_port() {
        let registry = Arc::new(WorkerRegistry::new());
        let spawner = WorkerSpawner {
            registry,
            worker_binary: PathBuf::from("test-binary"),
        };

        let port = spawner.find_available_port().await;
        assert!(port.is_ok());
        assert!(port.unwrap() >= 9100 && port.unwrap() < 9200);
    }

    #[test]
    fn test_find_worker_binary() {
        // This test just verifies the function doesn't panic
        let result = WorkerSpawner::find_worker_binary();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_spawn_worker_duplicate() {
        let registry = Arc::new(WorkerRegistry::new());

        // Pre-register a worker
        let worker = WorkerEntry::new(
            "worker-1".to_string(),
            "model".to_string(),
            "CPU-0".to_string(),
            12345,
            9100,
        );
        registry.register(worker).unwrap();

        let spawner = WorkerSpawner {
            registry,
            worker_binary: PathBuf::from("test-binary"),
        };

        // Try to spawn duplicate
        let result = spawner
            .spawn_worker("job-1", "worker-1", "model", "CPU-0")
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }
}
```

### Step 3: Update lib.rs

```rust
// Add spawn module
pub mod spawn;

// Re-export
pub use spawn::WorkerSpawner;
```

### Step 4: Wire Up in job_router.rs

```rust
// Add to imports
use rbee_hive_worker_lifecycle::WorkerSpawner;

// Add to JobState
pub struct JobState {
    pub registry: Arc<JobRegistry<String>>,
    pub model_catalog: Arc<ModelCatalog>,
    pub model_provisioner: Arc<ModelProvisioner>,
    pub worker_registry: Arc<WorkerRegistry>, // TEAM-270
    pub worker_spawner: Arc<WorkerSpawner>,   // TEAM-271: Added
}

// Replace WorkerSpawn TODO stub
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

    // Convert device number to device string
    let device_str = format!("GPU-{}", device);

    match state
        .worker_spawner
        .spawn_worker(&job_id, &worker, &model, &device_str)
        .await
    {
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
                .context(&e.to_string())
                .human("❌ Spawn failed for '{}': {}")
                .emit();
            return Err(e);
        }
    }
}
```

### Step 5: Initialize in main.rs

```rust
// Add imports
use rbee_hive_worker_lifecycle::{WorkerRegistry, WorkerSpawner};

// After model_provisioner initialization
let worker_registry = Arc::new(WorkerRegistry::new());

NARRATE
    .action("worker_registry_init")
    .context(&worker_registry.len().to_string())
    .human("👷 Worker registry initialized ({} workers)")
    .emit();

let worker_spawner = Arc::new(
    WorkerSpawner::new(worker_registry.clone())
        .expect("Failed to initialize worker spawner")
);

NARRATE
    .action("worker_spawner_init")
    .human("🚀 Worker spawner initialized")
    .emit();

// Update HiveState
let job_state = http::jobs::HiveState {
    registry: job_registry,
    model_catalog,
    model_provisioner,
    worker_registry,  // TEAM-270
    worker_spawner,   // TEAM-271: Added
};
```

### Step 6: Update http/jobs.rs

```rust
// Add imports
use rbee_hive_worker_lifecycle::{WorkerRegistry, WorkerSpawner};

// Update HiveState
pub struct HiveState {
    pub registry: Arc<JobRegistry<String>>,
    pub model_catalog: Arc<ModelCatalog>,
    pub model_provisioner: Arc<ModelProvisioner>,
    pub worker_registry: Arc<WorkerRegistry>,  // TEAM-270
    pub worker_spawner: Arc<WorkerSpawner>,    // TEAM-271: Added
}

// Update From implementation
impl From<HiveState> for crate::job_router::JobState {
    fn from(state: HiveState) -> Self {
        Self {
            registry: state.registry,
            model_catalog: state.model_catalog,
            model_provisioner: state.model_provisioner,
            worker_registry: state.worker_registry,  // TEAM-270
            worker_spawner: state.worker_spawner,    // TEAM-271: Added
        }
    }
}
```

---

## ✅ Acceptance Criteria

- [ ] WorkerSpawner struct implemented
- [ ] spawn_worker() function working
- [ ] Port allocation working (9100-9200 range)
- [ ] Process spawning working (or documented limitation)
- [ ] Worker registration working
- [ ] WorkerSpawn operation wired up
- [ ] Narration events emitted with `.job_id()`
- [ ] Unit tests passing (3+ tests)
- [ ] `cargo check --bin rbee-hive` passes
- [ ] `cargo test --package rbee-hive-worker-lifecycle` passes

---

## 🧪 Testing Commands

```bash
# Check compilation
cargo check --package rbee-hive-worker-lifecycle
cargo check --bin rbee-hive

# Run unit tests
cargo test --package rbee-hive-worker-lifecycle

# Manual testing (requires worker binary)
cargo run --bin rbee-hive -- --port 8600

# In another terminal
curl -X POST http://localhost:8600/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation": "worker_spawn", "hive_id": "localhost", "model": "test-model", "worker": "worker-1", "device": 0}'
```

---

## 📝 Handoff Checklist

Create `TEAM_271_HANDOFF.md` with:

- [ ] WorkerSpawner implementation complete
- [ ] spawn_worker() function working
- [ ] Port allocation demonstrated
- [ ] Example narration output
- [ ] Known limitations (worker binary)
- [ ] Notes for TEAM-272

---

## 🚨 Known Limitations

### 1. Worker Binary May Not Exist

**Current:** Worker binary (`llama-worker`) may not be built yet.

**Impact:** Spawn will fail with clear error message.

**Workaround:** Document this limitation. TEAM-271 can:
1. Create a mock worker binary for testing
2. Document the requirement
3. Proceed with implementation

**Mock Worker Binary:**
```bash
# Create simple mock for testing
cat > /tmp/mock-worker.sh << 'EOF'
#!/bin/bash
echo "Mock worker starting..."
echo "Model: $2"
echo "Device: $4"
echo "Port: $6"
sleep 3600  # Keep running
EOF
chmod +x /tmp/mock-worker.sh

# Use in WorkerSpawner for testing
let worker_binary = PathBuf::from("/tmp/mock-worker.sh");
```

### 2. No Health Check

**Current:** Uses 2-second delay instead of actual health check.

**Future:** Implement HTTP health check to worker:
```rust
let health_url = format!("http://127.0.0.1:{}/health", port);
for _ in 0..10 {
    if reqwest::get(&health_url).await.is_ok() {
        break;
    }
    tokio::time::sleep(Duration::from_millis(500)).await;
}
```

### 3. No Process Monitoring

**Current:** No monitoring of worker process after spawn.

**Future:** Implement process monitoring to detect crashes.

---

## 🎓 Learning Resources

- **tokio::process:** https://docs.rs/tokio/latest/tokio/process/
- **Port allocation:** https://doc.rust-lang.org/std/net/struct.TcpListener.html
- **Process management:** https://doc.rust-lang.org/std/process/

---

## 📚 Reference Implementations

- **daemon-lifecycle:** Process spawning patterns
- **hive-lifecycle/start.rs:** Binary resolution and spawning

---

**TEAM-271: Spawn those workers! 🚀💪**

**This is the hardest phase - take your time and test thoroughly!**
