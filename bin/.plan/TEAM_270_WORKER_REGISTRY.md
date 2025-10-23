# TEAM-270: Worker Registry & Types

**Phase:** 4 of 9  
**Estimated Effort:** 20-24 hours  
**Prerequisites:** TEAM-269 complete  
**Blocks:** TEAM-271 (Worker Spawn)

---

## 🎯 Mission

Implement worker registry for tracking running worker processes. Similar to model-catalog but uses in-memory Arc<Mutex<HashMap>> storage since workers are ephemeral.

**Deliverables:**
1. ✅ WorkerEntry struct with all fields
2. ✅ WorkerStatus enum
3. ✅ WorkerRegistry with Arc<Mutex<HashMap>>
4. ✅ CRUD operations (register, get, remove, list)
5. ✅ Unit tests
6. ✅ Documentation

---

## 📁 Files to Create/Modify

```
bin/25_rbee_hive_crates/worker-lifecycle/
├── src/
│   ├── lib.rs          ← Module exports
│   ├── types.rs        ← WorkerEntry, WorkerStatus
│   └── registry.rs     ← WorkerRegistry implementation
├── Cargo.toml          ← Add dependencies
└── README.md           ← Document API
```

---

## 🏗️ Implementation Guide

### Step 1: Add Dependencies (Cargo.toml)

```toml
[dependencies]
anyhow = "1.0"
chrono = { version = "0.4", features = ["serde"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[dev-dependencies]
# Add dev dependencies as needed
```

### Step 2: Implement Types (types.rs)

```rust
// TEAM-270: Worker types
use serde::{Deserialize, Serialize};

/// Worker entry in the registry
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerEntry {
    /// Unique worker ID
    pub id: String,

    /// Model being served
    pub model_id: String,

    /// Device (e.g., "CPU-0", "GPU-0")
    pub device: String,

    /// Process ID
    pub pid: u32,

    /// HTTP port
    pub port: u16,

    /// Current status
    pub status: WorkerStatus,

    /// When the worker was started
    pub started_at: chrono::DateTime<chrono::Utc>,
}

/// Worker status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WorkerStatus {
    /// Worker is starting up
    Starting,

    /// Worker is ready to serve requests
    Ready,

    /// Worker is processing a request
    Busy,

    /// Worker has been stopped
    Stopped,

    /// Worker failed to start or crashed
    Failed {
        /// Error message
        error: String,
    },
}

impl WorkerEntry {
    /// Create a new worker entry
    pub fn new(
        id: String,
        model_id: String,
        device: String,
        pid: u32,
        port: u16,
    ) -> Self {
        Self {
            id,
            model_id,
            device,
            pid,
            port,
            status: WorkerStatus::Starting,
            started_at: chrono::Utc::now(),
        }
    }

    /// Check if worker is ready
    pub fn is_ready(&self) -> bool {
        matches!(self.status, WorkerStatus::Ready)
    }

    /// Check if worker is running (not stopped or failed)
    pub fn is_running(&self) -> bool {
        !matches!(
            self.status,
            WorkerStatus::Stopped | WorkerStatus::Failed { .. }
        )
    }
}
```

### Step 3: Implement Registry (registry.rs)

```rust
// TEAM-270: Worker registry implementation
use crate::types::{WorkerEntry, WorkerStatus};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// In-memory worker registry
///
/// Uses Arc<Mutex<HashMap>> for thread-safe access.
/// Workers are ephemeral - registry is cleared on hive restart.
#[derive(Clone)]
pub struct WorkerRegistry {
    workers: Arc<Mutex<HashMap<String, WorkerEntry>>>,
}

impl WorkerRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            workers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Register a new worker
    ///
    /// # Errors
    /// Returns error if worker ID already exists
    pub fn register(&self, worker: WorkerEntry) -> Result<()> {
        let mut workers = self.workers.lock().unwrap();

        if workers.contains_key(&worker.id) {
            return Err(anyhow!("Worker '{}' already exists", worker.id));
        }

        workers.insert(worker.id.clone(), worker);
        Ok(())
    }

    /// Get a worker by ID
    ///
    /// # Errors
    /// Returns error if worker not found
    pub fn get(&self, id: &str) -> Result<WorkerEntry> {
        let workers = self.workers.lock().unwrap();
        workers
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow!("Worker '{}' not found", id))
    }

    /// Remove a worker from registry
    ///
    /// # Errors
    /// Returns error if worker not found
    pub fn remove(&self, id: &str) -> Result<WorkerEntry> {
        let mut workers = self.workers.lock().unwrap();
        workers
            .remove(id)
            .ok_or_else(|| anyhow!("Worker '{}' not found", id))
    }

    /// List all workers
    pub fn list(&self) -> Vec<WorkerEntry> {
        let workers = self.workers.lock().unwrap();
        workers.values().cloned().collect()
    }

    /// List workers by status
    pub fn list_by_status(&self, status_filter: fn(&WorkerStatus) -> bool) -> Vec<WorkerEntry> {
        let workers = self.workers.lock().unwrap();
        workers
            .values()
            .filter(|w| status_filter(&w.status))
            .cloned()
            .collect()
    }

    /// Update worker status
    ///
    /// # Errors
    /// Returns error if worker not found
    pub fn update_status(&self, id: &str, status: WorkerStatus) -> Result<()> {
        let mut workers = self.workers.lock().unwrap();
        let worker = workers
            .get_mut(id)
            .ok_or_else(|| anyhow!("Worker '{}' not found", id))?;
        worker.status = status;
        Ok(())
    }

    /// Check if worker exists
    pub fn contains(&self, id: &str) -> bool {
        let workers = self.workers.lock().unwrap();
        workers.contains_key(id)
    }

    /// Get registry size (number of workers)
    pub fn len(&self) -> usize {
        let workers = self.workers.lock().unwrap();
        workers.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all workers (for testing or shutdown)
    pub fn clear(&self) {
        let mut workers = self.workers.lock().unwrap();
        workers.clear();
    }
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_worker(id: &str) -> WorkerEntry {
        WorkerEntry::new(
            id.to_string(),
            "test-model".to_string(),
            "CPU-0".to_string(),
            12345,
            9100,
        )
    }

    #[test]
    fn test_register_and_get() {
        let registry = WorkerRegistry::new();
        let worker = create_test_worker("worker-1");

        registry.register(worker.clone()).unwrap();
        let retrieved = registry.get("worker-1").unwrap();

        assert_eq!(retrieved.id, "worker-1");
        assert_eq!(retrieved.model_id, "test-model");
    }

    #[test]
    fn test_register_duplicate() {
        let registry = WorkerRegistry::new();
        let worker = create_test_worker("worker-1");

        registry.register(worker.clone()).unwrap();
        let result = registry.register(worker);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[test]
    fn test_remove() {
        let registry = WorkerRegistry::new();
        let worker = create_test_worker("worker-1");

        registry.register(worker).unwrap();
        assert_eq!(registry.len(), 1);

        let removed = registry.remove("worker-1").unwrap();
        assert_eq!(removed.id, "worker-1");
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_list() {
        let registry = WorkerRegistry::new();

        for i in 0..3 {
            let worker = create_test_worker(&format!("worker-{}", i));
            registry.register(worker).unwrap();
        }

        let workers = registry.list();
        assert_eq!(workers.len(), 3);
    }

    #[test]
    fn test_update_status() {
        let registry = WorkerRegistry::new();
        let worker = create_test_worker("worker-1");

        registry.register(worker).unwrap();

        registry
            .update_status("worker-1", WorkerStatus::Ready)
            .unwrap();

        let updated = registry.get("worker-1").unwrap();
        assert_eq!(updated.status, WorkerStatus::Ready);
    }

    #[test]
    fn test_list_by_status() {
        let registry = WorkerRegistry::new();

        // Add ready worker
        let mut worker1 = create_test_worker("worker-1");
        worker1.status = WorkerStatus::Ready;
        registry.register(worker1).unwrap();

        // Add starting worker
        let worker2 = create_test_worker("worker-2");
        registry.register(worker2).unwrap();

        let ready_workers = registry.list_by_status(|s| matches!(s, WorkerStatus::Ready));
        assert_eq!(ready_workers.len(), 1);
        assert_eq!(ready_workers[0].id, "worker-1");
    }

    #[test]
    fn test_contains() {
        let registry = WorkerRegistry::new();
        let worker = create_test_worker("worker-1");

        assert!(!registry.contains("worker-1"));
        registry.register(worker).unwrap();
        assert!(registry.contains("worker-1"));
    }

    #[test]
    fn test_worker_is_ready() {
        let mut worker = create_test_worker("worker-1");
        assert!(!worker.is_ready());

        worker.status = WorkerStatus::Ready;
        assert!(worker.is_ready());
    }

    #[test]
    fn test_worker_is_running() {
        let mut worker = create_test_worker("worker-1");
        assert!(worker.is_running()); // Starting is running

        worker.status = WorkerStatus::Ready;
        assert!(worker.is_running());

        worker.status = WorkerStatus::Stopped;
        assert!(!worker.is_running());
    }
}
```

### Step 4: Update lib.rs

```rust
// TEAM-270: Worker lifecycle crate
#![warn(missing_docs)]
#![warn(clippy::all)]

//! rbee-hive-worker-lifecycle
//!
//! Worker lifecycle management for rbee-hive.
//!
//! This crate provides:
//! - Worker registry for tracking running workers
//! - Worker spawning and management
//! - Process lifecycle management

/// Worker registry
pub mod registry;
/// Worker types
pub mod types;

// Re-export main types
pub use registry::WorkerRegistry;
pub use types::{WorkerEntry, WorkerStatus};
```

---

## ✅ Acceptance Criteria

- [ ] WorkerEntry struct defined with all fields
- [ ] WorkerStatus enum with 5 variants
- [ ] WorkerRegistry implemented with Arc<Mutex<HashMap>>
- [ ] register() method working
- [ ] get() method working
- [ ] remove() method working
- [ ] list() method working
- [ ] update_status() method working
- [ ] Unit tests passing (8+ tests)
- [ ] `cargo check --package rbee-hive-worker-lifecycle` passes
- [ ] `cargo test --package rbee-hive-worker-lifecycle` passes
- [ ] Public API documented

---

## 🧪 Testing Commands

```bash
# Check compilation
cargo check --package rbee-hive-worker-lifecycle

# Run unit tests
cargo test --package rbee-hive-worker-lifecycle

# Run with output
cargo test --package rbee-hive-worker-lifecycle -- --nocapture

# Check documentation
cargo doc --package rbee-hive-worker-lifecycle --open
```

---

## 📝 Handoff Checklist

Create `TEAM_270_HANDOFF.md` with:

- [ ] WorkerRegistry implementation complete
- [ ] All CRUD operations working
- [ ] Unit tests passing (8+ tests)
- [ ] Example usage code
- [ ] Notes for TEAM-271 (spawning)

---

## 🎓 Key Patterns

### 1. Arc<Mutex<HashMap>> Pattern

**Why:** Workers are ephemeral and need thread-safe in-memory storage.

**Usage:**
```rust
let workers = Arc::new(Mutex::new(HashMap::new()));

// In methods
let mut workers = self.workers.lock().unwrap();
workers.insert(key, value);
```

### 2. Clone-able Registry

**Why:** Registry needs to be shared across threads.

**Usage:**
```rust
#[derive(Clone)]
pub struct WorkerRegistry {
    workers: Arc<Mutex<HashMap<String, WorkerEntry>>>,
}
```

### 3. Builder Pattern for WorkerEntry

```rust
let worker = WorkerEntry::new(id, model, device, pid, port);
// Status defaults to Starting
```

---

## 📚 Reference Implementations

- **job-server:** Arc<Mutex<HashMap>> patterns
- **model-catalog:** Similar CRUD operations (but filesystem-based)
- **hive-lifecycle:** Registry patterns

---

## 🚨 Common Issues

### Issue 1: Mutex Poisoning

If a thread panics while holding the lock, the Mutex becomes "poisoned".

**Solution:** Use `.unwrap()` for now (acceptable in v0.1.0).

### Issue 2: Deadlocks

Holding multiple locks can cause deadlocks.

**Solution:** Keep lock scope minimal, release before calling other methods.

---

**TEAM-270: Build that registry! 📋🚀**
