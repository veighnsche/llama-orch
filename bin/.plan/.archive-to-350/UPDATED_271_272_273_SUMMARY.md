# Updated TEAM-271, TEAM-272, TEAM-273 Summary

**Date:** Oct 23, 2025  
**Status:** Architecture corrected per CORRECTION_269_TO_272_ARCHITECTURE_FIX.md

---

## ✅ TEAM-273: Shared Artifact Catalog (COMPLETE)

**Created:** `bin/25_rbee_hive_crates/artifact-catalog/`

**Purpose:** Eliminate duplication between model-catalog and worker-catalog

**Key Abstractions:**
- `Artifact` trait - Generic artifact interface
- `ArtifactCatalog<T>` trait - CRUD operations
- `FilesystemCatalog<T>` - Concrete filesystem implementation
- `VendorSource` trait - Download from HuggingFace/GitHub/local
- `ArtifactProvisioner<T>` - Multi-vendor coordination

**Impact:** ~400-600 LOC savings across model and worker catalogs

---

## 🔧 TEAM-271: Worker Spawn (CORRECTED)

**Mission:** Spawn workers in hive (STATELESS - no tracking)

**Key Changes from Original:**
- ❌ NO worker registry in hive
- ❌ NO worker tracking in hive
- ✅ Just spawn process and return
- ✅ Worker sends heartbeat to QUEEN (not hive!)
- ✅ Hive is stateless executor

**Implementation:**
```rust
// bin/20_rbee_hive/src/spawn.rs
pub async fn spawn_worker(
    job_id: &str,
    worker_id: &str,
    model_id: &str,
    device: &str,
    queen_url: &str, // ← Worker sends heartbeat HERE
) -> Result<SpawnResult> {
    // 1. Find available port
    // 2. Find worker binary (from worker-catalog)
    // 3. Spawn process with --queen-url
    // 4. Return PID and port (hive doesn't track it)
}
```

**Worker Binary Resolution:**
```
~/.cache/rbee/workers/
├── cpu-llm-worker-rbee/
│   ├── linux/cpu-llm-worker-rbee
│   ├── macos/cpu-llm-worker-rbee
│   └── windows/cpu-llm-worker-rbee.exe
├── cuda-llm-worker-rbee/
│   ├── linux/cuda-llm-worker-rbee
│   └── windows/cuda-llm-worker-rbee.exe
└── metal-llm-worker-rbee/
    └── macos/metal-llm-worker-rbee
```

**Files:**
- `bin/20_rbee_hive/src/spawn.rs` - Stateless spawn logic
- `bin/20_rbee_hive/src/job_router.rs` - WorkerSpawn operation

---

## 🔧 TEAM-272: Worker Operations (CORRECTED)

**Mission:** Implement WorkerList, WorkerGet, WorkerDelete (query queen, execute on hive)

**Key Changes from Original:**
- ❌ NO worker registry in hive
- ✅ WorkerList queries queen's registry via HTTP
- ✅ WorkerGet queries queen's registry via HTTP
- ✅ WorkerDelete kills process by PID (gets PID from queen)

**Implementation:**
```rust
// bin/20_rbee_hive/src/job_router.rs

Operation::WorkerList { hive_id } => {
    // TEAM-272: Query queen's registry
    // TODO: HTTP GET to queen /v1/workers?hive_id={hive_id}
    
    NARRATE
        .action("worker_list_start")
        .job_id(&job_id)
        .human("⚠️  Hive doesn't track workers. Query queen's registry instead.")
        .emit();
}

Operation::WorkerDelete { hive_id, id } => {
    // TEAM-272: Get PID from queen, kill process
    // TODO: HTTP GET to queen /v1/workers/{id} to get PID
    // Then kill process by PID
    
    #[cfg(unix)]
    {
        use nix::sys::signal::{kill, Signal};
        kill(Pid::from_raw(pid as i32), Signal::SIGTERM)?;
        // Wait, then SIGKILL if needed
    }
}
```

**Files:**
- `bin/20_rbee_hive/src/job_router.rs` - Worker operations (query queen)

---

## 🆕 TEAM-273: Worker Catalog & Provisioner (NEW)

**Mission:** Create worker catalog (mirrors model catalog) for worker binary management

**File Structure:**
```
bin/25_rbee_hive_crates/worker-catalog/
├── src/
│   ├── lib.rs
│   ├── types.rs          ← WorkerBinary, WorkerType, Platform
│   ├── catalog.rs        ← WorkerCatalog (uses artifact-catalog)
│   └── provisioner/
│       ├── mod.rs        ← WorkerProvisioner
│       ├── github.rs     ← GitHubReleaseVendor
│       └── local.rs      ← LocalBuildVendor
└── Cargo.toml
```

**Worker Types:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerType {
    CpuLlm,      // cpu-llm-worker-rbee
    CudaLlm,     // cuda-llm-worker-rbee
    MetalLlm,    // metal-llm-worker-rbee
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Platform {
    Linux,
    MacOS,
    Windows,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerBinary {
    id: String,                    // "cpu-llm-worker-rbee-v0.1.0-linux"
    worker_type: WorkerType,
    platform: Platform,
    path: PathBuf,
    size: u64,
    status: ArtifactStatus,
    version: String,
}

impl Artifact for WorkerBinary { /* ... */ }
```

**Vendors:**

1. **GitHubReleaseVendor** - Download from GitHub releases
```rust
impl VendorSource for GitHubReleaseVendor {
    async fn download(&self, id: &str, dest: &Path, job_id: &str) -> Result<u64> {
        // Download from github.com/rbee-ai/rbee/releases/download/v0.1.0/cpu-llm-worker-rbee-linux
    }
    
    fn supports(&self, id: &str) -> bool {
        id.starts_with("GH:") || id.contains("github.com")
    }
}
```

2. **LocalBuildVendor** - Build from source
```rust
impl VendorSource for LocalBuildVendor {
    async fn download(&self, id: &str, dest: &Path, job_id: &str) -> Result<u64> {
        // cargo build --release --bin cpu-llm-worker-rbee
        // Copy from target/release/ to dest
    }
    
    fn supports(&self, id: &str) -> bool {
        id.starts_with("local:")
    }
}
```

**Storage Layout:**
```
~/.cache/rbee/workers/
├── cpu-llm-worker-rbee-v0.1.0-linux/
│   ├── metadata.json
│   └── cpu-llm-worker-rbee
├── cuda-llm-worker-rbee-v0.1.0-linux/
│   ├── metadata.json
│   └── cuda-llm-worker-rbee
└── metal-llm-worker-rbee-v0.1.0-macos/
    ├── metadata.json
    └── metal-llm-worker-rbee
```

---

## 🏗️ Corrected Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ QUEEN (Orchestrator)                                        │
│ - Tracks workers via heartbeats                             │
│ - Routes inference requests to workers                      │
│ - Worker registry (who's alive, what they serve)            │
│ - Manages hive lifecycle (start/stop/install)               │
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
│ - Uses worker-catalog to find binaries                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Process spawn
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ WORKER (Inference Engine)                                   │
│ - Sends heartbeat to QUEEN (not hive!)                      │
│ - Serves inference requests                                 │
│ - Implements worker contract                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Catalog Comparison

| Feature | Model Catalog | Worker Catalog |
|---------|---------------|----------------|
| **Base** | artifact-catalog | artifact-catalog |
| **Artifact Type** | ModelEntry | WorkerBinary |
| **Vendors** | HuggingFace | GitHub, LocalBuild |
| **Storage** | ~/.cache/rbee/models/ | ~/.cache/rbee/workers/ |
| **ID Format** | "meta-llama/Llama-2-7b" | "cpu-llm-worker-rbee-v0.1.0-linux" |
| **Platforms** | N/A (models are cross-platform) | linux/macos/windows |

---

## 🎯 Implementation Order

1. ✅ **TEAM-273A:** Create `artifact-catalog` shared crate (DONE)
2. **TEAM-269:** Update model-catalog to use artifact-catalog
3. **TEAM-273B:** Create worker-catalog using artifact-catalog
4. **TEAM-271:** Implement worker spawn (uses worker-catalog)
5. **TEAM-272:** Implement worker operations (queries queen)

---

## ✅ Key Corrections Applied

1. **Worker Registry Location:** Moved from hive to queen
2. **Heartbeat Destination:** Workers send to queen (not hive)
3. **Hive Role:** Stateless executor (no tracking)
4. **Worker Catalog:** NEW - mirrors model catalog pattern
5. **Shared Abstractions:** artifact-catalog eliminates duplication

---

**All corrections applied! Ready for implementation! 🚀**
