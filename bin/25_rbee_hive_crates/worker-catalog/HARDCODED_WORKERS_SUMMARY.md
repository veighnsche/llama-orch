# Hardcoded Worker Catalog - Complete!

## Status: ✅ COMPLETE

Successfully hardcoded the 3 worker binaries into the worker catalog system!

## Problem

The hive needs to know about the 3 worker binaries (cpu, cuda, metal) so it can:
1. Build them using `lifecycle-local/build.rs`
2. Install them using `lifecycle-local/install.rs`
3. Track them in the worker catalog

## Solution

### 1. Worker Catalog (`worker-catalog/src/lib.rs`)

Added `hardcoded_workers()` function that returns the 3 worker definitions:

```rust
pub fn hardcoded_workers() -> Vec<(WorkerType, &'static str, &'static str)> {
    vec![
        (WorkerType::CpuLlm, "llm-worker-rbee-cpu", "llm-worker-rbee"),
        (WorkerType::CudaLlm, "llm-worker-rbee-cuda", "llm-worker-rbee"),
        (WorkerType::MetalLlm, "llm-worker-rbee-metal", "llm-worker-rbee"),
    ]
}
```

### 2. Worker Types (`worker-catalog/src/types.rs`)

Updated `WorkerType` with build metadata:

```rust
impl WorkerType {
    // Binary name (what gets installed)
    pub fn binary_name(&self) -> &str {
        match self {
            WorkerType::CpuLlm => "llm-worker-rbee-cpu",
            WorkerType::CudaLlm => "llm-worker-rbee-cuda",
            WorkerType::MetalLlm => "llm-worker-rbee-metal",
        }
    }

    // Crate name (what gets built)
    pub fn crate_name(&self) -> &str {
        "llm-worker-rbee"  // All workers in same crate!
    }

    // Features needed for build
    pub fn build_features(&self) -> Option<&str> {
        match self {
            WorkerType::CpuLlm => Some("cpu"),
            WorkerType::CudaLlm => Some("cuda"),
            WorkerType::MetalLlm => Some("metal"),
        }
    }
}
```

### 3. Build System (`lifecycle-local/src/build.rs`)

Added `features` field to `BuildConfig`:

```rust
pub struct BuildConfig {
    pub daemon_name: String,
    pub target: Option<String>,
    pub job_id: Option<String>,
    pub features: Option<String>,  // NEW: For feature-gated builds
}
```

Build command now supports features:

```bash
# CPU worker
cargo build --bin llm-worker-rbee-cpu --features cpu

# CUDA worker
cargo build --bin llm-worker-rbee-cuda --features cuda

# Metal worker
cargo build --bin llm-worker-rbee-metal --features metal
```

## How It Works

### Building a Worker

```rust
use lifecycle_local::{BuildConfig, build_daemon};
use worker_catalog::WorkerType;

let worker_type = WorkerType::CpuLlm;

let config = BuildConfig {
    daemon_name: worker_type.binary_name().to_string(),  // "llm-worker-rbee-cpu"
    target: None,
    job_id: Some("job-123".to_string()),
    features: worker_type.build_features().map(|s| s.to_string()),  // Some("cpu")
};

let binary_path = build_daemon(config).await?;
```

### Installing a Worker

```rust
use lifecycle_local::{InstallConfig, install_daemon};

let config = InstallConfig {
    daemon_name: "llm-worker-rbee-cpu",
    local_binary_path: Some(binary_path),  // From build step
    job_id: Some("job-123".to_string()),
};

install_daemon(config).await?;
```

### Catalog Integration

```rust
use worker_catalog::{WorkerCatalog, WorkerBinary, WorkerType, Platform};

let catalog = WorkerCatalog::new()?;

// Get hardcoded workers
for (worker_type, binary_name, crate_name) in WorkerCatalog::hardcoded_workers() {
    println!("Worker: {} (crate: {})", binary_name, crate_name);
    println!("Features: {:?}", worker_type.build_features());
}

// Find a specific worker
let worker = catalog.find_by_type_and_platform(
    WorkerType::CpuLlm,
    Platform::Linux
);
```

## The 3 Workers

| Worker Type | Binary Name | Crate | Features | Platform |
|-------------|-------------|-------|----------|----------|
| CpuLlm | `llm-worker-rbee-cpu` | `llm-worker-rbee` | `cpu` | All |
| CudaLlm | `llm-worker-rbee-cuda` | `llm-worker-rbee` | `cuda` | Linux |
| MetalLlm | `llm-worker-rbee-metal` | `llm-worker-rbee` | `metal` | macOS |

## Build Commands

```bash
# CPU worker (works everywhere)
cargo build --bin llm-worker-rbee-cpu --features cpu

# CUDA worker (requires CUDA toolkit)
cargo build --bin llm-worker-rbee-cuda --features cuda

# Metal worker (requires macOS)
cargo build --bin llm-worker-rbee-metal --features metal
```

## Installation Paths

After installation, workers are located at:
- `~/.local/bin/llm-worker-rbee-cpu`
- `~/.local/bin/llm-worker-rbee-cuda`
- `~/.local/bin/llm-worker-rbee-metal`

## Files Changed

1. **worker-catalog/src/lib.rs**
   - Added `hardcoded_workers()` function

2. **worker-catalog/src/types.rs**
   - Updated `WorkerType::binary_name()` to match actual binary names
   - Added `WorkerType::crate_name()` for build system
   - Added `WorkerType::build_features()` for feature flags

3. **lifecycle-local/src/build.rs**
   - Added `features` field to `BuildConfig`
   - Updated build command to support `--features` flag

4. **lifecycle-local/src/install.rs**
   - Updated `BuildConfig` initialization to include `features: None`

## Next Steps

The hive can now:
1. ✅ Query `WorkerCatalog::hardcoded_workers()` to get available workers
2. ✅ Build workers using `build_daemon()` with correct features
3. ✅ Install workers using `install_daemon()`
4. ✅ Track installed workers in the catalog

## Example: Full Workflow

```rust
use worker_catalog::{WorkerCatalog, WorkerType, Platform};
use lifecycle_local::{BuildConfig, InstallConfig, build_daemon, install_daemon};

// Step 1: Get worker info from catalog
let worker_type = WorkerType::CpuLlm;
let binary_name = worker_type.binary_name();
let features = worker_type.build_features();

// Step 2: Build the worker
let build_config = BuildConfig {
    daemon_name: binary_name.to_string(),
    target: None,
    job_id: Some("job-123".to_string()),
    features: features.map(|s| s.to_string()),
};
let binary_path = build_daemon(build_config).await?;

// Step 3: Install the worker
let install_config = InstallConfig {
    daemon_name: binary_name,
    local_binary_path: Some(binary_path),
    job_id: Some("job-123".to_string()),
};
install_daemon(install_config).await?;

// Step 4: Add to catalog
let catalog = WorkerCatalog::new()?;
let worker = WorkerBinary::new(
    format!("{}-v0.1.0-linux", binary_name),
    worker_type,
    Platform::Linux,
    PathBuf::from(format!("~/.local/bin/{}", binary_name)),
    1024 * 1024,  // Size
    "0.1.0".to_string(),
);
catalog.add(worker)?;
```

Perfect! The 3 workers are now hardcoded and ready to be built/installed by the hive! 🎉
