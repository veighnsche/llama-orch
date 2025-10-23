# rbee-hive-worker-catalog

**TEAM-273: Worker catalog for managing worker binaries**

## Purpose

Manages worker binaries for different platforms and worker types.
Built on top of `artifact-catalog` for consistency with model-catalog.

## Storage

Workers are stored in:
- **Linux/Mac:** `~/.cache/rbee/workers/`
- **Windows:** `%LOCALAPPDATA%\rbee\workers\`

## Worker Types

- **CpuLlm** - `cpu-llm-worker-rbee` - CPU-based LLM inference
- **CudaLlm** - `cuda-llm-worker-rbee` - CUDA-based LLM inference
- **MetalLlm** - `metal-llm-worker-rbee` - Metal-based LLM inference (macOS)

## Platforms

- **Linux** - Linux binaries
- **MacOS** - macOS binaries
- **Windows** - Windows binaries (.exe)

## Storage Layout

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

## Usage

```rust
use rbee_hive_worker_catalog::{WorkerCatalog, WorkerBinary, WorkerType, Platform};
use rbee_hive_artifact_catalog::ArtifactCatalog;

// Create catalog
let catalog = WorkerCatalog::new()?;

// Add a worker
let worker = WorkerBinary::new(
    "cpu-llm-worker-rbee-v0.1.0-linux".to_string(),
    WorkerType::CpuLlm,
    Platform::Linux,
    PathBuf::from("/path/to/cpu-llm-worker-rbee"),
    1_024_000, // 1 MB
    "0.1.0".to_string(),
);
catalog.add(worker)?;

// Find worker by type and platform
let worker = catalog.find_by_type_and_platform(
    WorkerType::CpuLlm,
    Platform::current(),
);

// List all workers
let workers = catalog.list();

// Get a specific worker
let worker = catalog.get("cpu-llm-worker-rbee-v0.1.0-linux")?;

// Remove a worker
catalog.remove("cpu-llm-worker-rbee-v0.1.0-linux")?;
```

## Architecture

Uses `artifact-catalog` as base:
- `WorkerBinary` implements `Artifact` trait
- `WorkerCatalog` delegates to `FilesystemCatalog<WorkerBinary>`
- Mirrors `model-catalog` pattern exactly

## Testing

```bash
cargo test --package rbee-hive-worker-catalog
```

## Created By

**TEAM-273** - Worker catalog using artifact-catalog abstraction
