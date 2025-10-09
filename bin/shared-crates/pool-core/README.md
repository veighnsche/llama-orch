# pool-core

**Created by:** TEAM-022  
**Status:** Active

Shared types and logic for pool management.

## Overview

`pool-core` provides the foundational types and utilities used by both `pool-ctl` (local pool management CLI) and `pool-managerd` (pool manager daemon, future).

## Features

- **Model Catalog**: Track available models per pool with metadata
- **Worker Types**: Type-safe worker and backend representations
- **Error Handling**: Comprehensive error types for pool operations

## Usage

```rust
use pool_core::catalog::{ModelCatalog, ModelEntry};
use pool_core::worker::{Backend, WorkerInfo};

// Create a new catalog
let mut catalog = ModelCatalog::new("my-pool".to_string());

// Add a model
let entry = ModelEntry {
    id: "tinyllama".to_string(),
    name: "TinyLlama 1.1B Chat".to_string(),
    // ... other fields
};
catalog.add_model(entry)?;

// Save to disk
catalog.save(Path::new(".test-models/catalog.json"))?;

// Load from disk
let loaded = ModelCatalog::load(Path::new(".test-models/catalog.json"))?;
```

## Types

### ModelCatalog

Manages the model catalog for a pool.

**Methods:**
- `new(pool_id)` - Create empty catalog
- `load(path)` - Load from JSON file
- `save(path)` - Save to JSON file
- `add_model(entry)` - Add model entry
- `remove_model(id)` - Remove model entry
- `find_model(id)` - Find model by ID

### Backend

Enum representing worker backends:
- `Cpu` - CPU backend
- `Metal` - Apple Metal backend
- `Cuda` - NVIDIA CUDA backend

### WorkerInfo

Worker metadata including ID, backend, model, GPU, port, and PID.

## Testing

```bash
cargo test -p pool-core
```

## License

GPL-3.0-or-later
