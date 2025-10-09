# rbees-pool

**Created by:** TEAM-022  
**Binary:** `rbees-pool`  
**Status:** Active (CP1 complete, CP2-CP3 in progress)

Local pool management CLI.

## Overview

`rbees-pool` provides command-line tools for managing a pool locally. It handles model downloads, worker spawning, and pool status.

## Installation

```bash
cargo build --release -p rbees-pool
```

Binary will be at: `target/release/rbees-pool`

## Commands

### Model Management

```bash
# Show model catalog
rbees-pool models catalog

# Register a model
rbees-pool models register tinyllama \
    --name "TinyLlama 1.1B Chat" \
    --repo "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --architecture llama

# Download a model (CP3)
rbees-pool models download tinyllama

# Unregister a model
rbees-pool models unregister tinyllama
```

### Worker Management (CP3)

```bash
# Spawn a worker
rbees-pool worker spawn metal --model tinyllama --gpu 0

# List running workers
rbees-pool worker list

# Stop a worker
rbees-pool worker stop worker-metal-0
```

### Pool Status

```bash
# Show pool status
rbees-pool status
```

## Catalog Format

Models are tracked in `.test-models/catalog.json`:

```json
{
  "version": "1.0",
  "pool_id": "mac.home.arpa",
  "updated_at": "2025-10-09T15:00:00Z",
  "models": [
    {
      "id": "tinyllama",
      "name": "TinyLlama 1.1B Chat",
      "path": ".test-models/tinyllama",
      "format": "safetensors",
      "size_gb": 2.2,
      "architecture": "llama",
      "downloaded": true,
      "backends": ["cpu", "metal", "cuda"],
      "metadata": {
        "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
      }
    }
  ]
}
```

## Implementation Status

- ✅ CP1: Basic CLI structure, catalog commands
- 🚧 CP2: Full catalog management
- 🚧 CP3: Model downloads, worker spawning

## License

GPL-3.0-or-later
