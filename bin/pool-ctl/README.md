# pool-ctl

**Created by:** TEAM-022  
**Binary:** `llorch-pool`  
**Status:** Active (CP1 complete, CP2-CP3 in progress)

Local pool management CLI.

## Overview

`pool-ctl` provides command-line tools for managing a pool locally. It handles model downloads, worker spawning, and pool status.

## Installation

```bash
cargo build --release -p pool-ctl
```

Binary will be at: `target/release/llorch-pool`

## Commands

### Model Management

```bash
# Show model catalog
llorch-pool models catalog

# Register a model
llorch-pool models register tinyllama \
    --name "TinyLlama 1.1B Chat" \
    --repo "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --architecture llama

# Download a model (CP3)
llorch-pool models download tinyllama

# Unregister a model
llorch-pool models unregister tinyllama
```

### Worker Management (CP3)

```bash
# Spawn a worker
llorch-pool worker spawn metal --model tinyllama --gpu 0

# List running workers
llorch-pool worker list

# Stop a worker
llorch-pool worker stop worker-metal-0
```

### Pool Status

```bash
# Show pool status
llorch-pool status
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
