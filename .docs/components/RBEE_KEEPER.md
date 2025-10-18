# Component: rbee-keeper (CLI Tool)

**Location:** `bin/rbee-keeper/`  
**Type:** Command-line interface  
**Language:** Rust  
**Purpose:** User-facing CLI to interact with rbee ecosystem

## Overview

`rbee-keeper` is the primary user interface for managing and interacting with the rbee orchestration system. It starts the queen-rbee orchestrator and provides commands for inference, model management, and system control.

## Responsibilities

### 1. Queen Bee Lifecycle Management

**Start Queen:**
- Spawns `queen-rbee` daemon process
- Validates queen is running and healthy
- Provides status feedback to user

**Stop Queen:**
- Sends graceful shutdown signal to queen-rbee
- Waits for clean shutdown
- Handles force-kill if needed

**Status Check:**
- Queries queen-rbee health endpoint
- Reports system status to user

### 2. User Commands

**Inference:**
```bash
rbee-keeper infer --node localhost --model "model.gguf" --prompt "Hello"
```
- Sends inference request to queen-rbee
- Streams results back to user
- Handles errors gracefully

**Model Management:**
```bash
rbee-keeper models list
rbee-keeper models download "hf:model/path"
```
- Lists available models
- Triggers model downloads
- Shows download progress

**Worker Management:**
```bash
rbee-keeper workers list
rbee-keeper workers spawn --model "model.gguf"
```
- Lists active workers
- Spawns new workers
- Monitors worker status

### 3. Configuration

**Config File:** `.llorch.toml`
- Queen-rbee connection settings
- Default model paths
- User preferences

**Environment Variables:**
- `RBEE_QUEEN_URL` - Override queen-rbee endpoint
- `RBEE_API_KEY` - Authentication (future)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ rbee-keeper (CLI)                                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Command Parser                                   │  │
│  │  - infer, models, workers, start, stop, status  │  │
│  └──────────────────────────────────────────────────┘  │
│                        │                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Queen Lifecycle Manager                          │  │
│  │  - Start queen-rbee daemon                       │  │
│  │  - Health checks                                 │  │
│  │  - Graceful shutdown                             │  │
│  └──────────────────────────────────────────────────┘  │
│                        │                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │ HTTP Client                                      │  │
│  │  - Sends requests to queen-rbee                  │  │
│  │  - Handles responses                             │  │
│  │  - Streams output                                │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │   queen-rbee     │
              │  (Orchestrator)  │
              └──────────────────┘
```

## Local vs Network Mode

### Local Mode (Default)
- Queen-rbee runs on same machine
- Uses `localhost` or `127.0.0.1`
- No SSH required
- Fast, low latency

**Detection:**
```rust
// If queen URL is localhost/127.0.0.1
if queen_url.contains("localhost") || queen_url.contains("127.0.0.1") {
    // Local mode - spawn queen directly
} else {
    // Network mode - connect to remote queen
}
```

### Network Mode (Remote)
- Queen-rbee runs on different machine
- Uses hostname or IP address
- SSH for remote management (future)
- Higher latency

**Configuration:**
```toml
[queen]
mode = "network"  # or "local"
host = "192.168.1.100"
port = 8080
```

## Key Files

- `src/main.rs` - CLI entry point
- `src/commands/infer.rs` - Inference command
- `src/commands/models.rs` - Model management
- `src/commands/workers.rs` - Worker management
- `src/commands/start.rs` - Start queen-rbee
- `src/commands/stop.rs` - Stop queen-rbee
- `src/client.rs` - HTTP client for queen-rbee API

## Dependencies

- `clap` - CLI argument parsing
- `reqwest` - HTTP client
- `tokio` - Async runtime
- `serde` - Serialization

## Current Status

**Implemented:**
- ✅ Basic CLI structure
- ✅ Inference command
- ✅ HTTP client to queen-rbee
- ✅ Local mode

**Missing:**
- ❌ Queen lifecycle management (start/stop)
- ❌ Network mode detection
- ❌ SSH support for remote queen
- ❌ Model management commands
- ❌ Worker management commands
- ❌ Configuration file support
- ❌ Progress streaming for downloads

## Related Components

- **queen-rbee** - Orchestrator that rbee-keeper controls
- **rbee-hive** - Worker pool manager (managed by queen)
- **llm-worker-rbee** - Worker processes (managed by hive)

## Future Enhancements

1. **Auto-start Queen** - Detect if queen is down, start automatically
2. **SSH Integration** - Start remote queen via SSH
3. **Multi-queen Support** - Connect to multiple queens
4. **Interactive Mode** - REPL for rapid testing
5. **Shell Completion** - Bash/zsh completion scripts

---

**Created by:** Multiple teams  
**Last Updated:** TEAM-096 | 2025-10-18  
**Status:** 🟡 PARTIAL - Core functionality exists, lifecycle management missing
