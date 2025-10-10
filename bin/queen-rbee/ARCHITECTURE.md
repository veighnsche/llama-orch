# queen-rbee Architecture

**Created by:** TEAM-043  
**Date:** 2025-10-10

---

## Overview

queen-rbee is the orchestrator daemon that manages both rbee-hive nodes and active workers through a dual registry system.

---

## Dual Registry System

### 1. Beehive Registry (Persistent - SQLite)

**Purpose:** Track remote rbee-hive pool manager nodes

**Storage:** SQLite database at `~/.rbee/beehives.db`

**Module:** `src/beehive_registry.rs`

**Schema:**
```sql
CREATE TABLE beehives (
    node_name TEXT PRIMARY KEY,
    ssh_host TEXT NOT NULL,
    ssh_port INTEGER NOT NULL DEFAULT 22,
    ssh_user TEXT NOT NULL,
    ssh_key_path TEXT,
    git_repo_url TEXT NOT NULL,
    git_branch TEXT NOT NULL,
    install_path TEXT NOT NULL,
    last_connected_unix INTEGER,
    status TEXT NOT NULL DEFAULT 'unknown'
)
```

**Operations:**
- `add_node()` - Register a new rbee-hive node
- `get_node()` - Retrieve node details
- `list_nodes()` - List all registered nodes
- `remove_node()` - Unregister a node
- `update_status()` - Update node status

**Lifecycle:** Persistent across restarts

---

### 2. Worker Registry (Ephemeral - In-Memory)

**Purpose:** Track active workers spawned by rbee-hive nodes

**Storage:** In-memory `HashMap` with `RwLock`

**Module:** `src/worker_registry.rs`

**Data Structure:**
```rust
pub struct WorkerInfo {
    pub id: String,
    pub url: String,
    pub model_ref: String,
    pub backend: String,
    pub device: u32,
    pub state: WorkerState,  // Loading, Idle, Busy
    pub slots_total: u32,
    pub slots_available: u32,
    pub vram_bytes: Option<u64>,
}
```

**Operations:**
- `register()` - Register a new worker
- `update_state()` - Update worker state
- `get()` - Get worker by ID
- `list()` - List all workers
- `remove()` - Remove worker
- `count()` - Count active workers

**Lifecycle:** Cleared on restart (ephemeral)

---

## HTTP API Endpoints

### Beehive Registry Endpoints

- `POST /v2/registry/beehives/add` - Add rbee-hive node
- `GET /v2/registry/beehives/list` - List rbee-hive nodes
- `POST /v2/registry/beehives/remove` - Remove rbee-hive node

### Worker Registry Endpoints (TODO)

- `POST /v1/workers/register` - Worker ready callback
- `GET /v1/workers/list` - List active workers
- `POST /v1/workers/update` - Update worker state
- `DELETE /v1/workers/{id}` - Remove worker

---

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│                       queen-rbee                             │
│                    (Orchestrator Daemon)                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────┐    ┌──────────────────────┐      │
│  │  Beehive Registry    │    │   Worker Registry    │      │
│  │  (SQLite)            │    │   (In-Memory)        │      │
│  ├──────────────────────┤    ├──────────────────────┤      │
│  │ • node_name          │    │ • worker_id          │      │
│  │ • ssh_host           │    │ • url                │      │
│  │ • ssh_user           │    │ • model_ref          │      │
│  │ • git_repo           │    │ • state              │      │
│  │ • install_path       │    │ • slots              │      │
│  │ • status             │    │ • vram_bytes         │      │
│  └──────────────────────┘    └──────────────────────┘      │
│           ↓                            ↑                     │
│  Persistent across                Cleared on                │
│  restarts                         restart                   │
└─────────────────────────────────────────────────────────────┘
           ↓                            ↑
    SSH commands                 HTTP callbacks
           ↓                            ↑
┌─────────────────────────────────────────────────────────────┐
│                      rbee-hive                               │
│                   (Pool Manager)                             │
│                   on remote nodes                            │
└─────────────────────────────────────────────────────────────┘
           ↓
    Spawns workers
           ↓
┌─────────────────────────────────────────────────────────────┐
│                   llm-worker-rbee                            │
│                   (Inference Worker)                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Workflow Example

### 1. Setup Phase (Beehive Registry)

```bash
# User adds a remote node
rbee-keeper setup add-node \
  --name mac \
  --ssh-host mac.home.arpa \
  --ssh-user vince \
  --ssh-key ~/.ssh/id_ed25519 \
  --git-repo https://github.com/user/llama-orch.git \
  --install-path ~/rbee

# queen-rbee:
# 1. Validates SSH connection
# 2. Saves to beehive_registry (SQLite)
# 3. Returns success
```

### 2. Runtime Phase (Worker Registry)

```bash
# User requests inference
rbee-keeper infer \
  --node mac \
  --model tinyllama \
  --prompt "hello"

# queen-rbee:
# 1. Looks up "mac" in beehive_registry
# 2. SSH to mac, spawns rbee-hive
# 3. rbee-hive spawns worker
# 4. Worker calls back: POST /v1/workers/register
# 5. queen-rbee adds to worker_registry (in-memory)
# 6. Returns worker URL to rbee-keeper
```

---

## Why Dual Registry?

### Beehive Registry (Persistent)
- **Nodes are infrastructure** - Should survive restarts
- **SSH credentials** - Need to be stored securely
- **Configuration** - Git repo, install paths, etc.
- **Status tracking** - Last connected, reachability

### Worker Registry (Ephemeral)
- **Workers are transient** - Spawned on demand
- **State changes frequently** - Loading → Idle → Busy
- **No persistence needed** - Re-discovered on startup
- **Performance** - Fast in-memory lookups

---

## File Structure

```
bin/queen-rbee/
├── src/
│   ├── main.rs                  # Entry point, initializes both registries
│   ├── beehive_registry.rs      # SQLite-backed beehive registry
│   ├── worker_registry.rs       # In-memory worker registry
│   ├── ssh.rs                   # SSH connection validation
│   └── http.rs                  # HTTP server + routes
├── Cargo.toml
└── ARCHITECTURE.md              # This file
```

---

## Future Enhancements

### Beehive Registry
- [ ] Encryption for SSH keys
- [ ] Node health monitoring
- [ ] Automatic SSH key rotation
- [ ] Multi-user support

### Worker Registry
- [ ] Worker heartbeat monitoring
- [ ] Automatic worker cleanup (stale workers)
- [ ] Load balancing across workers
- [ ] Worker metrics aggregation

---

## Testing

### Beehive Registry Tests
```bash
cd bin/queen-rbee
cargo test beehive_registry
```

### Worker Registry Tests
```bash
cd bin/queen-rbee
cargo test worker_registry
```

### Integration Tests
```bash
cd test-harness/bdd
cargo run --bin bdd-runner -- --tags @setup
```

---

**Status:** ✅ Dual registry system implemented and compiling
