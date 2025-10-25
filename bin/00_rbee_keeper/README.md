# rbee-keeper

**Status:** ✅ IMPLEMENTED (TEAM-158, TEAM-293)  
**Purpose:** Thin HTTP client for queen-rbee  
**Binary Names:** `rbee-keeper` (CLI), `rbee-keeper-gui` (GUI)  
**LOC:** ~450 lines (CLI) + ~1,948 lines (GUI + docs)

## 🎨 AVAILABLE INTERFACES

rbee-keeper is available in **two interfaces** that share the same business logic:

### CLI (Command Line Interface)
```bash
# Traditional command-line interface
rbee-keeper infer --model llama-7b --prompt "Hello"
rbee-keeper hive start --host localhost
```

**Use when:**
- ✅ Scripting/automation
- ✅ Remote SSH sessions
- ✅ CI/CD pipelines
- ✅ Server environments

### GUI (Graphical User Interface)
```bash
# Desktop application (Tauri + React)
rbee-keeper-gui
```

**Use when:**
- ✅ Interactive exploration
- ✅ Visual feedback needed
- ✅ Desktop environment
- ✅ New users learning the system

**📖 See [README_GUI.md](./README_GUI.md) for GUI documentation**  
**🚀 See [QUICKSTART_GUI.md](./QUICKSTART_GUI.md) for 5-minute quick start**

## 🌍 Platform Support

rbee-keeper is designed to work cross-platform:

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux** | ✅ Fully Supported | Primary development platform |
| **macOS** | ⚠️ Ready (Untested) | Implementation complete, needs testing |
| **Windows** | ⚠️ Ready (Untested) | Implementation complete, SSH may require setup |

**See [CROSS_PLATFORM.md](./CROSS_PLATFORM.md) for platform-specific details**

---

## 🎯 CRITICAL ARCHITECTURE PRINCIPLE

# ⚠️ rbee-keeper is a THIN HTTP CLIENT ⚠️

**DO NOT OVER-ENGINEER THIS BINARY!**

```
User → rbee-keeper (CLI) → queen-rbee (HTTP API) → Everything else
```

## What rbee-keeper IS:
- ✅ A simple HTTP client that talks to queen-rbee
- ✅ CLI argument parser (clap)
- ✅ Queen lifecycle manager (auto-start if needed)
- ✅ SSE stream consumer (prints events to stdout)
- ✅ ~450 lines of code total

## What rbee-keeper is NOT:
- ❌ A complex application with business logic
- ❌ A daemon that runs in the background
- ❌ An SSH client (only queen-rbee uses SSH)
- ❌ An orchestrator (only queen-rbee orchestrates)
- ❌ Something that needs separate "command" modules

## Implementation:
**ALL logic is in `src/main.rs`** - No separate command files needed!

Each command is just:
1. Ensure queen is running
2. Make HTTP request
3. Display response
4. Cleanup

That's it. Don't make it more complicated.

---

## 🐝 QUEEN-RBEE LIFECYCLE MANAGEMENT

### Ephemeral Mode (Default)

**Pattern:** Start queen → Do task → Stop queen

```bash
# User runs inference
rbee infer --node gpu-0 --model llama-7b --prompt "Hello"

# Behind the scenes:
1. Check if queen-rbee is running (GET http://localhost:8080/health)
2. If NOT running:
   - Spawn queen-rbee daemon with ephemeral database
   - Wait for health check (30s timeout)
   - Mark as "started by keeper"
3. Submit task to queen (POST http://localhost:8080/v2/tasks)
4. Stream results back to user
5. When task completes:
   - IF keeper started the queen → Shutdown queen (cascading cleanup)
   - IF queen was already running → Leave it running
```

**Key Behavior:**
- ✅ **Only shutdown queens that keeper started**
- ✅ **Never shutdown pre-existing queens** (other processes may be using them)
- ✅ **Ephemeral database** (`/tmp/queen-rbee-ephemeral.db`) for one-off tasks
- ✅ **Cascading shutdown** (queen → hives → workers)

### Daemon Mode (Future)

**Pattern:** Connect to persistent queen

```bash
# Queen already running (started by systemd or manually)
rbee infer --node gpu-0 --model llama-7b --prompt "Hello"

# Behind the scenes:
1. Check if queen-rbee is running (GET http://localhost:8080/health)
2. Queen is already running → Skip startup
3. Submit task to queen (POST http://localhost:8080/v2/tasks)
4. Stream results back to user
5. When task completes → Leave queen running
```

**Use case:** Long-running orchestration, multiple concurrent keeper sessions

---

## 📋 AVAILABLE COMMANDS

### Inference

```bash
rbee infer --node gpu-0 --model llama-7b --prompt "Hello world"
  ↓
POST http://localhost:8080/v2/tasks
  { "node": "gpu-0", "model": "llama-7b", "prompt": "Hello world" }
  ↓
Queen orchestrates: find/start hive → find/spawn worker → route request
  ↓
Stream SSE tokens back to user
```

### Node Registry Management

```bash
# Add a remote hive node
rbee setup add-node gpu-node-1 --ssh-host 192.168.1.100 --ssh-user admin
  ↓
POST http://localhost:8080/v2/registry/beehives/add

# List registered nodes
rbee setup list-nodes
  ↓
GET http://localhost:8080/v2/registry/beehives/list

# Remove a node
rbee setup remove-node gpu-node-1
  ↓
POST http://localhost:8080/v2/registry/beehives/remove
```

### Worker Management

```bash
# List all workers across all hives
rbee workers list
  ↓
GET http://localhost:8080/v2/workers/list

# Check worker health on specific node
rbee workers health --node gpu-0
  ↓
GET http://localhost:8080/v2/workers/health?node=gpu-0

# Shutdown a worker
rbee workers shutdown --id worker-123
  ↓
POST http://localhost:8080/v2/workers/shutdown
```

### Installation

```bash
# Install rbee binaries to user paths
rbee install

# Install rbee binaries to system paths (requires sudo)
rbee install --system
```

### Log Viewing

```bash
# View logs from a node
rbee logs --node gpu-0

# Follow logs in real-time
rbee logs --node gpu-0 --follow
  ↓
GET http://localhost:8080/v2/logs?node=gpu-0&follow=true
```

---

## 🏗️ ARCHITECTURE

### Binary Structure

```
bin/00_rbee_keeper/
├── src/
│   ├── main.rs          (~12 LOC - entry point)
│   ├── cli.rs           (CLI parsing & routing - IN BINARY)
│   └── lib.rs           (Re-exports from crates)
└── Cargo.toml

Dependencies:
├── rbee-keeper-commands (Command implementations)
├── rbee-keeper-config   (Config loading)
├── rbee-keeper-queen-lifecycle (Queen daemon management)
└── rbee-keeper-polling  (Health polling)
```

**IMPORTANT:** CLI entry point and HTTP server are implemented DIRECTLY in the binary,
not as separate crates. This keeps the entry point logic tightly coupled to the binary.

### Lifecycle Chain

```
┌─────────────────────────────────────────────────────┐
│ User types: rbee infer --node gpu-0 --model llama   │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│ rbee-keeper (CLI Binary)                            │
│  1. Parse CLI arguments (rbee-keeper-cli)           │
│  2. ensure_queen_rbee_running()                     │
│     - Check queen health                            │
│     - If not running: spawn queen daemon            │
│     - Wait for ready                                │
│  3. POST http://localhost:8080/v2/tasks             │
│  4. Stream SSE response to stdout                   │
│  5. If keeper started queen: shutdown queen         │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP
                   ▼
┌─────────────────────────────────────────────────────┐
│ queen-rbee (Orchestrator Daemon)                    │
│  - Receives inference request                       │
│  - Finds/starts hive for node (via SSH if remote)   │
│  - Requests hive to spawn worker                    │
│  - Routes request to worker                         │
│  - Relays SSE stream back to keeper                 │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP or SSH
                   ▼
┌─────────────────────────────────────────────────────┐
│ rbee-hive (Pool Daemon)                             │
│  - Spawns llm-worker-rbee process                   │
│  - Routes inference to worker                       │
└──────────────────┬──────────────────────────────────┘
                   │ spawns
                   ▼
┌─────────────────────────────────────────────────────┐
│ llm-worker-rbee (Worker)                            │
│  - Loads model, generates tokens                    │
└─────────────────────────────────────────────────────┘
```

**Key Point:** rbee-keeper NEVER uses SSH. Only queen-rbee manages remote hives via SSH.

---

## 🚫 WHAT THIS BINARY DOES NOT DO

### ❌ NO SSH

rbee-keeper does NOT use SSH. Only queen-rbee uses SSH to manage remote hives.

```bash
# ❌ BANNED (old architecture violation)
rbee hive models download llama-7b --host gpu-0  # Direct SSH to hive

# ✅ CORRECT (via queen-rbee)
rbee infer --node gpu-0 --model llama-7b --prompt "test"  # Queen handles SSH
```

### ❌ NO Direct Hive Communication

All hive operations go through queen-rbee HTTP API.

### ❌ NO Orchestration Logic

rbee-keeper submits requests. queen-rbee makes ALL orchestration decisions.

### ❌ NO Daemon Mode

rbee-keeper is a CLI tool that exits after each command. It does NOT run as a background daemon.

---

## 🔗 DEPENDENCIES

### Internal Crates

```toml
[dependencies]
rbee-keeper-cli = { path = "../05_rbee_keeper_crates/cli" }
rbee-keeper-commands = { path = "../05_rbee_keeper_crates/commands" }
rbee-keeper-config = { path = "../05_rbee_keeper_crates/config" }
rbee-keeper-queen-lifecycle = { path = "../05_rbee_keeper_crates/queen-lifecycle" }
rbee-keeper-pool-client = { path = "../05_rbee_keeper_crates/pool-client" }
```

### External Crates

```toml
[dependencies]
anyhow = "1.0"
tokio = { version = "1", features = ["rt-multi-thread"] }
```

---

## 📊 MIGRATION STATUS

**Source:** `bin/old.rbee-keeper/` (1,252 LOC)  
**Target:** `bin/00_rbee_keeper/` (~187 LOC binary + 1,065 LOC in crates)

**Crate Decomposition:**
- `cli` - 175 LOC (CLI parsing)
- `config` - 44 LOC (Config loading)
- `queen-lifecycle` - 75 LOC (Queen daemon management)
- `pool-client` - 115 LOC (HTTP client)
- `commands` - 817 LOC (Command implementations)
- **Binary** - 12 LOC (main.rs entry point)

**Architecture Fixes:**
- ❌ Removed `ssh.rs` (14 LOC) - SSH is banned
- ❌ Removed `commands/hive.rs` (84 LOC) - Direct hive access is banned
- ✅ All operations now via queen-rbee HTTP API

---

## ✅ ACCEPTANCE CRITERIA

### Compilation

```bash
cd bin/00_rbee_keeper
cargo check
cargo clippy -- -D warnings
cargo build --release
```

### Functionality

- [ ] `rbee infer` starts ephemeral queen, runs inference, shuts down queen
- [ ] `rbee infer` with pre-existing queen leaves queen running
- [ ] `rbee setup add-node` registers node in queen registry
- [ ] `rbee workers list` queries queen worker registry
- [ ] `rbee install` copies binaries to standard paths
- [ ] All commands work without SSH access

### Architecture Compliance

- [ ] No SSH code in binary or crates
- [ ] No direct hive communication
- [ ] All operations via queen-rbee HTTP API
- [ ] Queen lifecycle properly managed (start if needed, shutdown if started by keeper)

---

## 📚 REFERENCES

### Planning Documents

- `.plan/.archive-130BC-134/TEAM_130C_RBEE_KEEPER_COMPLETE_RESPONSIBILITIES.md`
  - Lines 1-21: Core principle (thin client)
  - Lines 23-59: Queen lifecycle management (ephemeral vs daemon mode)
  - Lines 166-211: Architecture violations to avoid

- `.plan/.archive-130BC-134/TEAM_134_rbee-keeper_INVESTIGATION_REPORT.md`
  - Complete decomposition analysis
  - Crate structure and LOC breakdown

### Source Code

- `bin/old.rbee-keeper/` - Original implementation
- `bin/05_rbee_keeper_crates/` - Decomposed crates

---

**Migration Status:** 🚧 NOT STARTED  
**Priority:** HIGH  
**Estimated Effort:** 4 days (after crates are migrated)

**Next Steps:**
1. Migrate crates in `05_rbee_keeper_crates/`
2. Create `src/main.rs` entry point
3. Wire up dependencies
4. Test queen lifecycle management
5. Verify all commands work via queen-rbee API
