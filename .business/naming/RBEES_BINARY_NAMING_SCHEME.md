# rbees Binary Naming Scheme

**Brand:** rbees  
**Date:** 2025-10-09  
**Status:** Proposed

---

## Executive Summary

With **rbees** as the brand, all binaries should follow a consistent naming pattern that:
1. Uses the `rbees` prefix for brand consistency
2. Keeps names short and CLI-friendly
3. Distinguishes daemons from CLIs
4. Is memorable and easy to type
5. **No pool daemon** - pool management is CLI-only (SSH-based)

---

## Current State (Inconsistent)

**Current names from specs:**
- `orchestratord` - Orchestrator daemon
- `llorch-ctl` (command: `llorch`) - Orchestrator CLI
- ~~`pool-managerd`~~ - **DOES NOT EXIST** (never built, see ARCHITECTURE_DECISION_NO_POOL_DAEMON.md)
- `pool-ctl` (command: `llorch-pool`) - Pool CLI
- `llorch-candled` - Worker daemon
- `bdd-runner` - Test runner

**Problems:**
- ❌ No brand consistency (llorch, orchestrator, pool)
- ❌ Mixed naming patterns (llorch-X, X-ctl, Xd)
- ❌ "llorch" is not the brand anymore
- ❌ Confusing relationship between binaries

---

## Proposed Naming Scheme

### Pattern: `rbees-<component>d` for daemons, `rbees` or `rbees-<scope>` for CLIs

**Core Principle:** All binaries start with `rbees` prefix for brand consistency.

**Architecture:** Only 2 daemons needed (orchestrator + workers). Pool management is CLI-only.

---

## The 4 Core Binaries

### 1. Orchestrator Daemon
**Binary:** `rbees-orcd`  
**Command:** `rbees-orcd`  
**Port:** 8080  
**Purpose:** Scheduling, admission, job queue, worker registry  
**Replaces:** `orchestratord`

**Alternative (more explicit):** `rbees-orchestratord`

---

### 2. Orchestrator CLI
**Binary:** `rbees-ctl` (or just `rbees`)  
**Command:** `rbees`  
**Purpose:** Control orchestrator, command pools via SSH, submit jobs  
**Replaces:** `llorch-ctl` (command: `llorch`)

**Usage:**
```bash
rbees jobs submit --model qwen --prompt "hello"
rbees jobs list
rbees pool status --host mac
rbees pool worker spawn --host mac --model qwen
```

---

### 3. Pool CLI
**Binary:** `rbees-pool`  
**Command:** `rbees-pool`  
**Purpose:** Local pool operations, model downloads, worker spawn  
**Replaces:** `pool-ctl` (command: `llorch-pool`)

**Note:** This is a CLI tool, NOT a daemon. It's called via SSH by `rbees` or run locally.

**Usage:**
```bash
rbees-pool models download tinyllama
rbees-pool worker spawn metal --model tinyllama
rbees-pool git pull
rbees-pool worker list
rbees-pool worker stop worker-123
```

---

### 4. Worker Daemon
**Binary:** `rbees-workerd`  
**Command:** `rbees-workerd`  
**Port:** 8001-8999  
**Purpose:** Inference execution (HTTP server)  
**Replaces:** `llorch-candled`

**Note:** This is the NVIDIA CUDA worker. Future workers:
- `rbees-workerd-metal` (Apple ARM)
- `rbees-workerd-cpu` (CPU fallback)

**Alternative (simpler):** `rbees-worker`

---

## Complete Naming Table

| Component | Current Name | Proposed Name | Command | Port | Type |
|-----------|--------------|---------------|---------|------|------|
| Orchestrator Daemon | `orchestratord` | `rbees-orcd` | `rbees-orcd` | 8080 | Daemon |
| Orchestrator CLI | `llorch-ctl` (`llorch`) | `rbees-ctl` | `rbees` | - | CLI |
| Pool CLI | `pool-ctl` (`llorch-pool`) | `rbees-pool` | `rbees-pool` | - | CLI |
| Worker Daemon | `llorch-candled` | `rbees-workerd` | `rbees-workerd` | 8001+ | Daemon |

---

## Architecture: Only 2 Daemons Needed

**From ARCHITECTURE_DECISION_NO_POOL_DAEMON.md:**

> pool-managerd (daemon) is NOT needed.
> The pool manager functionality is fully provided by `pool-ctl` CLI.

**Why Pool Manager Doesn't Need to Be a Daemon:**
- Pool management is **control operations**, not **data plane operations**
- Control operations (download models, spawn workers, stop workers) are CLI commands, not long-running services
- Workers are already HTTP daemons (keep model in VRAM, accept inference requests)
- Pool operations are on-demand (run once, exit)

**The Correct Architecture:**
```
Control Plane (SSH):
  rbees (CLI) → SSH → rbees-pool (CLI) → spawns → rbees-workerd (daemon)

Data Plane (HTTP):
  Client → rbees-orcd (daemon) → HTTP → rbees-workerd (daemon)
```

---

## Command Examples

### Orchestrator Operations (via `rbees`)

```bash
# Daemon control (M1+)
rbees orchestrator start
rbees orchestrator stop
rbees orchestrator status

# Job operations (M1+)
rbees jobs submit --model qwen --prompt "hello"
rbees jobs list
rbees jobs cancel job-123

# Pool operations (remote via SSH)
rbees pool status --host mac
rbees pool worker spawn --host mac --model qwen
rbees pool models download tinyllama --host mac
```

### Pool Operations (via `rbees-pool`)

**Note:** This is a CLI tool, NOT a daemon. Called locally or via SSH.

```bash
# Local operations
rbees-pool models download tinyllama
rbees-pool models list
rbees-pool git pull

# Worker operations
rbees-pool worker spawn metal --model tinyllama
rbees-pool worker list
rbees-pool worker stop worker-123
```

---

## Directory Structure

```
bin/
├── rbees-orcd/                    # Orchestrator daemon (M1+)
│   ├── src/main.rs
│   └── Cargo.toml
│
├── rbees-ctl/                     # Orchestrator CLI (M0+)
│   ├── src/main.rs
│   └── Cargo.toml
│
├── rbees-pool/                    # Pool CLI (M0+)
│   ├── src/main.rs
│   └── Cargo.toml
│
└── rbees-workerd/                 # Worker daemon (M0+)
    ├── src/main.rs
    └── Cargo.toml
```

---

## Shared Crates

**Pattern:** `rbees-<component>-core`

```
libs/
├── rbees-orchestrator-core/       # Shared orchestrator logic
│   └── Cargo.toml
│
└── rbees-pool-core/               # Shared pool CLI logic
    └── Cargo.toml
```

**Note:** `rbees-pool-core` is shared between `rbees-ctl` (for SSH calls) and `rbees-pool` (for local execution).

---

## Installation & Usage

### System Installation

```bash
# Install all binaries
cargo install --path bin/rbees-orcd
cargo install --path bin/rbees-ctl
cargo install --path bin/rbees-pool
cargo install --path bin/rbees-workerd

# Binaries available as:
rbees-orcd
rbees
rbees-pool
rbees-workerd
```

### User Experience

```bash
# Main CLI (short and sweet)
rbees jobs submit --model qwen --prompt "hello"

# Pool CLI (scoped, called via SSH or locally)
rbees-pool models download tinyllama

# Daemons (explicit)
rbees-orcd --config /etc/rbees/orchestrator.toml
rbees-workerd --model qwen --gpu 0 --port 8001
```

---

## Systemd Service Names

```ini
# /etc/systemd/system/rbees-orcd.service
[Unit]
Description=rbees Orchestrator Daemon
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/rbees-orcd --config /etc/rbees/orchestrator.toml
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/rbees-workerd@.service
[Unit]
Description=rbees Worker Daemon (%i)
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/rbees-workerd --worker-id %i --config /etc/rbees/workers/%i.toml
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

**Commands:**
```bash
sudo systemctl start rbees-orcd
sudo systemctl start rbees-workerd@worker-metal-0
sudo systemctl status rbees-orcd
```

**Note:** No pool daemon service - `rbees-pool` is a CLI tool called on-demand.

---

## Migration Path

### Phase 1: Rename Binaries

1. Rename directories:
   - `orchestratord/` → `rbees-orcd/`
   - `llorch-ctl/` → `rbees-ctl/`
   - `pool-ctl/` → `rbees-pool/`
   - `llorch-candled/` → `rbees-workerd/`

2. Update Cargo.toml names:
   ```toml
   [package]
   name = "rbees-orcd"
   ```

3. Update binary names in Cargo.toml:
   ```toml
   [[bin]]
   name = "rbees-orcd"
   path = "src/main.rs"
   ```

**Note:** No `pool-managerd` directory exists - it was never built (see ARCHITECTURE_DECISION_NO_POOL_DAEMON.md)

### Phase 2: Update References

1. Update all documentation
2. Update all scripts
3. Update all specs
4. Update all tests
5. Update CI/CD

### Phase 3: Update Shared Crates

1. Rename:
   - `orchestrator-core` → `rbees-orchestrator-core`
   - `pool-core` → `rbees-pool-core`

2. Update dependencies in all Cargo.toml files

---

## Final Recommendation

### Recommended Names (Hybrid Approach)

**Daemons (HTTP servers):**
- `rbees-orcd` - Orchestrator daemon
- `rbees-workerd` - Worker daemon

**CLIs (SSH/local tools):**
- `rbees` - Main CLI (orchestrator control)
- `rbees-pool` - Pool CLI (local operations, called via SSH)

**Shared Crates:**
- `rbees-orchestrator-core`
- `rbees-pool-core`

**Rationale:**
- ✅ Brand consistency (all start with `rbees`)
- ✅ Short main CLI (`rbees` is 5 letters)
- ✅ Clear scoping (`rbees-pool` for pool operations)
- ✅ Unix daemon convention (`-d` suffix for daemons only)
- ✅ Easy to type and remember
- ✅ Professional and modern
- ✅ No unnecessary daemons (pool is CLI-only)

---

## Summary

**Current (inconsistent):**
- `orchestratord`, `llorch-ctl` (`llorch`), `pool-ctl` (`llorch-pool`), `llorch-candled`

**Proposed (consistent):**
- `rbees-orcd`, `rbees`, `rbees-pool`, `rbees-workerd`

**Key Changes:**
1. All binaries use `rbees` prefix
2. Main CLI is just `rbees` (short and memorable)
3. Daemons use `-d` suffix (Unix convention)
4. Pool CLI is `rbees-pool` (clear scoping, SSH-callable)
5. Worker daemon is `rbees-workerd` (clear purpose)
6. **No pool daemon** - pool management is CLI-only

**Benefits:**
- ✅ Brand consistency across all binaries
- ✅ Easy to discover (`rbees<tab>` shows all)
- ✅ Professional naming convention
- ✅ Short main command (`rbees`)
- ✅ Clear daemon vs CLI distinction
- ✅ Simpler architecture (only 2 daemons needed)

---

## Architecture Summary

**2 Daemons (HTTP servers):**
1. `rbees-orcd` - Orchestrator daemon (M1+)
2. `rbees-workerd` - Worker daemon (M0+)

**2 CLIs (SSH/local tools):**
3. `rbees` - Orchestrator CLI (M0+)
4. `rbees-pool` - Pool CLI (M0+)

**2 Shared Crates:**
5. `rbees-orchestrator-core` - Shared orchestrator logic
6. `rbees-pool-core` - Shared pool logic

**HARD RULES:**
- ✅ Only 2 daemons needed (orchestrator + workers)
- ✅ Pool management is CLI-only (no daemon)
- ✅ Control plane uses SSH (rbees → rbees-pool via SSH)
- ✅ Data plane uses HTTP (orchestrator → workers)
- ✅ Shared crates contain common logic
- ❌ CTL NEVER starts REPL or conversation
- ❌ Agentic API is HTTP-based, not CLI-based

**Current bash scripts map to:**
- `llorch-remote` → `rbees` (orchestrator CLI)
- `llorch-models` → `rbees-pool` (pool CLI)
- `llorch-git` → `rbees-pool` (pool CLI)

**Implementation order:**
1. M0: rbees-pool-core + rbees-pool (local operations) ✅
2. M0: rbees-orchestrator-core + rbees (SSH to pools) ✅
3. M1: rbees-orcd (orchestrator daemon) 🔜

---

**Status:** Proposed  
**Next Steps:** Review and approve, then execute migration

---

**rbees: Your distributed swarm, consistently named.** 🐝
