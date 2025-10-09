# llorch-ctl

**Created by:** TEAM-022  
**Binary:** `llorch`  
**Status:** Active (CP1 complete)

Orchestrator control CLI - Remote pool control via SSH.

## Overview

`llorch-ctl` provides command-line tools for controlling pools remotely via SSH. It wraps `pool-ctl` commands and executes them on remote hosts.

## Installation

```bash
cargo build --release -p llorch-ctl
```

Binary will be at: `target/release/llorch`

## Commands

### Remote Model Management

```bash
# Show catalog on remote pool
llorch pool models catalog --host mac.home.arpa

# Register model on remote pool
llorch pool models register qwen-0.5b \
    --host mac.home.arpa \
    --name "Qwen2.5 0.5B Instruct" \
    --repo "Qwen/Qwen2.5-0.5B-Instruct" \
    --architecture qwen

# Download model on remote pool (CP3)
llorch pool models download qwen-0.5b --host mac.home.arpa
```

### Remote Worker Management (CP3)

```bash
# Spawn worker on remote pool
llorch pool worker spawn metal \
    --model tinyllama \
    --host mac.home.arpa \
    --gpu 0

# List workers on remote pool
llorch pool worker list --host mac.home.arpa

# Stop worker on remote pool
llorch pool worker stop worker-metal-0 --host mac.home.arpa
```

### Git Operations

```bash
# Pull latest changes on remote pool
llorch pool git pull --host mac.home.arpa

# Show git status on remote pool
llorch pool git status --host mac.home.arpa

# Build pool-ctl on remote pool
llorch pool git build --host mac.home.arpa
```

### Pool Status

```bash
# Show status of remote pool
llorch pool status --host mac.home.arpa
```

## SSH Requirements

- SSH access to all pool hosts
- SSH keys configured (no password prompts)
- `llorch-pool` binary built on remote hosts
- Repository cloned at `~/Projects/llama-orch` on remote hosts

## Architecture

```
llorch (on blep)
    ↓ SSH
pool-ctl (on mac/workstation)
    ↓ spawn
llorch-candled (worker)
```

**Control Plane:** SSH  
**Data Plane:** HTTP (workers)

## Example Workflow

```bash
# 1. Update git on remote pool
llorch pool git pull --host mac.home.arpa

# 2. Build pool-ctl on remote
llorch pool git build --host mac.home.arpa

# 3. Register model
llorch pool models register tinyllama \
    --host mac.home.arpa \
    --name "TinyLlama 1.1B Chat" \
    --repo "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --architecture llama

# 4. Download model (CP3)
llorch pool models download tinyllama --host mac.home.arpa

# 5. Spawn worker (CP3)
llorch pool worker spawn metal \
    --model tinyllama \
    --host mac.home.arpa
```

## Implementation Status

- ✅ CP1: SSH wrapper, basic commands
- 🚧 CP2: Remote catalog management
- 🚧 CP3: Remote model downloads, worker spawning

## License

GPL-3.0-or-later
