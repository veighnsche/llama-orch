# pool-ctl SPEC — Pool Manager Control CLI

**Status**: Draft  
**Version**: 0.1.0  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)  
**Binary Name**: `pool-ctl` (produces `llorch-pool` command)

---

## Table of Contents

### 1. Executive Summary
- [1.1 Purpose](#11-purpose)
- [1.2 Architecture Role](#12-architecture-role)
- [1.3 Design Principles](#13-design-principles)

### 2. Requirements
- [2.1 Functional Requirements](#21-functional-requirements)
- [2.2 Non-Functional Requirements](#22-non-functional-requirements)

### 3. Commands
- [3.1 Model Management](#31-model-management)
- [3.2 Git Operations](#32-git-operations)
- [3.3 Worker Lifecycle](#33-worker-lifecycle)
- [3.4 Development Tools](#34-development-tools)

### 4. Implementation
- [4.1 Crate Structure](#41-crate-structure)
- [4.2 Dependencies](#42-dependencies)

---

## 1. Executive Summary

### 1.1 Purpose

`pool-ctl` (command: `llorch-pool`) is the pool manager control CLI. It runs locally on pool manager hosts (mac.home.arpa, workstation.home.arpa, blep.home.arpa) and provides:

1. **Model Provisioning**: Download models from HuggingFace
2. **Git Operations**: Manage repo and submodules
3. **Worker Lifecycle**: Spawn/stop workers locally
4. **Development Tools**: Setup, doctor, environment checks

**This is NOT the pool-managerd daemon.** This is a CLI tool that:
- **M0**: Executes operations directly (shell commands)
- **M1+**: Can also call pool-managerd HTTP API (if daemon running)

### 1.2 Architecture Role

```
Pool Manager Host (mac.home.arpa)
├── llorch-pool (this CLI)
│   ├── models download tinyllama    → hf download ...
│   ├── git pull                     → git pull && git submodule update
│   ├── worker spawn metal           → llorch-candled --backend metal ...
│   └── dev setup                    → Install deps, configure env
│
└── pool-managerd (daemon, M1+)
    └── HTTP API :9200
        ├── POST /workers/spawn
        ├── GET /status
        └── POST /models/download (optional)
```

**Called by:**
- Local operators on pool host: `llorch-pool models download tinyllama`
- Remote orchestrator via SSH: `ssh mac "llorch-pool models download tinyllama"`
- Orchestrator CLI: `llorch pool models download tinyllama --host mac`

**Replaces:**
- `scripts/llorch-models` (bash)
- `scripts/llorch-git` (bash)

### 1.3 Design Principles

**POOL-001**: This CLI runs locally on pool manager hosts  
**POOL-002**: This CLI is a dumb executor (no intelligent decisions)  
**POOL-003**: This CLI can work standalone (M0) or with daemon (M1+)  
**POOL-004**: This CLI MUST use modern `hf` CLI (not deprecated `huggingface-cli`)  
**POOL-005**: This CLI MUST document all HuggingFace repos (no guessing)  
**POOL-006**: This CLI MUST handle git submodules robustly  
**POOL-007**: This CLI MUST provide rich output with progress indicators

---

## 2. Requirements

### 2.1 Functional Requirements

#### [POOL-F-001] Model Management
The CLI MUST provide model provisioning:
- **MUST**: Download models using `hf` CLI
- **MUST**: List downloaded models
- **MUST**: Show model info (size, repo, files)
- **MUST**: Verify model integrity
- **MUST**: Delete models
- **MUST**: Show disk usage
- **MUST**: Display catalog with verified HuggingFace repos
- **MUST**: Handle conversion (PyTorch → GGUF) for gpt2, granite
- **MUST**: Compute SHA256 checksums after download

#### [POOL-F-002] Git Operations
The CLI MUST provide git operations:
- **MUST**: Show status (repo + submodules)
- **MUST**: Pull latest changes
- **MUST**: Sync to origin/main (hard reset)
- **MUST**: List submodules with status
- **MUST**: Switch submodule branches
- **MUST**: Update submodules (single or all)
- **MUST**: Reset submodules to committed versions
- **SHOULD**: Clean untracked files

#### [POOL-F-003] Worker Lifecycle
The CLI MUST provide worker management:
- **MUST**: Spawn worker locally (llorch-candled)
- **MUST**: Stop worker by ID
- **MUST**: List running workers
- **SHOULD**: Show worker logs
- **MAY**: Call pool-managerd HTTP API if daemon running

#### [POOL-F-004] Development Tools
The CLI MUST provide dev utilities:
- **MUST**: Environment check (doctor)
- **MUST**: Setup wizard
- **MUST**: Dependency verification
- **SHOULD**: Configuration validation

### 2.2 Non-Functional Requirements

#### [POOL-NF-001] Performance
- Command startup MUST be < 100ms
- Model catalog loading MUST be < 50ms
- Progress indicators MUST update every 100ms

#### [POOL-NF-002] Usability
- Help text MUST be comprehensive
- Error messages MUST be actionable
- Output MUST be colored (unless --no-color)

#### [POOL-NF-003] Reliability
- The CLI MUST validate inputs before execution
- The CLI MUST handle network failures gracefully
- The CLI MUST provide clear error messages

---

## 3. Commands

### 3.1 Model Management

#### [POOL-CMD-M-001] Catalog
```bash
llorch-pool models catalog
```
**MUST**:
- Display all 10 models with metadata
- Show HuggingFace repo URLs (verified)
- Show expected sizes
- Show download status (downloaded/not downloaded)

#### [POOL-CMD-M-002] List
```bash
llorch-pool models list
```
**MUST**:
- List all downloaded models in `.test-models/`
- Show file sizes
- Show total disk usage

#### [POOL-CMD-M-003] Download
```bash
llorch-pool models download <MODEL> [--force]
```
**MUST**:
- Verify `hf` command exists
- Download using: `hf download <repo> <file> --local-dir <dir>`
- Show progress indicator
- Handle conversion if needed (gpt2, granite)
- Compute SHA256 checksum
- Skip if exists unless --force

**Models (verified HuggingFace repos):**
- `tinyllama` - TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
- `qwen` - Qwen/Qwen2.5-0.5B-Instruct-GGUF
- `qwen-fp16` - Qwen/Qwen2.5-0.5B-Instruct-GGUF
- `phi3` - microsoft/Phi-3-mini-4k-instruct-gguf
- `llama3` - QuantFactory/Meta-Llama-3-8B-Instruct-GGUF
- `llama32` - tensorblock/Komodo-Llama-3.2-3B-v2-fp16-GGUF
- `llama2` - TheBloke/Llama-2-7B-GGUF
- `mistral` - TheBloke/Mistral-7B-Instruct-v0.2-GGUF
- `gpt2` - openai-community/gpt2 (requires conversion)
- `granite` - ibm-granite/granite-4.0-micro (requires conversion)

#### [POOL-CMD-M-004] Info
```bash
llorch-pool models info <MODEL>
```
**MUST**:
- Show catalog metadata
- Show local files if downloaded
- Show disk usage
- Show config.json if present

#### [POOL-CMD-M-005] Verify
```bash
llorch-pool models verify <MODEL>
```
**MUST**:
- Check if model files exist
- Verify file sizes > 0
- Report status for each file

#### [POOL-CMD-M-006] Delete
```bash
llorch-pool models delete <MODEL> [--force]
```
**MUST**:
- Show disk usage before deletion
- Prompt for confirmation unless --force
- Remove model directory

### 3.2 Git Operations

#### [POOL-CMD-G-001] Status
```bash
llorch-pool git status
```
**MUST**:
- Show current branch and commit
- Show uncommitted changes
- List all submodules with status
- Show Rust toolchain version

#### [POOL-CMD-G-002] Pull
```bash
llorch-pool git pull
```
**MUST**:
- Pull latest changes from origin
- Update all submodules recursively
- Show summary of changes

#### [POOL-CMD-G-003] Sync
```bash
llorch-pool git sync [--force]
```
**MUST**:
- Hard reset to origin/main
- Update all submodules to committed versions
- Prompt for confirmation unless --force

#### [POOL-CMD-G-004] Submodules
```bash
llorch-pool git submodules list
llorch-pool git submodules status
llorch-pool git submodules update [--all | --name NAME]
llorch-pool git submodules reset --name NAME
llorch-pool git submodules branch --name NAME [--branch BRANCH]
```
**MUST**:
- List all submodules: reference/candle, reference/candle-vllm, reference/llama.cpp, reference/mistral.rs
- Show current branch for each
- Support switching branches (e.g., candle metal-fixes)
- Support updating single or all submodules
- Support resetting to committed version

### 3.3 Worker Lifecycle

#### [POOL-CMD-W-001] Spawn
```bash
llorch-pool worker spawn <BACKEND> [--model MODEL] [--gpu ID] [--port PORT]
```
**MUST**:
- Validate backend (cpu, cuda, metal)
- Validate model exists locally
- Spawn llorch-candled process
- Show worker PID and port
- Wait for ready callback (optional)

**Example:**
```bash
llorch-pool worker spawn metal --model tinyllama --gpu 0 --port 8001
```

**Spawns:**
```bash
llorch-candled \
  --worker-id worker-metal-0 \
  --model .test-models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --gpu 0 \
  --port 8001 \
  --callback-url http://localhost:9200/workers/ready
```

#### [POOL-CMD-W-002] Stop
```bash
llorch-pool worker stop <WORKER_ID>
```
**MUST**:
- Find worker process by ID
- Send graceful shutdown signal
- Wait for process exit
- Report success/failure

#### [POOL-CMD-W-003] List
```bash
llorch-pool worker list
```
**MUST**:
- List all running llorch-candled processes
- Show worker ID, backend, model, GPU, port, PID
- Show uptime

### 3.4 Development Tools

#### [POOL-CMD-D-001] Doctor
```bash
llorch-pool dev doctor
```
**MUST** check:
- Rust toolchain (rustc, cargo)
- `hf` CLI availability
- Git and submodules status
- CUDA/Metal availability (if applicable)
- Disk space
- Report all issues with suggested fixes

#### [POOL-CMD-D-002] Setup
```bash
llorch-pool dev setup
```
**MUST**:
- Install `hf` CLI if missing
- Configure git submodules
- Download recommended model (tinyllama)
- Validate environment
- Generate default config

---

## 4. Implementation

### 4.1 Crate Structure

```
bin/pool-ctl/
├── .specs/
│   └── 00_pool-ctl.md
├── Cargo.toml
├── catalog.toml                    # Model catalog (shared with llorch-ctl)
├── src/
│   ├── main.rs                     # Entry point
│   ├── cli.rs                      # Clap CLI definitions
│   ├── config.rs                   # Configuration
│   ├── error.rs                    # Error types
│   │
│   ├── commands/
│   │   ├── mod.rs
│   │   ├── models.rs               # Model commands
│   │   ├── git.rs                  # Git commands
│   │   ├── worker.rs               # Worker commands
│   │   └── dev.rs                  # Dev commands
│   │
│   ├── catalog/
│   │   ├── mod.rs
│   │   ├── parser.rs               # TOML parser
│   │   └── models.rs               # Model metadata
│   │
│   ├── git/
│   │   ├── mod.rs
│   │   ├── operations.rs           # Git operations
│   │   └── submodules.rs           # Submodule management
│   │
│   ├── models/
│   │   ├── mod.rs
│   │   ├── download.rs             # Download via hf CLI
│   │   ├── verify.rs               # Verification
│   │   └── convert.rs              # PyTorch→GGUF conversion
│   │
│   ├── worker/
│   │   ├── mod.rs
│   │   ├── spawn.rs                # Spawn llorch-candled
│   │   ├── lifecycle.rs            # Stop/list workers
│   │   └── process.rs              # Process management
│   │
│   └── utils/
│       ├── mod.rs
│       ├── progress.rs             # Progress indicators
│       ├── output.rs               # Colored output
│       └── process.rs              # Process execution
│
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```

### 4.2 Dependencies

```toml
[dependencies]
clap = { version = "4", features = ["derive", "cargo", "env"] }
tokio = { version = "1", features = ["process", "fs"] }
serde = { version = "1", features = ["derive"] }
toml = "0.8"
anyhow = "1"
thiserror = "1"
colored = "2"
indicatif = "0.17"
sha2 = "0.10"
walkdir = "2"
which = "6"
sysinfo = "0.30"

[dev-dependencies]
tempfile = "3"
assert_cmd = "2"
predicates = "3"
```

---

## 5. Model Catalog

### Catalog Format

```toml
# bin/pool-ctl/catalog.toml (shared with llorch-ctl)

[[models]]
id = "tinyllama"
name = "TinyLlama 1.1B Chat"
repo = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
file = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
size_bytes = 669000000
params = "1.1B"
description = "Standard Llama architecture, simplest for testing"
verified = true
requires_conversion = false

[[models]]
id = "gpt2"
name = "GPT-2 FP32"
repo = "openai-community/gpt2"
file = "gpt2-fp32.gguf"
size_bytes = 500000000
params = "124M"
description = "Original transformer architecture"
verified = true
requires_conversion = true
converter = "llama.cpp"
converter_script = "reference/llama.cpp/convert_hf_to_gguf.py"
converter_args = ["--outtype", "f32"]

# ... 8 more models
```

---

## 6. Example Usage

### Local Operations (on pool host)

```bash
# Model management
llorch-pool models catalog
llorch-pool models download tinyllama
llorch-pool models list
llorch-pool models verify tinyllama
llorch-pool models delete phi3

# Git operations
llorch-pool git status
llorch-pool git pull
llorch-pool git sync
llorch-pool git submodules list
llorch-pool git submodules branch --name reference/candle --branch metal-fixes
llorch-pool git submodules update --name reference/candle

# Worker lifecycle
llorch-pool worker spawn metal --model tinyllama --gpu 0
llorch-pool worker list
llorch-pool worker stop worker-metal-0

# Development
llorch-pool dev doctor
llorch-pool dev setup
```

### Remote Operations (called by orchestrator)

```bash
# From blep (orchestrator host)
llorch pool models download tinyllama --host mac
  → SSH: ssh mac.home.arpa "cd ~/Projects/llama-orch && llorch-pool models download tinyllama"

llorch pool worker spawn metal --host mac --model tinyllama
  → SSH: ssh mac.home.arpa "cd ~/Projects/llama-orch && llorch-pool worker spawn metal --model tinyllama"
```

---

## 7. Implementation Priority

### Phase 1: Model Management (Week 1)
- Implement catalog parser
- Implement model download (hf CLI)
- Implement model list/info/verify
- Write unit tests

### Phase 2: Git Operations (Week 2)
- Implement git status/pull/sync
- Implement submodule management
- Write integration tests

### Phase 3: Worker Lifecycle (Week 3)
- Implement worker spawn
- Implement worker stop/list
- Write integration tests

### Phase 4: Development Tools (Week 4)
- Implement dev doctor
- Implement dev setup
- Write E2E tests
- Delete bash scripts

---

**Version**: 0.1.0  
**Last Updated**: 2025-10-09  
**Status**: Draft (ready for implementation)

---

**End of Specification**
