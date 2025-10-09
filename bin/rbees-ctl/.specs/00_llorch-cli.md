# rbees-ctl SPEC — Orchestrator Control CLI

**Status**: Draft  
**Version**: 0.1.0  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)  
**Binary Name**: `rbees-ctl` (produces `rbees` command)

---

## Table of Contents

### 1. Executive Summary
- [1.1 Purpose](#11-purpose)
- [1.2 Architecture Realization](#12-architecture-realization)
- [1.3 Design Principles](#13-design-principles)
- [1.4 Command Structure](#14-command-structure)

### 2. Requirements
- [2.1 Functional Requirements](#21-functional-requirements)
- [2.2 Non-Functional Requirements](#22-non-functional-requirements)
- [2.3 Out of Scope](#23-out-of-scope)

### 3. Command Structure
- [3.1 Top-Level Commands](#31-top-level-commands)
- [3.2 Git Management](#32-git-management)
- [3.3 Model Management](#33-model-management)
- [3.4 Remote Operations](#34-remote-operations)
- [3.5 Build & Test](#35-build--test)
- [3.6 Development Tools](#36-development-tools)

### 4. Configuration
- [4.1 Configuration Files](#41-configuration-files)
- [4.2 Environment Variables](#42-environment-variables)
- [4.3 Precedence Rules](#43-precedence-rules)

### 5. Model Catalog
- [5.1 Catalog Structure](#51-catalog-structure)
- [5.2 Model Metadata](#52-model-metadata)
- [5.3 Verification](#53-verification)

### 6. Remote Execution
- [6.1 SSH Configuration](#61-ssh-configuration)
- [6.2 Host Profiles](#62-host-profiles)
- [6.3 Command Forwarding](#63-command-forwarding)

### 7. Testing & Verification
- [7.1 Unit Tests](#71-unit-tests)
- [7.2 Integration Tests](#72-integration-tests)
- [7.3 End-to-End Tests](#73-end-to-end-tests)

### 8. Implementation
- [8.1 Crate Structure](#81-crate-structure)
- [8.2 Dependencies](#82-dependencies)
- [8.3 Error Handling](#83-error-handling)

---

## 1. Executive Summary

### 1.1 Purpose

`rbees-ctl` (command: `rbees`) is the orchestrator control CLI for the llama-orch system. It is a CLI client that will call the `rbees-orcd` daemon HTTP API (M2+) and provides development tooling for commanding pool managers (M0).

**Core Value Propositions:**
1. **Orchestrator Control**: CLI client for rbees-orcd daemon (M2+) and direct pool control (M0)
2. **Pool Manager Commands**: Commands pool managers (mac.home.arpa, workstation.home.arpa) via SSH
3. **Development Tooling**: Git operations, model downloads, environment setup
4. **Type Safety**: Rust guarantees eliminate shell script pitfalls

**Evolution Path:**
- **M0**: CLI directly commands pools via SSH (no rbees-orcd daemon yet)
- **M2+**: CLI calls rbees-orcd HTTP API, which commands pools via HTTP

**Architecture Realization:**
```
blep.home.arpa (orchestrator)
    ↓ rbees pool <command> --host mac
mac.home.arpa (pool manager - Metal)
    ↓ executes locally
rbees-workerd (worker - Metal backend)

blep.home.arpa (orchestrator)
    ↓ rbees pool <command> --host workstation
workstation.home.arpa (pool manager - CUDA)
    ↓ executes locally
rbees-workerd (worker - CUDA backend)

blep.home.arpa (orchestrator + pool manager - CPU)
    ↓ rbees pool <command> (local)
rbees-workerd (worker - CPU backend)
```

**Replaces:**
- `scripts/homelab/llorch-remote` → `rbees` commands (orchestrator → pools)

**Works with:**
- `bin/rbees-pool/` → `rbees-pool` commands (local pool operations)
- `bin/rbees-orcd/` → Daemon (M2+, HTTP server)
- `bin/pool-managerd/` → Daemon (M1+, HTTP server)

### 1.2 Design Principles

**CLI-001**: The CLI MUST be a single binary (`rbees`) with subcommands  
**CLI-002**: Commands MUST follow the pattern: `llorch <domain> <action> [args]`  
**CLI-003**: The CLI MUST work locally and remotely (via `--remote` flag)  
**CLI-004**: The CLI MUST be self-documenting (comprehensive `--help`)  
**CLI-005**: The CLI MUST provide rich, colored output with progress indicators  
**CLI-006**: The CLI MUST handle errors gracefully with actionable messages  
**CLI-007**: The CLI MUST support both interactive and CI/CD modes  
**CLI-008**: The CLI MUST be future-proof for rbees-orcd/pool-managerd/worker-orcd operations

### 1.2 Architecture Realization

**The bash scripts were already implementing the orchestrator pattern!**

**What we had:**
- `llorch-remote mac.home.arpa models-download tinyllama` 
  - = Orchestrator (blep) telling pool manager (mac) to provision model
- `llorch-remote workstation.home.arpa build cuda`
  - = Orchestrator (blep) telling pool manager (workstation) to build worker
- `llorch-remote mac.home.arpa test metal`
  - = Orchestrator (blep) telling pool manager (mac) to run tests

**What we're building:**
```
llorch (orchestrator CLI on blep)
├── pool (orchestrator → pool manager commands)
│   ├── register <host>           # Register pool manager
│   ├── list                      # List all pools
│   ├── status <host>             # Get pool status
│   ├── models                    # Model provisioning commands
│   │   ├── download <model> --host <host>
│   │   ├── list --host <host>
│   │   └── verify <model> --host <host>
│   ├── git                       # Git commands on pool
│   │   ├── status --host <host>
│   │   ├── pull --host <host>
│   │   └── sync --host <host>
│   ├── worker                    # Worker lifecycle
│   │   ├── spawn --host <host> --backend <backend>
│   │   ├── stop --host <host> --worker-id <id>
│   │   └── list --host <host>
│   └── build                     # Build on pool
│       └── worker <backend> --host <host>
├── jobs (orchestrator job management - future M2)
│   ├── submit
│   ├── list
│   ├── cancel
│   └── status
└── dev (development utilities)
    ├── setup
    ├── doctor
    └── check
```

### 1.3 Design Principles

**ORCH-001**: This CLI controls rbees-orcd daemon (M2+) and pools (M0+)  
**ORCH-002**: Shares logic with rbees-orcd via orchestrator-core crate  
**ORCH-003**: Commands to pool managers use SSH (M0) or HTTP (M2+)  
**ORCH-004**: The CLI makes intelligent decisions (scheduling, admission) in M0  
**ORCH-005**: The rbees-orcd daemon makes decisions in M2+ (CLI calls HTTP API)  
**ORCH-006**: The CLI MUST NEVER start REPL or conversation (HARD RULE)  
**ORCH-007**: The CLI MUST be self-documenting (comprehensive `--help`)  
**ORCH-008**: The CLI MUST provide rich, colored output with progress indicators  
**ORCH-009**: The CLI MUST handle errors gracefully with actionable messages

### HARD RULE: NO REPL, NO CONVERSATION

**FORBIDDEN:**
```bash
# ❌ NEVER implement these
llorch chat
llorch repl
llorch interactive
llorch conversation

# ❌ NEVER implement interactive mode
$ llorch
> submit job with llama3
> list jobs
```

**WHY:**
- Agentic API is HTTP-based (POST /v2/tasks with SSE streaming)
- Conversations happen via web UI or HTTP clients
- CLI is for CONTROL and AUTOMATION, not CONVERSATION
- Terminal is not the UX for LLM interactions

**CORRECT:**
```bash
# ✅ Single commands for automation
llorch jobs submit --model llama3 --prompt "hello"
llorch jobs list --format json | jq
llorch jobs cancel job-123
```

---

## 2. Requirements

### 2.1 Functional Requirements

#### [CLI-F-001] Git Management
The CLI MUST provide git operations with submodule support:
- **MUST**: status, pull, sync, submodules list
- **MUST**: submodule branch switching
- **MUST**: submodule update (single or all)
- **MUST**: submodule reset to committed version
- **SHOULD**: clean untracked files (interactive)
- **SHOULD**: branch listing

#### [CLI-F-002] Model Management
The CLI MUST provide model download and management:
- **MUST**: list downloaded models
- **MUST**: show catalog with verified HuggingFace repos
- **MUST**: download models using `hf` CLI
- **MUST**: verify model integrity
- **MUST**: show model info (size, repo, local files)
- **SHOULD**: delete models
- **SHOULD**: show disk usage
- **MAY**: convert models (PyTorch → GGUF)

#### [CLI-F-003] Remote Operations
The CLI MUST support remote execution via SSH:
- **MUST**: execute any local command on remote host
- **MUST**: support host profiles (mac.home.arpa, workstation.home.arpa)
- **MUST**: forward git commands to remote
- **MUST**: forward model commands to remote
- **SHOULD**: support SSH key authentication
- **SHOULD**: support SSH config integration
- **MAY**: support SSH agent forwarding

#### [CLI-F-004] Build & Test
The CLI MUST provide build and test operations:
- **MUST**: build worker-orcd (rbees-workerd) with backend selection
- **MUST**: run unit tests
- **MUST**: run integration tests
- **MUST**: run smoke tests
- **SHOULD**: build rbees-orcd (future)
- **SHOULD**: build pool-managerd (future)
- **MAY**: support custom test filters

#### [CLI-F-005] Development Tools
The CLI MUST provide development utilities:
- **MUST**: environment check (doctor command)
- **MUST**: dependency verification
- **SHOULD**: setup wizard for new developers
- **SHOULD**: configuration validation
- **MAY**: benchmark utilities

### 2.2 Non-Functional Requirements

#### [CLI-NF-001] Performance
- Command startup MUST be < 100ms (cold start)
- Model catalog loading MUST be < 50ms
- SSH connection MUST timeout after 10s
- Progress indicators MUST update every 100ms

#### [CLI-NF-002] Usability
- Help text MUST be comprehensive and examples-driven
- Error messages MUST be actionable (suggest fixes)
- Output MUST be colored and formatted (unless --no-color)
- Commands MUST support --json for machine parsing

#### [CLI-NF-003] Reliability
- The CLI MUST handle network failures gracefully
- The CLI MUST validate inputs before execution
- The CLI MUST provide rollback on partial failures
- The CLI MUST log operations for debugging

#### [CLI-NF-004] Maintainability
- Code MUST be modular (one module per domain)
- Commands MUST be testable in isolation
- Configuration MUST be versioned
- Breaking changes MUST be documented

### 2.3 Out of Scope

**CLI-OOS-001**: The CLI MUST NOT embed rbees-orcd/pool-managerd/worker-orcd binaries  
**CLI-OOS-002**: The CLI MUST NOT manage runtime daemons (use systemd/launchd)  
**CLI-OOS-003**: The CLI MUST NOT provide a TUI (terminal UI) in v0.1.0  
**CLI-OOS-004**: The CLI MUST NOT support Windows natively (WSL only)

---

## 3. Command Structure

### 3.1 Top-Level Commands

```bash
llorch --version
llorch --help
llorch <domain> <action> [options]
```

**Domains:**
- `git` - Git operations with submodule support
- `models` - Model download and management
- `remote` - Remote execution wrapper
- `build` - Build binaries
- `test` - Run tests
- `dev` - Development utilities

### 3.2 Git Management

#### [CLI-GIT-001] Status Command
```bash
llorch git status [--remote HOST]
```
**MUST**:
- Show current branch and commit
- Show uncommitted changes
- List all submodules with status
- Display system info (Rust version, OS)

#### [CLI-GIT-002] Pull Command
```bash
llorch git pull [--remote HOST]
```
**MUST**:
- Pull latest changes from origin
- Update all submodules recursively
- Show summary of changes

#### [CLI-GIT-003] Sync Command
```bash
llorch git sync [--remote HOST] [--force]
```
**MUST**:
- Hard reset to origin/main
- Update all submodules to committed versions
- Prompt for confirmation unless --force

#### [CLI-GIT-004] Submodules Command
```bash
llorch git submodules [--remote HOST]
llorch git submodules list
llorch git submodules status
llorch git submodules update [--all | --name NAME]
llorch git submodules reset --name NAME
llorch git submodules branch --name NAME [--branch BRANCH]
```
**MUST**:
- List all submodules with paths and commits
- Show current branch for each submodule
- Support switching submodule branches
- Support updating single or all submodules

**Submodules:**
- `reference/candle`
- `reference/candle-vllm`
- `reference/llama.cpp`
- `reference/mistral.rs`

### 3.3 Model Management

#### [CLI-MODEL-001] Catalog Command
```bash
llorch models catalog [--remote HOST]
```
**MUST**:
- Display all available models with metadata
- Show HuggingFace repo URLs (verified)
- Show expected sizes
- Show download status (local/not downloaded)

#### [CLI-MODEL-002] List Command
```bash
llorch models list [--remote HOST]
```
**MUST**:
- List all downloaded models
- Show file sizes
- Show total disk usage
- Support filtering by model name

#### [CLI-MODEL-003] Download Command
```bash
llorch models download <MODEL> [--remote HOST] [--force]
```
**MUST**:
- Download model using `hf` CLI
- Verify `hf` command availability
- Show progress indicator
- Handle conversion if needed (gpt2, granite)
- Compute SHA256 checksum after download
- Skip if already exists unless --force

**Supported Models:**
- `tinyllama` - TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF (669 MB)
- `qwen` - Qwen/Qwen2.5-0.5B-Instruct-GGUF (352 MB)
- `qwen-fp16` - Qwen/Qwen2.5-0.5B-Instruct-GGUF (1.1 GB)
- `phi3` - microsoft/Phi-3-mini-4k-instruct-gguf (2.4 GB)
- `llama3` - QuantFactory/Meta-Llama-3-8B-Instruct-GGUF (4.9 GB)
- `llama32` - tensorblock/Komodo-Llama-3.2-3B-v2-fp16-GGUF (6.4 GB)
- `llama2` - TheBloke/Llama-2-7B-GGUF (7.2 GB)
- `mistral` - TheBloke/Mistral-7B-Instruct-v0.2-GGUF (14 GB)
- `gpt2` - openai-community/gpt2 (500 MB, requires conversion)
- `granite` - ibm-granite/granite-4.0-micro (8 GB, requires conversion)

#### [CLI-MODEL-004] Info Command
```bash
llorch models info <MODEL> [--remote HOST]
```
**MUST**:
- Show catalog metadata (repo, size, description)
- Show local files if downloaded
- Show disk usage
- Show config.json if present

#### [CLI-MODEL-005] Verify Command
```bash
llorch models verify <MODEL> [--remote HOST]
```
**MUST**:
- Check if model files exist
- Verify file sizes > 0
- Report status for each file
- Exit with error if verification fails

#### [CLI-MODEL-006] Delete Command
```bash
llorch models delete <MODEL> [--remote HOST] [--force]
```
**MUST**:
- Show disk usage before deletion
- Prompt for confirmation unless --force
- Remove model directory
- Report success/failure

### 3.4 Remote Operations

#### [CLI-REMOTE-001] Exec Command
```bash
llorch remote exec <HOST> <COMMAND> [ARGS...]
```
**MUST**:
- Execute arbitrary command on remote host
- Forward stdin/stdout/stderr
- Preserve exit codes
- Support SSH key authentication

#### [CLI-REMOTE-002] Command Forwarding
```bash
llorch git status --remote mac.home.arpa
llorch models list --remote workstation.home.arpa
llorch build worker cuda --remote workstation.home.arpa
```
**MUST**:
- Forward command to remote host via SSH
- Execute `rbees` binary on remote
- Stream output back to local terminal
- Handle SSH connection failures gracefully

#### [CLI-REMOTE-003] Host Profiles
```bash
llorch remote hosts list
llorch remote hosts add <NAME> <USER@HOST>
llorch remote hosts remove <NAME>
```
**MUST**:
- Store host profiles in config
- Support aliases (mac, workstation, gpu-node-1)
- Validate SSH connectivity
- Support SSH config integration

### 3.5 Build & Test

#### [CLI-BUILD-001] Worker Build
```bash
llorch build worker <BACKEND> [--remote HOST] [--release]
```
**MUST**:
- Build rbees-workerd with specified backend
- Support backends: cpu, cuda, metal
- Default to release mode
- Show build progress
- Report binary location

**Future:**
```bash
llorch build orchestrator [--remote HOST]
llorch build pool-manager [--remote HOST]
```

#### [CLI-TEST-001] Test Commands
```bash
llorch test unit [--remote HOST] [--backend BACKEND]
llorch test integration [--remote HOST] [--backend BACKEND]
llorch test smoke [--remote HOST] [--backend BACKEND]
llorch test all [--remote HOST] [--backend BACKEND]
```
**MUST**:
- Run specified test suite
- Support backend filtering
- Show test results
- Exit with error on test failures

### 3.6 Development Tools

#### [CLI-DEV-001] Doctor Command
```bash
llorch dev doctor [--remote HOST]
```
**MUST**:
- Check Rust toolchain (rustc, cargo)
- Check `hf` CLI availability
- Check git and submodules
- Check SSH configuration (if remote)
- Check CUDA/Metal availability (if applicable)
- Report all issues with suggested fixes

#### [CLI-DEV-002] Setup Command
```bash
llorch dev setup [--remote HOST]
```
**MUST**:
- Install required dependencies
- Configure git submodules
- Download recommended models
- Validate environment
- Generate default config

#### [CLI-DEV-003] Check Command
```bash
llorch dev check [--remote HOST]
```
**MUST**:
- Validate configuration files
- Check for common issues
- Verify binary paths
- Test SSH connectivity (if remote)

---

## 4. Configuration

### 4.1 Configuration Files

#### [CLI-CFG-001] Config File Location
```toml
# ~/.config/llorch/config.toml (Linux/macOS)
# %APPDATA%\llorch\config.toml (Windows)
```

#### [CLI-CFG-002] Config File Structure
```toml
[general]
model_base_dir = ".test-models"
repo_root = "~/Projects/llama-orch"
color = true
log_level = "info"

[remote]
default_host = "mac.home.arpa"
ssh_timeout = 10

[remote.hosts.mac]
user = "vinceliem"
host = "mac.home.arpa"
identity_file = "~/.ssh/id_ed25519"

[remote.hosts.workstation]
user = "vince"
host = "workstation.home.arpa"
identity_file = "~/.ssh/id_ed25519"

[models]
catalog_url = "https://raw.githubusercontent.com/veighnsche/llama-orch/main/tools/llorch-cli/catalog.toml"
cache_ttl = 3600

[build]
default_backend = "cuda"
release = true
```

### 4.2 Environment Variables

#### [CLI-ENV-001] Environment Variable Support
```bash
LLORCH_CONFIG_PATH      # Override config file location
LLORCH_MODEL_BASE_DIR   # Override model directory
LLORCH_REPO_ROOT        # Override repo root
LLORCH_REMOTE_HOST      # Override default remote host
LLORCH_LOG_LEVEL        # Override log level (trace, debug, info, warn, error)
LLORCH_NO_COLOR         # Disable colored output
```

### 4.3 Precedence Rules

#### [CLI-PREC-001] Configuration Precedence
Priority (highest to lowest):
1. Command-line flags
2. Environment variables
3. Config file
4. Built-in defaults

---

## 5. Model Catalog

### 5.1 Catalog Structure

#### [CLI-CAT-001] Catalog File Format
```toml
# tools/llorch-cli/catalog.toml

[[models]]
id = "tinyllama"
name = "TinyLlama 1.1B Chat"
repo = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
file = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
size = "669M"
params = "1.1B"
description = "Standard Llama architecture, simplest for testing"
verified = true
requires_conversion = false

[[models]]
id = "gpt2"
name = "GPT-2 FP32"
repo = "openai-community/gpt2"
file = "gpt2-fp32.gguf"
size = "500M"
params = "124M"
description = "Original transformer architecture"
verified = true
requires_conversion = true
converter = "llama.cpp"
```

### 5.2 Model Metadata

#### [CLI-META-001] Required Metadata Fields
**MUST** include:
- `id` - Unique identifier
- `name` - Human-readable name
- `repo` - HuggingFace repo (verified)
- `file` - Target filename
- `size` - Expected size
- `verified` - Whether repo URL is verified

**SHOULD** include:
- `params` - Parameter count
- `description` - Brief description
- `requires_conversion` - Whether conversion needed
- `converter` - Conversion tool (if needed)

### 5.3 Verification

#### [CLI-VER-001] Catalog Verification
The CLI MUST:
- Validate catalog TOML syntax
- Verify all required fields present
- Check for duplicate IDs
- Warn on unverified repos

---

## 6. Remote Execution

### 6.1 SSH Configuration

#### [CLI-SSH-001] SSH Connection
The CLI MUST:
- Use SSH key authentication by default
- Support SSH config (~/.ssh/config)
- Timeout after 10 seconds
- Provide clear error messages on connection failure

#### [CLI-SSH-002] SSH Command Format
```bash
ssh -o BatchMode=yes \
    -o ConnectTimeout=10 \
    -o StrictHostKeyChecking=no \
    user@host "cd /path/to/repo && rbees <command>"
```

### 6.2 Host Profiles

#### [CLI-PROF-001] Host Profile Storage
Profiles MUST be stored in config file:
```toml
[remote.hosts.mac]
user = "vinceliem"
host = "mac.home.arpa"
identity_file = "~/.ssh/id_ed25519"
repo_path = "~/Projects/llama-orch"
```

### 6.3 Command Forwarding

#### [CLI-FWD-001] Command Forwarding Logic
When `--remote HOST` is specified:
1. Resolve host from profile or use as-is
2. Establish SSH connection
3. Change to repo directory on remote
4. Execute `llorch <command>` on remote
5. Stream output back to local terminal
6. Preserve exit code

---

## 7. Testing & Verification

### 7.1 Unit Tests

#### [CLI-TEST-U-001] Unit Test Coverage
**MUST** test:
- Config parsing
- Catalog parsing
- Command argument parsing
- Error handling
- Model metadata validation

### 7.2 Integration Tests

#### [CLI-TEST-I-001] Integration Test Coverage
**MUST** test:
- Git operations (with test repo)
- Model download (with small test model)
- Remote execution (with mock SSH)
- Build commands (with test crate)

### 7.3 End-to-End Tests

#### [CLI-TEST-E-001] E2E Test Coverage
**SHOULD** test:
- Full workflow: clone → pull → build → test
- Remote operations with real SSH
- Model download and verification

---

## 8. Implementation

### 8.1 Crate Structure

```
tools/llorch-cli/
├── Cargo.toml
├── src/
│   ├── main.rs           # CLI entry point
│   ├── cli.rs            # Clap CLI definitions
│   ├── config.rs         # Configuration management
│   ├── error.rs          # Error types
│   ├── commands/
│   │   ├── mod.rs
│   │   ├── git.rs        # Git subcommands
│   │   ├── models.rs     # Model subcommands
│   │   ├── remote.rs     # Remote subcommands
│   │   ├── build.rs      # Build subcommands
│   │   ├── test.rs       # Test subcommands
│   │   └── dev.rs        # Dev subcommands
│   ├── catalog/
│   │   ├── mod.rs
│   │   ├── parser.rs     # Catalog TOML parser
│   │   └── models.rs     # Model metadata
│   ├── git/
│   │   ├── mod.rs
│   │   ├── operations.rs # Git operations
│   │   └── submodules.rs # Submodule management
│   ├── models/
│   │   ├── mod.rs
│   │   ├── download.rs   # Model download logic
│   │   └── verify.rs     # Model verification
│   ├── remote/
│   │   ├── mod.rs
│   │   ├── ssh.rs        # SSH connection
│   │   └── forward.rs    # Command forwarding
│   └── utils/
│       ├── mod.rs
│       ├── progress.rs   # Progress indicators
│       └── output.rs     # Colored output
├── catalog.toml          # Model catalog
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```

### 8.2 Dependencies

#### [CLI-DEP-001] Required Dependencies
```toml
[dependencies]
clap = { version = "4", features = ["derive", "cargo"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
toml = "0.8"
anyhow = "1"
thiserror = "1"
colored = "2"
indicatif = "0.17"
reqwest = { version = "0.11", features = ["blocking"] }
sha2 = "0.10"
walkdir = "2"
```

### 8.3 Error Handling

#### [CLI-ERR-001] Error Types
```rust
#[derive(thiserror::Error, Debug)]
pub enum CliError {
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Git operation failed: {0}")]
    Git(String),
    
    #[error("Model operation failed: {0}")]
    Model(String),
    
    #[error("Remote execution failed: {0}")]
    Remote(String),
    
    #[error("Build failed: {0}")]
    Build(String),
    
    #[error("Test failed: {0}")]
    Test(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
```

#### [CLI-ERR-002] Error Messages
Error messages MUST:
- Be actionable (suggest fixes)
- Include context (what was being attempted)
- Show relevant paths/commands
- Provide links to documentation when applicable

---

## 9. Migration Plan

### 9.1 Phase 1: Core Commands (Week 1)
- Implement `llorch git` subcommands
- Implement `llorch models` subcommands
- Implement config management
- Implement catalog parser

### 9.2 Phase 2: Remote Support (Week 2)
- Implement `llorch remote` subcommands
- Implement SSH connection logic
- Implement command forwarding
- Test with real remote hosts

### 9.3 Phase 3: Build & Test (Week 3)
- Implement `llorch build` subcommands
- Implement `llorch test` subcommands
- Implement `llorch dev` subcommands
- Write comprehensive tests

### 9.4 Phase 4: Polish & Documentation (Week 4)
- Write user documentation
- Write developer documentation
- Add examples and tutorials
- Delete old bash scripts

---

## 10. Future Enhancements

### 10.1 Orchestratord Integration (M2+)
```bash
llorch orchestrator start [--config PATH]
llorch orchestrator stop
llorch orchestrator status
llorch orchestrator logs
```

### 10.2 Pool Manager Integration (M1+)
```bash
llorch pool-manager start [--config PATH]
llorch pool-manager stop
llorch pool-manager status
llorch pool-manager workers list
```

### 10.3 Worker Integration (M0+)
```bash
llorch worker start <BACKEND> [--model MODEL] [--gpu ID]
llorch worker stop <ID>
llorch worker list
llorch worker logs <ID>
```

### 10.4 TUI Mode
```bash
llorch tui
```
Interactive terminal UI for monitoring and management.

---

**Version**: 0.1.0  
**Last Updated**: 2025-10-09  
**Status**: Draft (ready for implementation)

---

**End of Specification**
