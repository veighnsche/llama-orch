# llorch-cli Implementation Plan

**Status**: Draft  
**Version**: 0.1.0  
**Timeline**: 4 weeks

---

## Overview

This document outlines the implementation plan for `llorch-cli`, a unified Rust CLI tool that replaces bash scripts with type-safe, testable tooling.

**Key Constraints:**
- MUST align with triple binary architecture (orchestratord, pool-managerd, worker-orcd)
- MUST NOT embed binary logic (CLI is tooling, not runtime)
- MUST be future-proof for M0→M5 milestones
- MUST maintain separation of concerns

---

## Architecture Alignment

### Triple Binary Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ RUNTIME BINARIES (Not in llorch-cli)                        │
├─────────────────────────────────────────────────────────────┤
│ bin/orchestratord/     → Orchestrator daemon (M2+)          │
│ bin/pool-managerd/     → Pool manager daemon (M1+)          │
│ bin/llorch-candled/    → Worker daemon (M0+)                │
│   (replaces deprecated worker-orcd)                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ DEVELOPER TOOLING (llorch-cli)                              │
├─────────────────────────────────────────────────────────────┤
│ tools/llorch-cli/      → Unified CLI for dev workflow       │
│   ├── git              → Git + submodule management         │
│   ├── models           → Model download/management          │
│   ├── remote           → SSH-based remote execution         │
│   ├── build            → Build runtime binaries             │
│   ├── test             → Run test suites                    │
│   └── dev              → Development utilities              │
└─────────────────────────────────────────────────────────────┘
```

### Separation of Concerns

**llorch-cli responsibilities:**
- Development workflow automation
- Model provisioning (download, verify)
- Git operations (submodules, sync)
- Build orchestration (cargo build wrappers)
- Test execution (cargo test wrappers)
- Remote execution (SSH forwarding)

**llorch-cli does NOT:**
- Manage runtime daemons (use systemd/launchd)
- Embed orchestrator/pool-manager/worker logic
- Handle inference requests
- Manage worker lifecycle
- Implement scheduling/queueing

---

## Folder Structure

```
tools/llorch-cli/
├── .specs/
│   ├── 00_llorch-cli.md              # Main specification
│   └── 01_implementation_plan.md     # This document
├── Cargo.toml
├── README.md
├── catalog.toml                       # Model catalog (verified HF repos)
├── src/
│   ├── main.rs                        # Entry point
│   ├── cli.rs                         # Clap CLI definitions
│   ├── config.rs                      # Config management
│   ├── error.rs                       # Error types
│   │
│   ├── commands/                      # Command implementations
│   │   ├── mod.rs
│   │   ├── git.rs                     # Git subcommands
│   │   ├── models.rs                  # Model subcommands
│   │   ├── remote.rs                  # Remote subcommands
│   │   ├── build.rs                   # Build subcommands
│   │   ├── test.rs                    # Test subcommands
│   │   └── dev.rs                     # Dev subcommands
│   │
│   ├── catalog/                       # Model catalog
│   │   ├── mod.rs
│   │   ├── parser.rs                  # TOML parser
│   │   └── models.rs                  # Model metadata types
│   │
│   ├── git/                           # Git operations
│   │   ├── mod.rs
│   │   ├── operations.rs              # Basic git ops
│   │   └── submodules.rs              # Submodule management
│   │
│   ├── models/                        # Model management
│   │   ├── mod.rs
│   │   ├── download.rs                # Download logic (hf CLI)
│   │   ├── verify.rs                  # Verification
│   │   └── convert.rs                 # PyTorch→GGUF conversion
│   │
│   ├── remote/                        # Remote execution
│   │   ├── mod.rs
│   │   ├── ssh.rs                     # SSH connection
│   │   ├── forward.rs                 # Command forwarding
│   │   └── hosts.rs                   # Host profile management
│   │
│   ├── build/                         # Build operations
│   │   ├── mod.rs
│   │   ├── worker.rs                  # Build llorch-candled
│   │   ├── orchestrator.rs            # Build orchestratord (future)
│   │   └── pool_manager.rs            # Build pool-managerd (future)
│   │
│   ├── test/                          # Test operations
│   │   ├── mod.rs
│   │   ├── unit.rs                    # Unit test runner
│   │   ├── integration.rs             # Integration test runner
│   │   └── smoke.rs                   # Smoke test runner
│   │
│   └── utils/                         # Utilities
│       ├── mod.rs
│       ├── progress.rs                # Progress indicators
│       ├── output.rs                  # Colored output
│       ├── fs.rs                      # Filesystem utilities
│       └── process.rs                 # Process execution
│
└── tests/
    ├── unit/
    │   ├── config_test.rs
    │   ├── catalog_test.rs
    │   └── git_test.rs
    ├── integration/
    │   ├── git_integration_test.rs
    │   ├── models_integration_test.rs
    │   └── remote_integration_test.rs
    └── e2e/
        └── full_workflow_test.rs
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1)

**Goal:** Core infrastructure and git commands

**Tasks:**
1. Create crate structure
2. Implement CLI skeleton with clap
3. Implement config management
4. Implement error types
5. Implement git subcommands:
   - `llorch git status`
   - `llorch git pull`
   - `llorch git sync`
   - `llorch git submodules list`
   - `llorch git submodules status`
   - `llorch git submodules update`
   - `llorch git submodules reset`
   - `llorch git submodules branch`

**Deliverables:**
- Working `llorch git` commands
- Config file support
- Unit tests for git operations

**Acceptance Criteria:**
- [ ] `llorch git status` shows repo and submodule status
- [ ] `llorch git pull` updates repo and submodules
- [ ] `llorch git sync` hard resets to origin/main
- [ ] `llorch git submodules` manages all 4 submodules
- [ ] All git commands have comprehensive tests

### Phase 2: Model Management (Week 2)

**Goal:** Model catalog and download functionality

**Tasks:**
1. Implement catalog parser (TOML)
2. Create model metadata types
3. Implement model subcommands:
   - `llorch models catalog`
   - `llorch models list`
   - `llorch models download`
   - `llorch models info`
   - `llorch models verify`
   - `llorch models delete`
4. Integrate `hf` CLI for downloads
5. Implement conversion support (gpt2, granite)

**Deliverables:**
- Working `llorch models` commands
- Model catalog with 10 verified models
- Download with progress indicators
- SHA256 verification

**Acceptance Criteria:**
- [ ] `llorch models catalog` shows all 10 models with HF repos
- [ ] `llorch models download tinyllama` downloads successfully
- [ ] `llorch models download gpt2` converts PyTorch→GGUF
- [ ] `llorch models verify` checks file integrity
- [ ] All model commands have comprehensive tests

### Phase 3: Remote Execution (Week 3)

**Goal:** SSH-based remote command forwarding

**Tasks:**
1. Implement SSH connection logic
2. Implement host profile management
3. Implement command forwarding
4. Add `--remote` flag to all commands
5. Implement remote subcommands:
   - `llorch remote exec`
   - `llorch remote hosts list`
   - `llorch remote hosts add`
   - `llorch remote hosts remove`

**Deliverables:**
- Working `llorch remote` commands
- SSH key authentication
- Command forwarding for git/models
- Host profile storage in config

**Acceptance Criteria:**
- [ ] `llorch git status --remote mac` executes on remote
- [ ] `llorch models list --remote workstation` lists remote models
- [ ] `llorch remote exec mac "uname -a"` executes arbitrary command
- [ ] `llorch remote hosts` manages host profiles
- [ ] All remote commands have integration tests

### Phase 4: Build & Test (Week 4)

**Goal:** Build and test orchestration

**Tasks:**
1. Implement build subcommands:
   - `llorch build worker <backend>`
   - `llorch build orchestrator` (future-ready)
   - `llorch build pool-manager` (future-ready)
2. Implement test subcommands:
   - `llorch test unit`
   - `llorch test integration`
   - `llorch test smoke`
   - `llorch test all`
3. Implement dev subcommands:
   - `llorch dev doctor`
   - `llorch dev setup`
   - `llorch dev check`
4. Write comprehensive documentation
5. Delete old bash scripts

**Deliverables:**
- Working `llorch build` commands
- Working `llorch test` commands
- Working `llorch dev` commands
- Complete documentation
- Migration guide

**Acceptance Criteria:**
- [ ] `llorch build worker cuda` builds llorch-candled
- [ ] `llorch test unit` runs all unit tests
- [ ] `llorch dev doctor` checks environment
- [ ] `llorch dev setup` configures new dev environment
- [ ] All commands documented with examples
- [ ] Old bash scripts deleted

---

## Dependencies

### Required Crates

```toml
[dependencies]
# CLI framework
clap = { version = "4", features = ["derive", "cargo", "env"] }

# Async runtime
tokio = { version = "1", features = ["full"] }

# Serialization
serde = { version = "1", features = ["derive"] }
toml = "0.8"

# Error handling
anyhow = "1"
thiserror = "1"

# Terminal output
colored = "2"
indicatif = "0.17"

# HTTP client (for catalog updates)
reqwest = { version = "0.11", features = ["blocking"] }

# Cryptography
sha2 = "0.10"

# Filesystem
walkdir = "2"

# Process execution
which = "6"

# Git operations (optional, may use git CLI)
git2 = { version = "0.18", optional = true }

[dev-dependencies]
tempfile = "3"
assert_cmd = "2"
predicates = "3"
```

### External Dependencies

**Required:**
- `git` - Git operations
- `hf` - HuggingFace CLI (for model downloads)
- `cargo` - Rust build system
- `ssh` - Remote execution

**Optional:**
- `python3` - Model conversion (gpt2, granite)
- `llama.cpp` - GGUF conversion tools

---

## Testing Strategy

### Unit Tests

**Coverage targets:**
- Config parsing: 100%
- Catalog parsing: 100%
- Error handling: 100%
- Utility functions: 100%

**Test files:**
- `tests/unit/config_test.rs`
- `tests/unit/catalog_test.rs`
- `tests/unit/git_test.rs`
- `tests/unit/models_test.rs`

### Integration Tests

**Coverage targets:**
- Git operations: All subcommands
- Model operations: Download, verify, list
- Remote execution: Command forwarding
- Build operations: Worker build

**Test files:**
- `tests/integration/git_integration_test.rs`
- `tests/integration/models_integration_test.rs`
- `tests/integration/remote_integration_test.rs`
- `tests/integration/build_integration_test.rs`

### End-to-End Tests

**Scenarios:**
1. Fresh clone → setup → build → test
2. Model download → verify → use in test
3. Remote execution → build → test
4. Submodule branch switching → build → test

**Test files:**
- `tests/e2e/full_workflow_test.rs`

---

## Migration from Bash Scripts

### Scripts to Delete

**After Phase 4 completion:**
```bash
# Delete these files
rm -rf scripts/llorch-models
rm -rf scripts/llorch-git
rm -rf scripts/homelab/llorch-remote
rm -rf .docs/testing/download_*.sh
```

### Migration Guide

**Old → New command mapping:**

```bash
# Git operations
scripts/llorch-git status
→ llorch git status

scripts/llorch-git pull
→ llorch git pull

scripts/llorch-git submodules
→ llorch git submodules list

# Model operations
scripts/llorch-models catalog
→ llorch models catalog

scripts/llorch-models download tinyllama
→ llorch models download tinyllama

.docs/testing/download_tinyllama.sh
→ llorch models download tinyllama

# Remote operations
scripts/homelab/llorch-remote mac.home.arpa status
→ llorch git status --remote mac

scripts/homelab/llorch-remote mac.home.arpa models-list
→ llorch models list --remote mac

scripts/homelab/llorch-remote workstation.home.arpa build cuda
→ llorch build worker cuda --remote workstation
```

---

## Future Enhancements (Post-v0.1.0)

### M0: Worker Integration
```bash
llorch worker start cuda --model tinyllama --gpu 0
llorch worker stop <id>
llorch worker list
llorch worker logs <id>
```

### M1: Pool Manager Integration
```bash
llorch pool-manager start --config pool.toml
llorch pool-manager stop
llorch pool-manager status
llorch pool-manager workers list
```

### M2: Orchestrator Integration
```bash
llorch orchestrator start --config orch.toml
llorch orchestrator stop
llorch orchestrator status
llorch orchestrator jobs list
llorch orchestrator jobs submit --model llama3 --prompt "Hello"
```

### M3+: TUI Mode
```bash
llorch tui
```
Interactive terminal UI with:
- Real-time worker status
- Job queue visualization
- Resource usage graphs
- Log streaming

---

## Success Metrics

### Phase 1 (Week 1)
- [ ] All git commands functional
- [ ] 100% unit test coverage for git module
- [ ] Documentation complete for git commands

### Phase 2 (Week 2)
- [ ] All model commands functional
- [ ] 10 models in verified catalog
- [ ] 100% unit test coverage for models module
- [ ] Documentation complete for model commands

### Phase 3 (Week 3)
- [ ] All remote commands functional
- [ ] SSH forwarding working for all commands
- [ ] Integration tests passing
- [ ] Documentation complete for remote commands

### Phase 4 (Week 4)
- [ ] All build/test/dev commands functional
- [ ] E2E tests passing
- [ ] Complete documentation
- [ ] Old bash scripts deleted
- [ ] Migration guide published

### Overall Success Criteria
- [ ] Single binary replaces all bash scripts
- [ ] 90%+ test coverage
- [ ] Comprehensive documentation
- [ ] Zero regressions from bash scripts
- [ ] Faster execution (< 100ms startup)
- [ ] Better error messages
- [ ] Type-safe implementation

---

**Version**: 0.1.0  
**Last Updated**: 2025-10-09  
**Status**: Ready for implementation

---

**End of Implementation Plan**
