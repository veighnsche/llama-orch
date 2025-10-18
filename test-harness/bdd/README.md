# BDD Test Harness

**Owner:** TEAM-037 (Testing Team)  
**Purpose:** Behavior-Driven Development tests for llama-orch system  
**Status:** Active

---

## ⚠️ MANDATORY: READ BEFORE WORKING ON BDD TESTS ⚠️

**🔴 FRAUD WARNING 🔴**

TEAM-068 attempted checklist fraud on 2025-10-11. They deleted 21 unimplemented functions from their checklist and claimed 100% completion when only 51% was done. User caught the fraud in < 1 minute.

**REQUIRED READING:**
1. **FRAUD_WARNING.md** - Critical incident summary
2. **CHECKLIST_INTEGRITY_RULES.md** - Prevention guidelines
3. **TEAM_068_FRAUD_INCIDENT.md** - Full incident report

**RULES:**
- ❌ NEVER delete checklist items
- ✅ ALWAYS mark incomplete items as `[ ] ... ❌ TODO`
- ✅ ALWAYS show accurate completion ratios (X/N format)
- ✅ ALWAYS be honest about status

**Checklist fraud = immediate detection + public shaming + forced correction**

---

## Overview

This directory contains Gherkin feature files that describe the behavior of the llama-orch distributed inference system. These tests validate the complete system behavior from a user's perspective.

## Structure

```
test-harness/bdd/
├── README.md                    # This file
├── src/
│   └── steps/                   # Step definitions (Rust)
│       ├── authentication.rs    # TEAM-097: Auth tests
│       ├── secrets.rs           # TEAM-097: Secrets tests
│       ├── validation.rs        # TEAM-097: Input validation tests
│       └── ...                  # Other step modules
└── tests/
    └── features/                # Gherkin feature files
        ├── 010-ssh-registry-management.feature
        ├── 020-model-catalog.feature
        ├── 030-model-provisioner.feature
        ├── 100-worker-rbee-lifecycle.feature
        ├── 110-rbee-hive-lifecycle.feature
        ├── 120-queen-rbee-lifecycle.feature
        ├── 130-inference-execution.feature
        ├── 140-input-validation.feature  # Expanded by TEAM-097
        ├── 200-concurrency-scenarios.feature
        ├── 210-failure-recovery.feature
        ├── 300-authentication.feature    # TEAM-097: NEW
        ├── 310-secrets-management.feature # TEAM-097: NEW
        └── 900-integration-e2e.feature
```

## Feature Files

### P0 Security Tests (TEAM-097)

**300-authentication.feature** - 20 scenarios
- API token validation (Bearer tokens)
- Timing-safe comparison (< 10% variance)
- Multi-component auth (queen, hive, worker)
- Concurrent auth requests
- Performance benchmarks (< 1ms overhead)

**310-secrets-management.feature** - 17 scenarios
- File-based credentials (0600 permissions)
- Systemd credential support
- Memory zeroization verification
- HKDF-SHA256 key derivation
- Hot reload with SIGHUP

**140-input-validation.feature** - 30+ scenarios (25 added by TEAM-097)
- Log injection prevention
- Path traversal prevention
- Command injection prevention
- SQL/XSS injection prevention
- Property-based fuzzing tests

### Core System Tests

**100-120 series:** Component lifecycle tests
**130:** Inference execution
**200-210:** Concurrency and failure recovery
**900:** End-to-end integration

## Critical Lifecycle Rules

### RULE 1: rbee-hive is a PERSISTENT HTTP DAEMON
- **Starts:** `rbee-hive daemon` or spawned by rbee-keeper
- **Runs:** Continuously as HTTP server on port 8080
- **Dies:** ONLY when receiving SIGTERM (Ctrl+C) or explicit shutdown
- **Does NOT die:** After spawning workers, after inference completes

### RULE 2: llm-worker-rbee is a PERSISTENT HTTP DAEMON
- **Starts:** Spawned by rbee-hive
- **Runs:** Continuously as HTTP server on port 8001+
- **Dies:** When idle timeout (5 min) OR rbee-hive sends shutdown OR SIGTERM
- **Does NOT die:** After inference completes (stays idle)

### RULE 3: rbee-keeper is a CLI (EPHEMERAL)
- **Starts:** User runs command
- **Runs:** Only during command execution
- **Dies:** After command completes (exit code 0 or 1)
- **Does NOT die:** Never stays running

### RULE 4: Ephemeral Mode (rbee-keeper spawns rbee-hive)
```
rbee-keeper spawns rbee-hive as child process
    ↓
rbee-hive spawns worker
    ↓
Inference completes
    ↓
rbee-keeper sends SIGTERM to rbee-hive
    ↓
rbee-hive cascades shutdown to worker
    ↓
All processes exit
```

### RULE 5: Persistent Mode (rbee-hive pre-started)
```
Operator starts: `rbee-hive daemon &`
    ↓
rbee-hive runs continuously
    ↓
rbee-keeper connects to existing rbee-hive
    ↓
Inference completes
    ↓
rbee-keeper exits
    ↓
rbee-hive continues running
    ↓
Worker continues running (until idle timeout)
```

### RULE 6: Cascading Shutdown
```
SIGTERM → rbee-hive
    ↓
rbee-hive → POST /v1/admin/shutdown → all workers
    ↓
Workers unload models and exit
    ↓
rbee-hive clears registry and exits
    ↓
Model catalog (SQLite) persists on disk
```

### RULE 7: Worker Idle Timeout
```
Worker completes inference → idle
    ↓
5 minutes elapse without new requests
    ↓
rbee-hive sends shutdown to worker
    ↓
Worker exits, VRAM freed
    ↓
rbee-hive continues running
```

### RULE 8: Process Ownership
- **IF** rbee-keeper spawned rbee-hive → rbee-keeper owns lifecycle
- **IF** operator started rbee-hive → operator owns lifecycle
- rbee-hive always owns worker lifecycle
- Workers never own their own lifecycle (managed by rbee-hive)

## Running Tests

### 🚀 Rust xtask Runner (TEAM-111)

**Professional-grade Rust test runner integrated with xtask:**
- ✅ Live output by default (see everything in real-time!)
- ✅ Failure-focused reporting (only failures shown at end)
- ✅ Auto-generated rerun commands (instant retry of failed tests)
- ✅ Type-safe Rust implementation
- ✅ Integrated with workspace tooling

**Quick Start:**
```bash
# Run all tests with live output
cargo xtask bdd:test

# Run specific tests
cargo xtask bdd:test --tags @auth
cargo xtask bdd:test --feature lifecycle

# Quiet mode (for CI/CD)
cargo xtask bdd:test --quiet

# Help
cargo xtask bdd:test --help
```

**Features:**
- Live output streaming (default)
- Quiet mode with progress spinner
- Tag filtering (`--tags @auth`)
- Feature filtering (`--feature lifecycle`)
- Compilation check before tests
- Test discovery and counting
- Failure extraction (multiple patterns)
- Timestamped log files
- Summary generation
- Rerun command generation

### Legacy: Direct BDD Runner

You can still run the BDD runner directly:

```bash
# From repository root
LLORCH_BDD_FEATURE_PATH=test-harness/bdd/tests/features \
  cargo run --bin bdd-runner
```

### Run MVP Scenarios Only
```bash
# Run scenarios tagged with @mvp
LLORCH_BDD_FEATURE_PATH=test-harness/bdd/tests/features/test-001-mvp.feature \
  cargo run --bin bdd-runner
```

### Run Specific Scenario
```bash
# Run by line number
LLORCH_BDD_FEATURE_PATH=test-harness/bdd/tests/features/test-001-mvp.feature:25 \
  cargo run --bin bdd-runner
```

### Run Lifecycle Tests Only
```bash
# Run scenarios tagged with @lifecycle
LLORCH_BDD_FEATURE_PATH=test-harness/bdd/tests/features \
  cargo run --bin bdd-runner -- --tags @lifecycle
```

## Architecture Context

### Components Under Test

| Component | Type | Purpose | Lifecycle |
|-----------|------|---------|-----------|
| **rbee-keeper** | CLI | Remote control, calls HTTP APIs | Ephemeral (dies after command) |
| **rbee-hive** | HTTP Daemon | Pool management, worker spawning | Persistent (dies on SIGTERM) |
| **llm-worker-rbee** | HTTP Daemon | Inference execution, model in VRAM | Persistent (dies on timeout/shutdown) |
| **queen-rbee** | HTTP Daemon | Orchestration, routing (M1+) | Persistent (future) |

### Storage Strategy (TEAM-030)

**Worker Registry:** In-memory HashMap (ephemeral)
- Location: rbee-hive process memory
- Lifecycle: Lost on rbee-hive restart
- Rationale: Workers are transient, no persistence needed

**Model Catalog:** SQLite database (persistent)
- Location: `~/.rbee/models.db`
- Lifecycle: Survives rbee-hive restarts
- Rationale: Models are large files, track downloads to prevent re-downloading

## Test Patterns

### Given-When-Then Structure
```gherkin
Given <precondition>
When <action>
Then <expected outcome>
And <additional assertion>
```

### Data Tables
```gherkin
Given a worker is registered with:
  | field      | value           |
  | id         | worker-abc123   |
  | state      | idle            |
```

### Multi-line Strings
```gherkin
When I run:
  """
  rbee-keeper infer \
    --node mac \
    --prompt "hello"
  """
```

### Tags for Filtering
- `@mvp` - MVP scenarios
- `@critical` - Critical path
- `@edge-case` - Edge case handling
- `@lifecycle` - Process lifecycle
- `@happy-path` - Happy path flows

## BDD Best Practices

### ✅ DO
- Write scenarios from user's perspective
- Use Given-When-Then structure
- Make scenarios independent
- Use tags for organization
- Document lifecycle expectations
- Reference spec IDs in traceability headers

### ❌ DON'T
- Test implementation details
- Create dependencies between scenarios
- Assume execution order
- Mix multiple concerns in one scenario
- Skip lifecycle clarifications

## Related Documentation

- `/bin/.specs/.gherkin/test-001.md` - Original test specification
- `/bin/.specs/.gherkin/test-001-mvp.md` - MVP specification
- `/bin/.specs/ARCHITECTURE_MODES.md` - Ephemeral vs Persistent modes
- `/bin/.specs/FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md` - HTTP architecture
- `/.docs/testing/BDD_WIRING.md` - BDD wiring patterns
- `/.docs/testing/BDD_RUST_MOCK_LESSONS_LEARNED.md` - BDD lessons learned
- `/test-harness/TEAM_RESPONSIBILITIES.md` - Testing team responsibilities

## Contributing

When adding new scenarios:

1. **Read the spec first** - Understand the behavior being tested
2. **Check existing scenarios** - Avoid duplication
3. **Use appropriate tags** - `@mvp`, `@critical`, `@edge-case`, `@lifecycle`
4. **Document lifecycle** - Clarify when processes start/stop
5. **Add traceability** - Reference spec IDs in comments
6. **Follow patterns** - Use Given-When-Then, data tables, multi-line strings
7. **Test independently** - Each scenario should be self-contained
8. **Sign your work** - Add team signature per dev-bee-rules.md

## Status

- ✅ **test-001.feature** - Complete (67 scenarios)
- ✅ **test-001-mvp.feature** - Complete (27 scenarios)
- ✅ **Lifecycle rules** - Documented and tested
- ⏳ **Step definitions** - To be implemented
- ⏳ **BDD runner integration** - To be wired

---

**Created by:** TEAM-037 (Testing Team)  
**Last Updated:** 2025-10-10  
**Status:** Active - Ready for step definition implementation

---
Verified by Testing Team 🔍
