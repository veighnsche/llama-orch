# BDD Test Harness

**Owner:** TEAM-037 (Testing Team)  
**Purpose:** Behavior-Driven Development tests for llama-orch system  
**Status:** Active

---

## Overview

This directory contains Gherkin feature files that describe the behavior of the llama-orch distributed inference system. These tests validate the complete system behavior from a user's perspective.

## Structure

```
test-harness/bdd/
├── README.md                    # This file
└── tests/
    └── features/
        ├── test-001.feature     # Complete test suite (67 scenarios)
        └── test-001-mvp.feature # MVP subset (27 scenarios)
```

## Feature Files

### test-001.feature (Complete Suite)

**Scenarios:** 67  
**Coverage:** All behaviors from TEST-001 specification

**Organized into:**
- Happy path flows (cold start, warm start)
- Worker registry check (Phase 1)
- Pool preflight (Phase 2)
- Model provisioning (Phase 3)
- Worker preflight (Phase 4)
- Worker startup (Phase 5)
- Worker registration (Phase 6)
- Worker health check (Phase 7)
- Inference execution (Phase 8)
- Edge cases (EC1-EC10)
- Pool manager lifecycle
- Error response format
- CLI commands

### test-001-mvp.feature (MVP Subset)

**Scenarios:** 27  
**Tags:** `@mvp`, `@critical`, `@edge-case`, `@lifecycle`

**Focus areas:**
- Critical happy paths (MVP-001, MVP-002)
- Essential model provisioning (MVP-003, MVP-004)
- Worker lifecycle (MVP-005, MVP-006, MVP-007)
- Pool manager daemon behavior (MVP-008, MVP-009)
- **Lifecycle rules** (MVP-010 through MVP-013)
- 10 critical edge cases (MVP-EC1 through MVP-EC10)
- Error format validation
- Success criteria validation

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

### Run All Features
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
