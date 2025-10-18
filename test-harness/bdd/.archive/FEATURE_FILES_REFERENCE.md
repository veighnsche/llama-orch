# Feature Files Reference Card
# Created by: TEAM-077
# Date: 2025-10-11
# Updated: 2025-10-11 14:24 - Clarified M0-M1 scope, M2+ features documented but deferred

## Quick Reference

### File Naming Convention
- **Format:** `XXY-feature-name.feature`
- **XX:** Feature group (00-99)
- **Y:** Sub-feature (0-9)
- **Example:** `010-ssh-registry-management.feature`

### M0-M1 Feature Files (12 files planned, 91 scenarios current)

```
010-ssh-registry-management.feature      (10 scenarios) M1
├─ SSH connections, authentication
├─ Node registry management
└─ Happy + Error scenarios

020-model-catalog.feature                (13 scenarios) M1
├─ Model download from Hugging Face
├─ Model catalog (SQLite)
├─ GGUF support and metadata
└─ Happy + Error scenarios (EH-007a/b, EH-008a/b/c, EC2)

025-worker-provisioning.feature          (? scenarios) M1 NEW!
├─ Build worker binaries from git
├─ Track worker types per rbee-hive
├─ Version compatibility
├─ Binary path management
└─ Future: Download pre-built binaries

040-rbee-hive-worker-registry.feature    (9 scenarios) M1
├─ Local worker lifecycle on ONE rbee-hive
├─ Worker registry queries
└─ Happy + Error scenarios (EH-002a/b, EC8)

050-preflight-validation.feature         (? scenarios) M1 NEW! (CONSOLIDATED)
├─ SSH preflight (connection, auth, latency, binary exists)
├─ rbee-hive preflight (HTTP API, version, backend catalog, resources)
└─ Worker preflight (RAM, VRAM, disk, backend availability)

080-worker-rbee-lifecycle.feature        (11 scenarios) M0-M1
├─ worker-rbee daemon startup
├─ Registration and health checks
├─ Loading progress monitoring
└─ Happy + Error scenarios (EH-012a/b/c, EH-016a, EC7)

090-rbee-hive-lifecycle.feature          (7 scenarios) M1
├─ rbee-hive daemon startup/shutdown
├─ Worker health monitoring
├─ Idle timeout enforcement
└─ Happy + Error scenarios (EH-014a/b, EC10)

110-inference-execution.feature          (11 scenarios) M0-M1
├─ SSE token streaming
├─ Cancellation handling
├─ Worker busy scenarios
└─ Happy + Error scenarios (EH-018a, EH-013a/b, EH-003a, EC1, EC4, EC6, Gap-G12a/b/c)

120-input-validation.feature             (6 scenarios) M1
├─ Model reference format validation
├─ Backend and device validation
├─ API key authentication
└─ Error scenarios (EH-015a/b/c, EH-017a/b)

130-cli-commands.feature                 (9 scenarios) M1
├─ Installation (user/system paths)
├─ Configuration loading
├─ Basic CLI operations
└─ Happy scenarios

140-end-to-end-flows.feature             (2 scenarios) M1
├─ Cold start integration test
├─ Warm start integration test
└─ E2E integration scenarios only
```

### M2+ Features (Documented but Deferred)

```
030-queen-rbee-worker-registry.feature   (? scenarios) M2
└─ Global worker view (orchestrator)

100-queen-rbee-lifecycle.feature         (3 scenarios) M2
└─ queen-rbee daemon modes (orchestrator)

150-authentication.feature               (? scenarios) M3
└─ auth-min (timing-safe tokens)

160-audit-logging.feature                (? scenarios) M3
└─ Immutable audit trail (GDPR)

170-input-validation.feature             (? scenarios) M3
└─ Injection attack prevention

180-secrets-management.feature           (? scenarios) M3
└─ Secure credential handling

190-deadline-propagation.feature         (? scenarios) M3
└─ Resource exhaustion prevention

200-rhai-scheduler.feature               (? scenarios) M2
└─ Programmable routing logic

210-queue-management.feature             (? scenarios) M2
└─ Job queue and admission control
```

## Key Terminology

### Daemons (3 types)
- **worker-rbee** - Worker daemon that loads models and serves inference
- **rbee-hive** - Pool manager daemon that spawns and manages workers
- **queen-rbee** - Orchestrator daemon that coordinates rbee-hive instances

### Components
- **rbee-keeper** - CLI tool (ephemeral, exits after command)
- **rbee-hive** - Pool manager (persistent daemon)
- **queen-rbee** - Orchestrator (persistent daemon)
- **worker-rbee** - Worker process (persistent daemon)

## BDD Architecture Principles

1. **Each feature = ONE capability**
   - ✅ "Model Provisioning" is a feature
   - ❌ "Error Handling" is NOT a feature

2. **Error scenarios belong WITH their feature**
   - ✅ Model provisioning errors in 020-model-provisioning.feature
   - ❌ NOT in separate "error-handling" file

3. **Happy path is NOT a feature**
   - ✅ Happy scenarios within each feature
   - ✅ E2E integration tests in 110-end-to-end-flows.feature
   - ❌ NOT a separate "happy-path" file

4. **Specific daemon names**
   - ✅ "worker-rbee lifecycle" (specific)
   - ✅ "rbee-hive lifecycle" (specific)
   - ✅ "queen-rbee lifecycle" (specific)
   - ❌ "daemon lifecycle" (ambiguous)

## Running Tests

### Run all tests
```bash
cargo run --bin bdd-runner
```

### Run specific feature
```bash
LLORCH_BDD_FEATURE_PATH=tests/features/010-ssh-registry-management.feature cargo run --bin bdd-runner
```

### Run specific feature group
```bash
LLORCH_BDD_FEATURE_PATH=tests/features/0 cargo run --bin bdd-runner  # All 0XX features
```

## Verification Commands

### Count scenarios
```bash
for f in tests/features/*.feature; do echo "$f: $(grep -c '^  Scenario:' $f) scenarios"; done
```

### Check for duplicates
```bash
grep "^  Scenario:" tests/features/*.feature | sort | uniq -d
```

### Verify compilation
```bash
cargo check --bin bdd-runner
```

## Total Count
- **M0-M1 Files:** 12 (10 existing + 2 new)
- **M2+ Files:** 9 (documented but deferred)
- **Current Scenarios:** 91
- **Format:** XXY numbering (010, 020, 025, 050...)
- **Status:** ✅ M0-M1 structure defined, preflight consolidated, M2+ documented for future
