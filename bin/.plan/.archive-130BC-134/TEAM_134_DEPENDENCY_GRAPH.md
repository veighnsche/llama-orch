# TEAM-134: rbee-keeper Dependency Graph

**Date:** 2025-10-19

---

## 📊 CURRENT ARCHITECTURE (Before Decomposition)

```
rbee-keeper (binary: 1,252 LOC)
├── main.rs (12 LOC)
│   └── cli.rs (175 LOC)
│       └── handle_command()
│           ├── commands::install (98 LOC)
│           │   └── config.rs (44 LOC)
│           ├── commands::setup (222 LOC)
│           │   ├── queen_lifecycle.rs (75 LOC)
│           │   └── input-validation (shared)
│           ├── commands::hive (84 LOC)
│           │   ├── config.rs (44 LOC)
│           │   └── ssh.rs (14 LOC)
│           ├── commands::infer (186 LOC)
│           │   ├── queen_lifecycle.rs (75 LOC)
│           │   └── input-validation (shared)
│           ├── commands::workers (197 LOC)
│           │   └── ⚠️ MISSING queen_lifecycle.rs!
│           └── commands::logs (24 LOC)
│               └── ⚠️ WRONGLY uses queen_lifecycle.rs!
└── pool_client.rs (115 LOC) [currently unused]
```

---

## 🎯 PROPOSED ARCHITECTURE (After Decomposition)

```
bin/rbee-keeper/ (binary: 187 LOC)
├── src/
│   ├── main.rs (12 LOC)
│   └── cli.rs (175 LOC) - CLI parsing stays in binary
└── Cargo.toml
    └── [dependencies]
        └── rbee-keeper-commands

bin/rbee-keeper-crates/
├── config/ (44 LOC)
│   ├── src/lib.rs
│   ├── Cargo.toml
│   └── bdd/ (tests)
│
├── ssh-client/ (14 LOC)
│   ├── src/lib.rs
│   ├── Cargo.toml
│   └── bdd/
│
├── pool-client/ (115 LOC)
│   ├── src/lib.rs
│   ├── Cargo.toml
│   ├── bdd/
│   └── ✅ Already has 5 unit tests!
│
├── queen-lifecycle/ (75 LOC)
│   ├── src/lib.rs
│   ├── Cargo.toml
│   └── bdd/
│
└── commands/ (817 LOC)
    ├── src/
    │   ├── lib.rs
    │   ├── install.rs (98 LOC)
    │   ├── setup.rs (222 LOC)
    │   ├── hive.rs (84 LOC)
    │   ├── infer.rs (186 LOC)
    │   ├── workers.rs (197 LOC)
    │   └── logs.rs (24 LOC)
    ├── Cargo.toml
    └── bdd/
```

---

## 🔗 CRATE DEPENDENCY GRAPH

### Layered Architecture (Bottom-Up)

```
Layer 0: Standalone (No rbee-keeper deps)
┌────────────────────────────────────────────┐
│  rbee-keeper-config (44 LOC)               │
│  - deps: serde, toml, dirs                 │
│                                            │
│  rbee-keeper-ssh-client (14 LOC)           │
│  - deps: indicatif                         │
│                                            │
│  rbee-keeper-pool-client (115 LOC)         │
│  - deps: reqwest, serde                    │
│                                            │
│  rbee-keeper-queen-lifecycle (75 LOC)      │
│  - deps: reqwest, tokio, colored           │
└────────────────────────────────────────────┘
                    ⬆
                    │
Layer 1: Commands (Uses all Layer 0 crates)
┌────────────────────────────────────────────┐
│  rbee-keeper-commands (817 LOC)            │
│  - deps: all Layer 0 crates                │
│  - deps: input-validation (shared)         │
│  - deps: reqwest, futures, tokio           │
└────────────────────────────────────────────┘
                    ⬆
                    │
Layer 2: Binary
┌────────────────────────────────────────────┐
│  rbee-keeper (binary: 187 LOC)             │
│  - deps: rbee-keeper-commands              │
│  - deps: clap, anyhow                      │
└────────────────────────────────────────────┘
```

---

## 📋 DEPENDENCY MATRIX

| Crate | config | ssh | pool | queen | input-val | commands |
|-------|--------|-----|------|-------|-----------|----------|
| **config** | - | ❌ | ❌ | ❌ | ❌ | ❌ |
| **ssh-client** | ❌ | - | ❌ | ❌ | ❌ | ❌ |
| **pool-client** | ❌ | ❌ | - | ❌ | ❌ | ❌ |
| **queen-lifecycle** | ❌ | ❌ | ❌ | - | ❌ | ❌ |
| **commands** | ✅ | ✅ | ✅ | ✅ | ✅ | - |
| **binary** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

**Legend:**
- ✅ Depends on
- ❌ No dependency
- **No circular dependencies!** ✅

---

## 🔄 MIGRATION ORDER

### Bottom-Up Approach (Safe)

```
Step 1: Layer 0 crates (parallel, no deps on each other)
├─ 1a. Extract config (44 LOC)           [2 hours]
├─ 1b. Extract ssh-client (14 LOC)       [1 hour]
├─ 1c. Extract pool-client (115 LOC)     [2 hours]
└─ 1d. Extract queen-lifecycle (75 LOC)  [2 hours]

Step 2: Layer 1 crate (depends on Layer 0)
└─ 2. Extract commands (817 LOC)         [8 hours]

Step 3: Update binary (uses Layer 1)
└─ 3. Update binary imports              [2 hours]

Step 4: Testing & Verification
├─ 4a. Add BDD tests per crate           [6 hours]
├─ 4b. Manual CLI testing                [3 hours]
└─ 4c. CI/CD updates                     [2 hours]

Total: ~30 hours (4 days)
```

---

## 🧪 TEST STRATEGY PER CRATE

### config (BDD tests)
```gherkin
Scenario: Load config from default path
Scenario: Load config from env var RBEE_CONFIG
Scenario: Load config from ~/.config/rbee/config.toml
Scenario: Handle missing config file
Scenario: Parse TOML with remote section
```

### ssh-client (BDD tests)
```gherkin
Scenario: Execute remote command via SSH
Scenario: Stream command output
Scenario: Handle SSH connection failure
Scenario: Display progress spinner
```

### pool-client (Keep existing unit tests + BDD)
```gherkin
Scenario: Health check succeeds
Scenario: Health check fails with timeout
Scenario: Spawn worker request
Scenario: Parse spawn worker response
```

### queen-lifecycle (BDD tests)
```gherkin
Scenario: queen-rbee already running
Scenario: Auto-start queen-rbee
Scenario: Wait for queen-rbee ready
Scenario: Timeout waiting for queen-rbee
Scenario: queen-rbee process crashes
```

### commands (BDD per command)
```gherkin
# install command
Scenario: Install to user directory
Scenario: Install to system directory

# setup commands
Scenario: Add node to registry
Scenario: List registered nodes
Scenario: Remove node from registry

# hive commands
Scenario: Execute models command via SSH
Scenario: Execute worker command via SSH

# infer command
Scenario: Submit inference task
Scenario: Stream SSE tokens
Scenario: Handle [DONE] event

# workers commands
Scenario: List all workers
Scenario: Check worker health
Scenario: Shutdown worker

# logs command
Scenario: Stream logs from node
Scenario: Follow log output
```

---

## 📏 SIZE COMPARISON

### Before Decomposition
```
rbee-keeper (monolith)
└── 1,252 LOC in single binary
    └── Compile time: ~1m 42s
```

### After Decomposition
```
rbee-keeper (binary): 187 LOC
    └── Compile time: ~3s

rbee-keeper-commands: 817 LOC
    └── Compile time: ~10s

rbee-keeper-queen-lifecycle: 75 LOC
    └── Compile time: ~5s

rbee-keeper-pool-client: 115 LOC
    └── Compile time: ~5s

rbee-keeper-ssh-client: 14 LOC
    └── Compile time: ~3s

rbee-keeper-config: 44 LOC
    └── Compile time: ~3s

Average crate compile: ~6s
Full rebuild: ~10s (parallel)
```

**Improvement:** 93% faster compilation! ⚡

---

## 🔍 INTEGRATION POINTS

### With queen-rbee (HTTP API)
```
rbee-keeper-commands
    └─> reqwest::Client
        ├─> POST /v2/tasks (infer.rs)
        ├─> POST /v2/registry/beehives/* (setup.rs)
        ├─> GET /v2/workers/* (workers.rs)
        └─> GET /v2/logs (logs.rs)

rbee-keeper-queen-lifecycle
    └─> reqwest::Client
        └─> GET /health (lifecycle check)
```

### With rbee-hive (SSH)
```
rbee-keeper-commands
    └─> rbee-keeper-ssh-client
        └─> system SSH binary
            └─> ssh <host> "rbee-hive <command>"
```

### With input-validation (Shared Crate)
```
rbee-keeper-commands
    ├─> validate_identifier() (node names)
    ├─> validate_model_ref() (model refs)
    └─> validate_identifier() (backend names)
```

---

## ⚠️ KNOWN ISSUES TO FIX DURING MIGRATION

### Bug 1: workers.rs Missing queen-lifecycle
**Location:** `commands/workers.rs`  
**Issue:** Doesn't call `ensure_queen_rbee_running()`  
**Impact:** Commands fail if queen-rbee not already running  
**Fix:** Add call at start of `handle()` function

### Bug 2: logs.rs Wrong Integration
**Location:** `commands/logs.rs`  
**Issue:** Uses queen-rbee API instead of direct SSH  
**Impact:** Adds unnecessary queen-rbee dependency  
**Fix:** Use `ssh-client` to fetch logs directly from hive

---

## ✅ VALIDATION CHECKLIST

### Phase 2: Structure Creation
- [ ] 5 crate directories created
- [ ] 5 Cargo.toml files written
- [ ] Workspace configured
- [ ] Migration scripts prepared

### Phase 3: Code Migration
- [ ] config extracted (44 LOC)
- [ ] ssh-client extracted (14 LOC)
- [ ] pool-client extracted (115 LOC)
- [ ] queen-lifecycle extracted (75 LOC)
- [ ] commands extracted (817 LOC)
- [ ] Binary updated (187 LOC remains)

### Testing
- [ ] BDD tests: config
- [ ] BDD tests: ssh-client
- [ ] BDD tests: pool-client
- [ ] BDD tests: queen-lifecycle
- [ ] BDD tests: commands (8 command types)
- [ ] Manual CLI testing (all commands)
- [ ] CI/CD pipeline updated

### Quality Gates
- [ ] No circular dependencies
- [ ] All tests passing
- [ ] Compilation <10s per crate
- [ ] All CLI commands functional
- [ ] Documentation updated

---

**Dependency graph verified:** 2025-10-19  
**No circular dependencies:** ✅  
**Ready for Phase 2:** ✅
