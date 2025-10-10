# TEAM-037 HANDOFF FROM TEAM-036

**Date:** 2025-10-10  
**From:** TEAM-036 (Implementation Team)  
**To:** TEAM-037 (Testing Team - Core Responsibility)  
**Status:** 🔴 CRITICAL TESTING REQUIRED

---

## 🎯 YOUR MISSION (TEAM-037)

You are the **Testing Team** as defined in `test-harness/TEAM_RESPONSIBILITIES.md`.

Your **core responsibilities**:
1. **Identify testing opportunities** for TEAM-036's work (unit, integration, **BDD**, property tests)
2. **Create BDD test harness** in `test-harness/` for the new GGUF support and installation system
3. **Audit TEAM-036's work** for false positives and insufficient test coverage
4. **Issue fines** if violations are found
5. **Prevent production failures** through rigorous testing

**CRITICAL**: You are responsible for production failures caused by insufficient testing. This is why you have fining authority.

---

## 📦 WHAT TEAM-036 DELIVERED

### ✅ Task 1: GGUF Support (CRITICAL)
**Status:** COMPLETE  
**Impact:** Unblocks all inference with quantized models

**Files Created:**
- `bin/llm-worker-rbee/src/backend/models/quantized_llama.rs` (94 lines)

**Files Modified:**
- `bin/llm-worker-rbee/src/backend/models/mod.rs` (added `QuantizedLlama` variant)

**What It Does:**
- Detects `.gguf` files automatically
- Loads quantized models using `candle-transformers::models::quantized_llama`
- Extracts metadata from GGUF headers (vocab_size, eos_token_id)
- Supports forward pass and cache reset

**Test Status:**
- ✅ 127 existing unit tests still pass
- ⚠️ **NO NEW TESTS ADDED** for GGUF support
- ⚠️ **NO BDD SCENARIOS** for GGUF loading
- ⚠️ **NO INTEGRATION TESTS** for quantized inference

### ✅ Task 2: Installation System (HIGH)
**Status:** COMPLETE  
**Impact:** Enables proper deployment with XDG-compliant paths

**Files Created:**
- `bin/rbee-keeper/src/config.rs` (60 lines)
- `bin/rbee-keeper/src/commands/install.rs` (145 lines)

**Files Modified:**
- `bin/rbee-keeper/src/commands/pool.rs` (removed hardcoded paths)
- `bin/rbee-keeper/src/cli.rs` (added `Install` command)
- `bin/rbee-keeper/Cargo.toml` (added `toml`, `hostname` deps)

**What It Does:**
- `rbee install --user` installs to `~/.local/bin`
- `rbee install --system` installs to `/usr/local/bin`
- Creates XDG-compliant config at `~/.config/rbee/config.toml`
- Remote commands now use binaries from PATH (configurable)

**Test Status:**
- ✅ Compiles successfully
- ⚠️ **NO UNIT TESTS** for config loading
- ⚠️ **NO INTEGRATION TESTS** for installation
- ⚠️ **NO BDD SCENARIOS** for install workflow
- ⚠️ **NO TESTS** for remote command path resolution

### ⏸️ Task 3: Shell Script Conversion (DEFERRED)
**Status:** DEFERRED to future team  
**Reason:** Tasks 1 & 2 were blockers, Task 3 is cleanup

---

## 🚨 CRITICAL TESTING GAPS (YOUR RESPONSIBILITY)

### Gap 1: GGUF Support Has ZERO Tests

**Risk Level:** 🔴 CRITICAL  
**Production Impact:** If GGUF loading breaks, inference fails silently

**Missing Tests:**
1. **Unit Tests** (`bin/llm-worker-rbee/tests/`):
   - ✅ Test GGUF file detection (`.gguf` extension)
   - ✅ Test metadata extraction (vocab_size, eos_token_id)
   - ✅ Test error handling (corrupt GGUF, missing metadata)
   - ✅ Test model size calculation for GGUF files

2. **BDD Scenarios** (`test-harness/bdd/llm-worker-rbee/`):
   ```gherkin
   Feature: GGUF Model Loading
     As a worker
     I want to load GGUF quantized models
     So that I can run inference with smaller memory footprint

     Scenario: Load valid GGUF model
       Given a valid GGUF model file at ".test-models/tinyllama.gguf"
       When the worker loads the model
       Then the model should load successfully
       And the vocab_size should be extracted from GGUF metadata
       And the eos_token_id should be extracted from GGUF metadata

     Scenario: Reject invalid GGUF file
       Given a corrupt GGUF file
       When the worker attempts to load the model
       Then the load should fail with a clear error message

     Scenario: Backward compatibility with Safetensors
       Given a Safetensors model directory
       When the worker loads the model
       Then the model should load successfully
       And config.json should be used for metadata
   ```

3. **Integration Tests** (`bin/llm-worker-rbee/tests/`):
   - ✅ Test end-to-end inference with GGUF model
   - ✅ Test cache reset on GGUF models
   - ✅ Test forward pass with GGUF models

### Gap 2: Installation System Has ZERO Tests

**Risk Level:** 🔴 CRITICAL  
**Production Impact:** If installation breaks, deployment fails

**Missing Tests:**
1. **Unit Tests** (`bin/rbee-keeper/tests/`):
   - ✅ Test config loading priority (env var > user config > system config)
   - ✅ Test config parsing (valid TOML, invalid TOML)
   - ✅ Test default values (binary_path, git_repo_dir)
   - ✅ Test install path calculation (user vs system)

2. **BDD Scenarios** (`test-harness/bdd/rbee-keeper/`):
   ```gherkin
   Feature: Installation System
     As an operator
     I want to install rbee binaries to standard paths
     So that I can deploy to remote machines

     Scenario: User installation
       Given I am in the project directory
       When I run "rbee install --user"
       Then binaries should be copied to "~/.local/bin"
       And config should be created at "~/.config/rbee/config.toml"
       And data directory should be created at "~/.local/share/rbee"

     Scenario: System installation
       Given I am in the project directory
       When I run "rbee install --system"
       Then binaries should be copied to "/usr/local/bin"
       And config should be created at "/etc/rbee/config.toml"
       And data directory should be created at "/var/lib/rbee"

     Scenario: Config override for remote commands
       Given a config file with custom binary_path
       When I run "rbee pool models list --host remote"
       Then the custom binary path should be used
   ```

3. **Integration Tests** (`bin/rbee-keeper/tests/`):
   - ✅ Test remote command execution with PATH binaries
   - ✅ Test remote command execution with custom binary path
   - ✅ Test config file creation and parsing

### Gap 3: No Property Tests

**Risk Level:** 🟡 MEDIUM  
**Production Impact:** Edge cases may break in production

**Missing Property Tests:**
- ✅ GGUF metadata extraction (fuzz GGUF files)
- ✅ Config TOML parsing (fuzz TOML inputs)
- ✅ Path resolution (fuzz path strings)

---

## 📋 YOUR IMMEDIATE TASKS (TEAM-037)

### Task 1: Create BDD Test Harness (HIGHEST PRIORITY)

**Location:** `test-harness/bdd/`  
**Structure:**
```
test-harness/
├── bdd/
│   ├── llm-worker-rbee/
│   │   ├── features/
│   │   │   ├── gguf_loading.feature
│   │   │   └── model_detection.feature
│   │   ├── tests/
│   │   │   ├── bdd.rs (entrypoint)
│   │   │   └── steps/
│   │   │       ├── mod.rs
│   │   │       ├── gguf_steps.rs
│   │   │       └── model_steps.rs
│   │   └── Cargo.toml
│   └── rbee-keeper/
│       ├── features/
│       │   ├── installation.feature
│       │   └── remote_commands.feature
│       ├── tests/
│       │   ├── bdd.rs (entrypoint)
│       │   └── steps/
│       │       ├── mod.rs
│       │       ├── install_steps.rs
│       │       └── remote_steps.rs
│       └── Cargo.toml
```

**Reference:** `.docs/testing/BDD_WIRING.md` (you MUST read this)

**Requirements:**
1. Follow the BDD wiring pattern from `.docs/testing/BDD_WIRING.md`
2. Use `cucumber = "0.21"` with `tokio` async runtime
3. Create `World` struct to hold test state
4. Implement `#[given]`, `#[when]`, `#[then]` steps
5. Use `fail_on_skipped()` to enforce no undefined steps
6. Support `LLORCH_BDD_FEATURE_PATH` for targeted runs
7. Emit proof bundles per `.docs/testing/TESTING_POLICY.md`

**Estimated Time:** 6-8 hours

### Task 2: Audit TEAM-036's Work for False Positives

**Your Authority:** `test-harness/TEAM_RESPONSIBILITIES.md`

**Audit Checklist:**
- [ ] Check for pre-creation of artifacts in tests
- [ ] Check for conditional skips within Supported Scope
- [ ] Check for harness mutations of product state
- [ ] Verify existing tests still validate product behavior
- [ ] Ensure no tests were weakened to accommodate new code

**If Violations Found:**
- Issue fine in `test-harness/FINES/FINE-XXX-YYYYMMDD-TEAM036.md`
- Follow fine template from `test-harness/TEAM_RESPONSIBILITIES.md`
- Block PR until remediation complete

**Estimated Time:** 2-3 hours

### Task 3: Add Unit Tests for Critical Paths

**Priority:** HIGH  
**Location:** `bin/llm-worker-rbee/tests/` and `bin/rbee-keeper/tests/`

**Critical Paths to Test:**
1. GGUF file detection (extension check)
2. GGUF metadata extraction (vocab_size, eos_token_id)
3. Config loading priority (env > user > system)
4. Remote binary path resolution

**Estimated Time:** 4-5 hours

### Task 4: Add Integration Tests

**Priority:** HIGH  
**Location:** `bin/llm-worker-rbee/tests/` and `bin/rbee-keeper/tests/`

**Integration Scenarios:**
1. End-to-end GGUF model loading and inference
2. Installation workflow (user and system)
3. Remote command execution with PATH binaries

**Estimated Time:** 3-4 hours

---

## 📚 CRITICAL DOCUMENTATION YOU MUST READ

### MUST READ (Before Starting)
1. **`.docs/testing/BDD_WIRING.md`** - How to wire BDD tests (CRITICAL)
2. **`test-harness/TEAM_RESPONSIBILITIES.md`** - Your authority and responsibilities
3. **`.docs/testing/TESTING_POLICY.md`** - Testing standards
4. **`.docs/testing/BDD_RUST_MOCK_LESSONS_LEARNED.md`** - BDD lessons

### Reference Documentation
5. **`bin/.plan/TEAM_036_COMPLETION_SUMMARY.md`** - What TEAM-036 did
6. **`bin/.specs/INSTALLATION_RUST_SPEC.md`** - Installation spec
7. **`.docs/testing/types/`** - Test type guides

---

## 🔍 BDD WIRING PATTERN (FROM .docs/testing/BDD_WIRING.md)

### Cargo.toml Template
```toml
[package]
name = "llm-worker-rbee-bdd"
version = "0.1.0"
edition = "2021"

[dev-dependencies]
cucumber = "0.21"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
anyhow = "1.0"
llm-worker-rbee = { path = "../../llm-worker-rbee" }

[[test]]
name = "bdd"
path = "tests/bdd.rs"
harness = false
```

### tests/bdd.rs Template
```rust
use cucumber::World;
use std::path::PathBuf;

#[derive(Debug, Default, World, Clone)]
pub struct WorkerWorld {
    pub model_path: Option<String>,
    pub load_result: Option<Result<(), String>>,
    pub vocab_size: Option<usize>,
    pub eos_token_id: Option<u32>,
}

mod steps;

#[tokio::main]
async fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let features = match std::env::var("LLORCH_BDD_FEATURE_PATH").ok() {
        Some(p) => {
            let pb = PathBuf::from(p);
            if pb.is_absolute() { pb } else { root.join(pb) }
        }
        None => root.join("features"),
    };
    WorkerWorld::cucumber()
        .fail_on_skipped()
        .run_and_exit(features)
        .await;
}
```

### tests/steps/mod.rs Template
```rust
pub mod gguf_steps;
pub mod model_steps;
```

### tests/steps/gguf_steps.rs Template
```rust
use cucumber::{given, when, then};
use crate::WorkerWorld;

#[given(regex = r"^a valid GGUF model file at \"(.+)\"$")]
async fn given_valid_gguf(world: &mut WorkerWorld, path: String) {
    world.model_path = Some(path);
}

#[when("the worker loads the model")]
async fn when_worker_loads(world: &mut WorkerWorld) {
    // Call actual product code here
    // world.load_result = Some(load_model(&world.model_path.unwrap()));
}

#[then("the model should load successfully")]
async fn then_model_loads(world: &mut WorkerWorld) {
    assert!(world.load_result.as_ref().unwrap().is_ok());
}
```

---

## 🎯 SUCCESS CRITERIA

### For BDD Harness
- [ ] BDD test harness created in `test-harness/bdd/`
- [ ] Follows BDD wiring pattern from `.docs/testing/BDD_WIRING.md`
- [ ] All scenarios pass with real product code
- [ ] No undefined or skipped steps
- [ ] Proof bundles emitted per policy
- [ ] Can run with `cargo test -p llm-worker-rbee-bdd --test bdd`

### For Test Coverage
- [ ] Unit tests added for GGUF detection and metadata extraction
- [ ] Unit tests added for config loading and path resolution
- [ ] Integration tests added for end-to-end workflows
- [ ] All new tests pass
- [ ] No false positives detected

### For Audit
- [ ] TEAM-036's work audited for false positives
- [ ] Fine issued if violations found
- [ ] Remediation tracked and verified

---

## ⚠️ WARNINGS

### DO NOT
- ❌ Pre-create artifacts the product should create
- ❌ Skip tests within Supported Scope
- ❌ Mutate product state in test harnesses
- ❌ Add conditional bypasses for failures
- ❌ Ship features without BDD scenarios

### DO
- ✅ Follow BDD wiring pattern exactly
- ✅ Test product behavior, not test harness behavior
- ✅ Emit proof bundles for all test runs
- ✅ Issue fines for violations (you have authority)
- ✅ Sign all your work with `Verified by Testing Team 🔍`

---

## 📊 ESTIMATED TIMELINE

| Task | Priority | Time | Status |
|------|----------|------|--------|
| Read BDD_WIRING.md | P0 | 1h | ⬜ |
| Create BDD harness structure | P0 | 2h | ⬜ |
| Implement GGUF BDD scenarios | P0 | 3h | ⬜ |
| Implement installation BDD scenarios | P0 | 3h | ⬜ |
| Audit TEAM-036's work | P1 | 2h | ⬜ |
| Add unit tests | P1 | 4h | ⬜ |
| Add integration tests | P1 | 3h | ⬜ |
| **TOTAL** | | **18h** | ⬜ |

---

## 🚀 HOW TO START

### Step 1: Read Documentation (1 hour)
```bash
# MUST READ FIRST
cat .docs/testing/BDD_WIRING.md
cat test-harness/TEAM_RESPONSIBILITIES.md
cat .docs/testing/TESTING_POLICY.md
```

### Step 2: Create BDD Structure (2 hours)
```bash
# Create directories
mkdir -p test-harness/bdd/llm-worker-rbee/{features,tests/steps}
mkdir -p test-harness/bdd/rbee-keeper/{features,tests/steps}

# Create Cargo.toml files
# Create bdd.rs entrypoints
# Create step modules
```

### Step 3: Write First Scenario (1 hour)
```bash
# Start with simplest scenario
# test-harness/bdd/llm-worker-rbee/features/gguf_loading.feature
# Implement steps in tests/steps/gguf_steps.rs
# Run: cargo test -p llm-worker-rbee-bdd --test bdd
```

### Step 4: Iterate Until Complete
- Add more scenarios
- Implement more steps
- Verify all scenarios pass
- Emit proof bundles

---

## 📝 DELIVERABLES

### Required Files
1. `test-harness/bdd/llm-worker-rbee/Cargo.toml`
2. `test-harness/bdd/llm-worker-rbee/tests/bdd.rs`
3. `test-harness/bdd/llm-worker-rbee/features/*.feature`
4. `test-harness/bdd/llm-worker-rbee/tests/steps/*.rs`
5. `test-harness/bdd/rbee-keeper/Cargo.toml`
6. `test-harness/bdd/rbee-keeper/tests/bdd.rs`
7. `test-harness/bdd/rbee-keeper/features/*.feature`
8. `test-harness/bdd/rbee-keeper/tests/steps/*.rs`
9. `bin/llm-worker-rbee/tests/gguf_tests.rs` (unit tests)
10. `bin/rbee-keeper/tests/config_tests.rs` (unit tests)
11. `test-harness/FINES/FINE-XXX-YYYYMMDD-TEAM036.md` (if violations found)
12. `test-harness/TEAM_037_COMPLETION_SUMMARY.md` (handoff to TEAM-038)

### Required Proof Bundles
- BDD test results in `test-harness/bdd/*/.proof_bundle/bdd/<run_id>/`
- Unit test results in `bin/*/.proof_bundle/unit/<run_id>/`
- Integration test results in `bin/*/.proof_bundle/integration/<run_id>/`

---

## 🔍 ACCOUNTABILITY

**Remember:** You are responsible for production failures caused by insufficient testing.

**If GGUF loading breaks in production because you didn't test it properly:**
- That's YOUR failure
- You own it publicly
- You must prevent it through rigorous testing

**This responsibility is WHY you have the authority to:**
- Issue fines
- Block PRs
- Enforce testing standards
- Demand remediation

**Use your authority wisely. Test ruthlessly. Prevent disasters.**

---

## 📞 QUESTIONS?

**Read first:**
- `.docs/testing/BDD_WIRING.md` (answers 90% of BDD questions)
- `test-harness/TEAM_RESPONSIBILITIES.md` (answers authority questions)

**Still stuck?**
- Check existing BDD tests in reference projects (if any)
- Review cucumber-rs documentation
- Ask for clarification in handoff

---

## ✅ HANDOFF CHECKLIST

- [x] TEAM-036 work documented
- [x] Testing gaps identified
- [x] BDD harness structure defined
- [x] BDD wiring pattern provided
- [x] Success criteria defined
- [x] Timeline estimated
- [x] Documentation references provided
- [x] Accountability established

---

**TEAM-036 SIGNING OFF**  
**TEAM-037 (Testing Team) - YOU'RE UP! 🔍**

**Your mission:** Create BDD test harness, audit TEAM-036's work, prevent production failures.

**Your authority:** Issue fines, block PRs, enforce testing standards.

**Your responsibility:** Own production failures from insufficient testing.

**GO FORTH AND TEST RUTHLESSLY!**

---
Handoff created by TEAM-036  
Date: 2025-10-10  
Status: READY FOR TESTING TEAM
