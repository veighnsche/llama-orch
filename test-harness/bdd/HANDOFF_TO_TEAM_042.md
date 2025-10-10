# HANDOFF TO TEAM-042: Implement BDD Step Definitions for TEST-001

**From:** TEAM-041  
**To:** TEAM-042  
**Date:** 2025-10-10  
**Status:** 🟡 STUBS COMPLETE - IMPLEMENTATION NEEDED

---

## What We Completed (TEAM-041)

### ✅ Specification & Architecture
1. **Added rbee-hive Registry module to TEST-001 spec** (`bin/.specs/.gherkin/test-001.md`)
   - SQLite database at `~/.rbee/beehives.db`
   - SSH connection details storage
   - Configuration flow via `rbee-keeper setup` commands
   - Registry lookup before SSH connections

2. **Updated TEST-001 feature file** (`test-harness/bdd/tests/features/test-001.feature`)
   - Added 7 new setup scenarios (@setup tag)
   - Updated happy path scenarios with registry integration
   - All scenarios use correct "rbee-hive" naming (not "BeeHive")

### ✅ BDD Infrastructure
3. **Created step definition stubs** for all registry scenarios:
   - `test-harness/bdd/src/steps/beehive_registry.rs` (NEW - 17 step definitions)
   - `test-harness/bdd/src/steps/happy_path.rs` (UPDATED - added 5 registry steps)
   - `test-harness/bdd/src/steps/background.rs` (UPDATED - added registry path step)

4. **Extended World state** (`test-harness/bdd/src/steps/world.rs`):
   - Added `registry_db_path: Option<String>`
   - Added `beehive_nodes: HashMap<String, BeehiveNode>`
   - Added `BeehiveNode` struct with all registry fields

5. **Build status:** ✅ `cargo check` passes - all stubs compile

---

## Your Mission (TEAM-042)

### 🎯 Primary Goal
**Implement all step definition stubs in `test-harness/bdd/` until the entire TEST-001 feature is executable.**

### 📋 Task Breakdown

#### **STEP 1: Survey Existing Implementations**
Look at what already exists in `bin/` binaries to understand the architecture:

```bash
# Check existing binaries
ls -la bin/*/src/

# Key binaries to study:
# - bin/queen-rbee/     (orchestrator - will have registry logic)
# - bin/rbee-keeper/    (CLI tool - configuration commands)
# - bin/llm-worker-rbee/ (worker - inference logic)
```

**What to look for:**
- How does `rbee-keeper` parse CLI commands?
- Does `queen-rbee` have any registry/SSH logic yet?
- How do binaries communicate (HTTP, SSH)?
- What data structures exist that match `World` state?

#### **STEP 2: Identify Implementation Gaps**
Compare the feature file requirements with what exists in `bin/`:

**Registry Operations (likely MISSING):**
- [ ] SQLite database creation/schema for `beehives` table
- [ ] SSH connection validation logic
- [ ] Registry CRUD operations (add, remove, list, query)
- [ ] `rbee-keeper setup` subcommands

**Orchestration Flow (may exist partially):**
- [ ] queen-rbee → rbee-hive SSH connection
- [ ] Model catalog operations
- [ ] Worker spawning and registration
- [ ] Inference request routing

**Narration System (check if exists):**
- [ ] `narrate()` function implementation
- [ ] SSE streaming for progress updates
- [ ] stdout/stderr routing

#### **STEP 3: Implement Step Definitions**
Work through step files in this order:

1. **`beehive_registry.rs`** (17 stubs) - Registry setup scenarios
   - Implement SQLite operations
   - Implement SSH validation
   - Wire up to `queen-rbee` HTTP API

2. **`happy_path.rs`** (registry integration steps)
   - Implement registry lookup before SSH
   - Wire up SSH connection establishment
   - Update `last_connected_unix` tracking

3. **Other step files** (as needed):
   - `model_provisioning.rs` - Model download/catalog
   - `worker_startup.rs` - Worker spawning
   - `inference_execution.rs` - Inference requests
   - `lifecycle.rs` - Shutdown cascading
   - `edge_cases.rs` - Error scenarios

#### **STEP 4: Fill Implementation Gaps in `bin/`**
If you find missing functionality in the binaries, implement it:

**Example: If `queen-rbee` doesn't have registry module:**
```bash
# Create the module
mkdir -p bin/queen-rbee/src/registry
touch bin/queen-rbee/src/registry/mod.rs
touch bin/queen-rbee/src/registry/db.rs
touch bin/queen-rbee/src/registry/ssh.rs
```

**Example: If `rbee-keeper` doesn't have setup commands:**
```rust
// Add to bin/rbee-keeper/src/cli.rs
#[derive(Subcommand)]
enum Commands {
    Infer { /* ... */ },
    Setup {
        #[command(subcommand)]
        command: SetupCommands,
    },
}

#[derive(Subcommand)]
enum SetupCommands {
    AddNode { /* ... */ },
    Install { /* ... */ },
    ListNodes,
    RemoveNode { /* ... */ },
}
```

#### **STEP 5: Run Tests Incrementally**
Test as you implement:

```bash
# Run specific scenario
LLORCH_BDD_FEATURE_PATH=tests/features/test-001.feature \
  cargo run --bin bdd-runner -- --tags @setup

# Run all TEST-001 scenarios
LLORCH_BDD_FEATURE_PATH=tests/features/test-001.feature \
  cargo run --bin bdd-runner

# Run with specific scenario name
LLORCH_BDD_FEATURE_PATH=tests/features/test-001.feature \
  cargo run --bin bdd-runner -- --name "Add remote rbee-hive node"
```

#### **STEP 6: Document What You Built**
Update the handoff document with:
- What you implemented
- What still needs work
- Any architectural decisions you made
- Known issues or limitations

---

## Key Files Reference

### Specifications
- `bin/.specs/.gherkin/test-001.md` - Full flow documentation with narration
- `test-harness/bdd/tests/features/test-001.feature` - Gherkin scenarios

### Step Definitions (STUBS - NEED IMPLEMENTATION)
- `test-harness/bdd/src/steps/beehive_registry.rs` - Registry setup (17 stubs)
- `test-harness/bdd/src/steps/happy_path.rs` - Happy path + registry integration
- `test-harness/bdd/src/steps/background.rs` - Background steps
- `test-harness/bdd/src/steps/world.rs` - World state

### Binaries (MAY NEED IMPLEMENTATION)
- `bin/queen-rbee/` - Orchestrator (needs registry module)
- `bin/rbee-keeper/` - CLI tool (needs setup commands)
- `bin/llm-worker-rbee/` - Worker (may be partially implemented)

### Supporting Crates
- `contracts/api-types/` - Shared types
- `contracts/config-schema/` - Configuration
- `libs/proof-bundle/` - Test proof bundles

---

## Acceptance Criteria

### ✅ Definition of Done
- [ ] All step definitions in `beehive_registry.rs` are implemented (not stubs)
- [ ] All step definitions in `happy_path.rs` are implemented (not stubs)
- [ ] Registry scenarios pass: `cargo run --bin bdd-runner -- --tags @setup`
- [ ] Happy path scenarios pass: `cargo run --bin bdd-runner -- --tags @happy`
- [ ] Full TEST-001 passes: `LLORCH_BDD_FEATURE_PATH=tests/features/test-001.feature cargo run --bin bdd-runner`
- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo fmt --check` passes
- [ ] All implementation gaps in `bin/` are filled

### 🎯 Success Metrics
- **0 pending steps** when running TEST-001
- **0 undefined steps** when running TEST-001
- **All scenarios pass** (green checkmarks)
- **Proof bundles generated** in `test-harness/bdd/.proof_bundle/`

---

## Tips & Gotchas

### 🔍 Finding Existing Code
```bash
# Search for similar patterns
rg "SQLite" bin/
rg "ssh" bin/ -i
rg "registry" bin/ -i
rg "narrate" bin/

# Check for HTTP clients
rg "reqwest" bin/
rg "axum" bin/
```

### 🧪 Testing Strategy
1. **Start with Background steps** - These set up the world state
2. **Implement Given steps next** - These set preconditions
3. **Then When steps** - These trigger actions
4. **Finally Then steps** - These verify outcomes

### 🚨 Common Pitfalls
- **Don't hardcode paths** - Use `shellexpand::tilde()` for `~/.rbee/`
- **Use World state** - Don't create global state
- **Mock external calls** - SSH, HTTP should be mockable
- **Follow existing patterns** - Look at `registry.rs`, `lifecycle.rs` for examples

### 📚 Useful References
- Cucumber Rust docs: https://cucumber-rs.github.io/cucumber/current/
- BDD runner: `test-harness/bdd/src/main.rs`
- Existing step patterns: `test-harness/bdd/src/steps/registry.rs`

---

## Questions?

If you get stuck:
1. Check `test-harness/bdd/src/steps/registry.rs` for worker registry patterns
2. Check `test-harness/bdd/src/steps/lifecycle.rs` for complex flows
3. Look at how other teams implemented similar steps
4. Document blockers in your handoff to TEAM-043

---

## Handoff Checklist

When you're done, create `HANDOFF_TO_TEAM_043.md` with:
- [x] What you implemented
- [x] Test results (pass/fail counts)
- [x] Known issues or limitations
- [x] What TEAM-043 should focus on next

---

**Good luck, TEAM-042! 🚀**

**Remember:** The goal is a **fully executable TEST-001** that validates the entire cross-node inference flow from cold start to completion.
