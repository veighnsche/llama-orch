# Commit 6592850 - Detailed File Changes

**Commit:** 6592850  
**Date:** 2025-10-10 14:12:16 +0200  
**Teams:** TEAM-036 (Implementation), TEAM-037 (Testing)  
**Message:** chore: remove unused test scripts and architecture docs from bin/.specs

---

## File-by-File Change Documentation

This document provides detailed change documentation for each file modified in commit 6592850.

---

## 📋 Planning Documents

### `bin/.plan/TEAM_036_COMPLETION_SUMMARY.md` [NEW +398 lines]
**Created by:** TEAM-036  
**Purpose:** Completion report for TEAM-036's work

**What was added:**
- Summary of all 3 tasks (GGUF support, installation system, shell script conversion)
- Detailed implementation notes for GGUF support
- Detailed implementation notes for installation system
- Rationale for deferring shell script conversion
- Test results (127 tests passing)
- Handoff notes to TEAM-037

**Key sections:**
1. Task 1: GGUF Support (CRITICAL) - Complete
2. Task 2: Installation Paths (HIGH) - Complete
3. Task 3: Shell Script Conversion (DEFERRED)
4. Testing Status
5. Next Steps

---

### `bin/.plan/TEAM_037_HANDOFF_FROM_TEAM_036.md` [NEW +547 lines]
**Created by:** TEAM-036  
**Purpose:** Handoff document to TEAM-037 (Testing Team)

**What was added:**
- Mission statement for TEAM-037
- Detailed list of TEAM-036's deliverables
- Test coverage gaps identified
- Testing requirements for GGUF support
- Testing requirements for installation system
- Fining authority reminder
- Priority matrix for testing tasks

**Key sections:**
1. Your Mission (TEAM-037)
2. What TEAM-036 Delivered
3. Test Coverage Gaps
4. Testing Requirements
5. Priority Matrix
6. References

---

## 🧪 BDD Test Suite

### `test-harness/bdd/README.md` [NEW +288 lines]
**Created by:** TEAM-037  
**Purpose:** BDD test harness documentation

**What was added:**
- Overview of BDD testing approach
- How to run tests with `bdd-runner`
- Test organization structure
- Feature file descriptions
- Lifecycle rules references
- Step definition guidelines
- Proof bundle integration notes

**Key sections:**
1. Overview
2. Running Tests
3. Test Organization
4. Feature Files
5. Step Definitions
6. Lifecycle Rules
7. References

---

### `test-harness/bdd/tests/features/test-001.feature` [NEW +788 lines]
**Created by:** TEAM-037  
**Purpose:** Complete BDD test suite for TEST-001

**What was added:**
- 67 Gherkin scenarios covering all TEST-001 behaviors
- Happy path scenarios (basic inference, streaming, cancellation)
- Error handling scenarios (VRAM exhaustion, model not found, network errors)
- Lifecycle scenarios (daemon persistence, cascading shutdown, idle timeout)
- Edge cases (concurrent requests, large prompts, zero tokens)
- CLI command scenarios
- Critical lifecycle rules documentation in comments

**Scenario categories:**
1. Basic Inference (happy path)
2. Streaming
3. Cancellation
4. Error Handling
5. Lifecycle (CRITICAL)
6. Error Response Format
7. CLI Commands

**Critical lifecycle rules documented:**
- RULE 1: rbee-hive is a PERSISTENT HTTP DAEMON
- RULE 2: llm-worker-rbee is a PERSISTENT HTTP DAEMON
- RULE 3: rbee-keeper is a CLI (EPHEMERAL)
- RULE 4: Ephemeral Mode
- RULE 5: Persistent Mode
- RULE 6: Cascading Shutdown
- RULE 7: Worker Idle Timeout
- RULE 8: Process Ownership

---

### `test-harness/bdd/tests/features/test-001-mvp.feature` [NEW +604 lines]
**Created by:** TEAM-037  
**Purpose:** MVP subset of TEST-001 with 27 critical scenarios

**What was added:**
- 27 prioritized scenarios for initial implementation
- Focus on critical path and essential edge cases
- Subset of full test-001.feature
- Same lifecycle rules documentation

**Scenario priorities:**
1. Critical path (basic inference)
2. Essential error handling (VRAM, model not found)
3. Basic lifecycle (daemon persistence)
4. Basic cancellation

---

## 📚 Specification Documents

### `bin/.specs/ARCHITECTURE_UPDATE.md` [NEW +279 lines]
**Created by:** TEAM-037  
**Purpose:** Document queen-rbee orchestration architecture

**What was added:**
- Critical clarification: rbee-keeper is a TESTING TOOL
- Updated architecture with queen-rbee as orchestrator
- Control flow diagrams showing queen-rbee → rbee-hive → workers
- Ephemeral vs persistent mode explanations
- Production vs testing usage patterns
- Cascading shutdown behavior

**Key sections:**
1. CRITICAL: rbee-keeper Purpose
2. What Changed (from old architecture)
3. New Architecture (queen-rbee orchestration)
4. Control Flow
5. Lifecycle Rules
6. Production vs Testing

---

### `bin/.specs/CRITICAL_RULES.md` [NEW +430 lines]
**Created by:** TEAM-037  
**Purpose:** P0 normative rules that MUST be followed

**What was added:**
- RULE 1: rbee-keeper is a TESTING TOOL (not for production)
- RULE 2: queen-rbee orchestrates everything
- RULE 3: Daemons are persistent (rbee-hive, workers)
- RULE 4: Cascading shutdown behavior
- RULE 5: Worker idle timeout (5 minutes)
- RULE 6: Process ownership rules
- Visual diagrams for each rule
- Examples of correct and incorrect usage

**Critical warnings:**
- ✅ USE rbee-keeper FOR: Testing, development, debugging
- ❌ DO NOT USE rbee-keeper FOR: Production, user-facing apps
- Production users should use: llama-orch SDK → queen-rbee directly

---

### `bin/.specs/LIFECYCLE_CLARIFICATION.md` [NEW +522 lines]
**Created by:** TEAM-037  
**Purpose:** Normative lifecycle rules for all processes

**What was added:**
- Problem statement (ambiguity in original spec)
- Correct architecture (queen-rbee orchestration)
- Detailed lifecycle rules for each process
- When processes start, run, and die
- Who orchestrates whom
- Ephemeral vs persistent modes
- Cascading shutdown behavior
- Worker idle timeout behavior
- Process ownership rules

**Key sections:**
1. Problem Statement
2. Correct Architecture
3. Lifecycle Rules (8 rules)
4. Process Responsibilities
5. Control Flow Diagrams
6. Examples

---

### `bin/.specs/SPEC_UPDATE_SUMMARY.md` [NEW +235 lines]
**Created by:** TEAM-037  
**Purpose:** Summary of all spec updates by TEAM-037

**What was added:**
- List of all files created
- List of all files modified
- List of all files deleted
- Rationale for each change
- Impact assessment
- References to lifecycle rules

**Key sections:**
1. Files Created
2. Files Modified
3. Files Deleted
4. Rationale
5. Impact
6. References

---

### `bin/.specs/TEAM_037_COMPLETION_SUMMARY.md` [NEW +527 lines]
**Created by:** TEAM-037  
**Purpose:** TEAM-037's completion report

**What was added:**
- Mission statement
- Deliverables list
- Lifecycle discoveries
- File structure overview
- Test suite statistics
- Next steps for TEAM-038

**Key discoveries:**
1. rbee-keeper is a testing tool
2. Daemons are persistent
3. Cascading shutdown
4. Worker idle timeout

---

### `bin/.specs/ARCHITECTURE_MODES.md` [MODIFIED +75/-22 lines]
**Modified by:** TEAM-037  
**Original by:** TEAM-030

**What changed:**
- Added critical warning: rbee-keeper is a TESTING TOOL
- Updated ephemeral mode description (rbee-keeper spawns queen-rbee)
- Updated persistent mode description (queen-rbee pre-started)
- Added cascading shutdown rules
- Updated control flow diagrams
- Added production vs testing usage patterns

**Specific changes:**
- Line 4: Added "Updated by: TEAM-037 (2025-10-10T14:02)"
- Lines 10-17: Added critical warning section
- Lines 24-26: Updated mode descriptions to include queen-rbee
- Throughout: Replaced direct rbee-hive references with queen-rbee orchestration

---

### `bin/.specs/COMPONENT_RESPONSIBILITIES_FINAL.md` [MODIFIED +18/-7 lines]
**Modified by:** TEAM-037  
**Original by:** TEAM-024

**What changed:**
- Added critical warning: rbee-keeper is a TESTING TOOL
- Updated component table to clarify queen-rbee orchestration
- Added production vs testing column
- Updated rbee-keeper description (testing tool, not production)
- Added cascading shutdown notes

**Specific changes:**
- Line 4: Added "Updated: 2025-10-10T14:02 (TEAM-037 - queen-rbee orchestration)"
- Lines 11-17: Added critical warning section
- Line 27: Updated rbee-keeper description to "TESTING TOOL"
- Throughout: Clarified queen-rbee as THE BRAIN

---

### `bin/.specs/FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md` [MODIFIED +54/-21 lines]
**Modified by:** TEAM-037  
**Original by:** TEAM-024

**What changed:**
- Updated control flow diagrams to show queen-rbee orchestration
- Clarified SSH control plane (queen-rbee → rbee-hive)
- Clarified HTTP inference plane (rbee-hive → workers)
- Added lifecycle rules references
- Updated process ownership section

**Specific changes:**
- Control flow diagrams updated to include queen-rbee
- SSH control plane now: queen-rbee → SSH → rbee-hive
- HTTP inference plane: rbee-hive → HTTP → workers
- Added references to CRITICAL_RULES.md and LIFECYCLE_CLARIFICATION.md

---

### `bin/.specs/FEATURE_TOGGLES.md` [MODIFIED +2/-1 lines]
**Modified by:** TEAM-037

**What changed:**
- Line 6: Added "Updated: 2025-10-10 (TEAM-037 - updated architecture reference)"
- Line 7: Updated architecture reference to point to CRITICAL_RULES.md

**Why:**
- Ensure readers see latest architecture (queen-rbee orchestration)
- Point to normative rules document

---

### `bin/.specs/INSTALLATION_RUST_SPEC.md` [MODIFIED +9/-3 lines]
**Modified by:** TEAM-036

**What changed:**
- Line 6: Added "Updated: 2025-10-10 (TEAM-036 - implemented XDG-compliant installation system)"
- Line 7: Added architecture reference to CRITICAL_RULES.md
- Updated status from "SPECIFICATION ONLY" to "IMPLEMENTED"
- Added references to `rbee-keeper install` command
- Added references to XDG Base Directory specification

**Why:**
- Document that installation system is now implemented
- Point to actual implementation in rbee-keeper

---

## 💻 Code Files

### `bin/llm-worker-rbee/src/backend/models/quantized_llama.rs` [NEW +93 lines]
**Created by:** TEAM-036  
**Purpose:** GGUF quantized model support

**What was added:**
- `QuantizedLlamaModel` struct wrapping candle's quantized_llama
- `load()` function to load GGUF files
- GGUF metadata extraction (vocab_size, eos_token_id)
- `forward()` function for inference
- `reset_cache()` function for cache management
- `eos_token_id()` and `vocab_size()` getters
- `architecture()` function returning "quantized-llama"

**Key implementation details:**
- Uses `candle_core::quantized::gguf_file::Content::read()` to load GGUF
- Extracts metadata from GGUF headers
- Wraps `candle_transformers::models::quantized_llama::ModelWeights`
- Supports Q4_K_M, Q5_K_M, and other quantization formats

**Critical impact:**
- Unblocks all inference with quantized models
- Enables smaller model sizes (4-bit, 5-bit quantization)
- Reduces VRAM requirements

---

### `bin/llm-worker-rbee/src/backend/models/mod.rs` [MODIFIED +37/-2 lines]
**Modified by:** TEAM-036  
**Original by:** TEAM-017

**What changed:**
- Line 5: Added "Modified by: TEAM-036 (added GGUF support for quantized models)"
- Line 16: Added `pub mod quantized_llama;`
- Line 27: Added `QuantizedLlama(quantized_llama::QuantizedLlamaModel)` to Model enum
- Line 41: Added `Model::QuantizedLlama(m) => m.forward(input_ids, position)` to forward()
- Line 50: Added `Model::QuantizedLlama(m) => m.eos_token_id()` to eos_token_id()
- Line 58: Added `Model::QuantizedLlama(m) => m.vocab_size()` to vocab_size()
- Line 66: Added `Model::QuantizedLlama(m) => m.reset_cache()` to reset_cache()
- Line 75: Added `Model::QuantizedLlama(_) => "quantized-llama"` to architecture()
- Lines 120-125: Added GGUF detection in load_model()
- Lines 180-185: Added GGUF size calculation in calculate_model_size()

**GGUF detection logic:**
```rust
if model_path.ends_with(".gguf") {
    let model = quantized_llama::QuantizedLlamaModel::load(path, device)?;
    return Ok(Model::QuantizedLlama(model));
}
```

**Critical impact:**
- Automatic GGUF file detection by extension
- Seamless integration with existing model factory
- Backward compatible with existing models

---

### `bin/rbee-keeper/src/commands/install.rs` [NEW +138 lines]
**Created by:** TEAM-036  
**Purpose:** XDG-compliant installation system

**What was added:**
- `InstallTarget` enum (User, System)
- `handle()` function implementing installation logic
- XDG Base Directory specification compliance
- User install: `~/.local/bin`, `~/.config/rbee`, `~/.local/share/rbee`
- System install: `/usr/local/bin`, `/etc/rbee`, `/var/lib/rbee`
- Binary copying from build directory
- Default config file generation
- Installation instructions

**Key implementation details:**
- Creates standard directories (bin, config, data/models)
- Copies binaries: rbee-keeper, rbee-hive, llm-worker-rbee
- Generates default config.toml
- Provides clear installation instructions
- Supports both user and system install

**Critical impact:**
- Unblocks deployment to production
- Follows Linux standards (XDG)
- Removes hardcoded paths

---

### `bin/rbee-keeper/src/config.rs` [NEW +59 lines]
**Created by:** TEAM-036  
**Purpose:** Configuration file loading with XDG support

**What was added:**
- `Config` struct with pool, paths, remote sections
- `PoolConfig` struct (name, listen_addr)
- `PathsConfig` struct (models_dir, catalog_db)
- `RemoteConfig` struct (binary_path, git_repo_dir)
- `load()` function with priority: RBEE_CONFIG env var > ~/.config/rbee/config.toml > /etc/rbee/config.toml
- TOML deserialization

**Key implementation details:**
- Environment variable override support
- Standard config locations
- Optional remote configuration
- Graceful fallback to defaults

**Critical impact:**
- Enables configurable paths
- Removes hardcoded paths
- Supports remote deployment

---

### `bin/rbee-keeper/Cargo.toml` [MODIFIED +3 lines]
**Modified by:** TEAM-036

**What changed:**
- Added dependency: `dirs = "5.0"`
- Added dependency: `toml = "0.8"`
- Added dependency: `serde = { version = "1.0", features = ["derive"] }`

**Why:**
- `dirs`: For XDG directory detection
- `toml`: For config file parsing
- `serde`: For config deserialization

---

### `bin/rbee-keeper/src/cli.rs` [MODIFIED +7 lines]
**Modified by:** TEAM-036  
**Original by:** TEAM-022, TEAM-027

**What changed:**
- Line 5: Added "Modified by: TEAM-036 (added Install command)"
- Lines 18-23: Added `Install` command with `--system` flag
- Line 18: Added comment "(TEAM-036)"

**Install command:**
```rust
Install {
    /// Install to system paths (requires sudo)
    #[arg(long)]
    system: bool,
}
```

---

### `bin/rbee-keeper/src/commands/mod.rs` [MODIFIED +2 lines]
**Modified by:** TEAM-036  
**Original by:** TEAM-022, TEAM-024

**What changed:**
- Line 5: Added "Modified by: TEAM-036 (added install command)"
- Line 9: Added `pub mod install;`

---

### `bin/rbee-keeper/src/commands/pool.rs` [MODIFIED +50/-22 lines]
**Modified by:** TEAM-036  
**Original by:** TEAM-022

**What changed:**
- Line 4: Added "Modified by: TEAM-036 (removed hardcoded paths, use binaries from PATH)"
- Added `get_remote_binary_path()` helper function
- Added `get_remote_repo_dir()` helper function
- Updated all commands to use configurable paths
- Removed hardcoded `~/Projects/llama-orch` paths
- Removed hardcoded `./target/release/rbee-hive` paths
- Now uses config system for remote paths

**Before:**
```rust
"cd ~/Projects/llama-orch && ./target/release/rbee-hive models list"
```

**After:**
```rust
let binary = get_remote_binary_path(); // Configurable, defaults to "rbee-hive"
format!("{} models list", binary)
```

**Critical impact:**
- Removes hardcoded paths
- Enables remote deployment
- Supports custom installation paths

---

### `bin/rbee-keeper/src/main.rs` [MODIFIED +1 line]
**Modified by:** TEAM-036  
**Original by:** TEAM-022, TEAM-027

**What changed:**
- Line 22: Added "Modified by: TEAM-036 (added config module and install command)"

---

## 🗑️ Deleted Files

### `bin/.specs/.gherkin/test-001-mvp-local.sh` [DELETED -90 lines]
**Why deleted:**
- Obsolete: Replaced by BDD test harness
- Shell scripts replaced by proper Gherkin features
- BDD runner (`bdd-runner`) now handles test execution

---

### `bin/.specs/.gherkin/test-001-mvp-preflight.sh` [DELETED -133 lines]
**Why deleted:**
- Obsolete: Replaced by BDD test harness
- Preflight checks now in BDD scenarios
- Proper test setup in step definitions

---

### `bin/.specs/.gherkin/test-001-mvp-run.sh` [DELETED -80 lines]
**Why deleted:**
- Obsolete: Replaced by BDD test harness
- Test execution now via `bdd-runner`
- Proper test orchestration in Cucumber

---

### `bin/.specs/ARCHITECTURE_DECISION_CLI_VS_HTTP.md` [DELETED -542 lines]
**Why deleted:**
- Obsolete: Superseded by ARCHITECTURE_UPDATE.md
- Old architecture decision no longer relevant
- New architecture clarified in CRITICAL_RULES.md

---

### `bin/.specs/ARCHITECTURE_DECISION_NO_POOL_DAEMON.md` [DELETED -396 lines]
**Why deleted:**
- Obsolete: Superseded by LIFECYCLE_CLARIFICATION.md
- Old decision no longer relevant
- Lifecycle rules now in normative document

---

### `bin/.specs/ARCHITECTURE_SUMMARY_TEAM025.md` [DELETED -235 lines]
**Why deleted:**
- Obsolete: Old team summary
- Superseded by newer architecture docs
- TEAM-025 work completed and documented elsewhere

---

### `bin/.specs/BINARY_ARCHITECTURE_COMPLETE.md` [DELETED -434 lines]
**Why deleted:**
- Obsolete: Superseded by COMPONENT_RESPONSIBILITIES_FINAL.md
- Old architecture no longer accurate
- New architecture in CRITICAL_RULES.md

---

### `bin/.specs/BINARY_STRUCTURE_CLARIFICATION.md` [DELETED -286 lines]
**Why deleted:**
- Obsolete: Superseded by ARCHITECTURE_UPDATE.md
- Old clarification no longer needed
- New clarification in LIFECYCLE_CLARIFICATION.md

---

### `bin/.specs/COMPLETE_BINARY_ARCHITECTURE.md` [DELETED -736 lines]
**Why deleted:**
- Obsolete: Superseded by FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md
- Old architecture no longer accurate
- New architecture consolidated in multiple docs

---

### `bin/.specs/CONTROL_PLANE_ARCHITECTURE_DECISION.md` [DELETED -715 lines]
**Why deleted:**
- Obsolete: Superseded by CRITICAL_RULES.md
- Old decision no longer relevant
- Control plane now clarified in FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md

---

## Summary Statistics

| Category | Files | Lines Added | Lines Removed | Net Change |
|----------|-------|-------------|---------------|------------|
| Planning | 2 | +945 | 0 | +945 |
| BDD Tests | 3 | +1,680 | 0 | +1,680 |
| Specs (New) | 5 | +2,047 | 0 | +2,047 |
| Specs (Modified) | 5 | +167 | -54 | +113 |
| Specs (Deleted) | 7 | 0 | -2,932 | -2,932 |
| Code (GGUF) | 2 | +130 | -2 | +128 |
| Code (Install) | 7 | +267 | -22 | +245 |
| Shell Scripts | 3 | 0 | -303 | -303 |
| **TOTAL** | **34** | **5,236** | **-3,313** | **+1,923** |

---

## Key Takeaways

### TEAM-036 Achievements
1. ✅ Implemented GGUF support (unblocks quantized models)
2. ✅ Implemented XDG-compliant installation (unblocks deployment)
3. ✅ Removed hardcoded paths (enables remote deployment)
4. ✅ Added configuration system (enables customization)

### TEAM-037 Achievements
1. ✅ Created comprehensive BDD test suite (67 scenarios)
2. ✅ Clarified critical lifecycle rules (8 rules)
3. ✅ Documented queen-rbee orchestration
4. ✅ Cleaned up obsolete documentation (7 files deleted)

### Critical Discoveries
1. **rbee-keeper is a testing tool** (not for production)
2. **Daemons are persistent** (don't die after inference)
3. **Cascading shutdown** (queen-rbee → rbee-hive → workers)
4. **Worker idle timeout** (5 minutes, frees VRAM)

---

**End of File Changes Documentation**
