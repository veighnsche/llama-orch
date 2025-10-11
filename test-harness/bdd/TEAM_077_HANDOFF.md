# TEAM-077 HANDOFF - FEATURE FILE REFACTORING

**From:** TEAM-076  
**To:** TEAM-077  
**Date:** 2025-10-11  
**Status:** CRITICAL REFACTORING REQUIRED

---

## Your Mission

**CRITICAL:** The `test-001.feature` file has lost its purpose. It was defined as ONE FEATURE but actually contains MANY FEATURES with different scenarios. This violates BDD best practices and makes tests unmaintainable.

**Your Goal:** Split `test-001.feature` into properly organized feature files, each representing a single feature with related scenarios.

---

## What TEAM-076 Accomplished

### Functions Implemented: 20 Total ✅
- 3 SSE streaming functions (download, loading, tokens)
- 7 worker management functions (health, state, spawn, callbacks)
- 3 registry operation functions (queries, SSH validation)
- 2 resource validation functions (RAM, backend checks)
- 4 inference operation functions (validation, streaming, state transitions)
- 1 model catalog function (registration with validation)

### Code Status
- ✅ 20 functions with real API calls
- ✅ 3 TODO markers removed
- ✅ 31 error codes introduced
- ✅ 6 files modified
- ✅ Zero compilation errors
- ✅ All tests connect to real product code in `/bin/`

**Combined Total:** 61 error handling functions (TEAM-074: 26, TEAM-075: 15, TEAM-076: 20)

---

## The Problem: test-001.feature

### Current State
- **File:** `test-harness/bdd/tests/features/test-001.feature`
- **Size:** 1676 lines
- **Scenarios:** 91 total
- **Problem:** ONE file contains MULTIPLE features mixed together

### What's Wrong
```gherkin
Feature: Cross-Node Inference Request Flow
  # BUT IT CONTAINS:
  # - SSH setup scenarios
  # - Registry management scenarios
  # - Model provisioning scenarios
  # - Worker preflight scenarios
  # - Worker startup scenarios
  # - Inference execution scenarios
  # - Error handling scenarios
  # - GGUF support scenarios
  # - And more...
```

**This is WRONG.** Each of these should be a separate feature file.

---

## Your Mission: Methodical Feature File Refactoring

### ⚠️ CRITICAL RULES

1. **NO SCENARIOS MAY BE LOST** - Every scenario must be migrated
2. **NO GAPS** - Verify all scenarios are accounted for
3. **DELETE test-001.feature ONLY AFTER** all scenarios are migrated
4. **WORK METHODICALLY** - Follow the phases below in order
5. **VERIFY AT EACH STEP** - Don't proceed until current phase is complete

---

## Phase 1: Investigation & Inventory (2-3 hours)

### Step 1.1: Read and Understand test-001.feature
```bash
cd test-harness/bdd/tests/features
wc -l test-001.feature  # Check size
grep "^  Scenario:" test-001.feature | wc -l  # Count scenarios
```

**Deliverable:** Understand the full scope

### Step 1.2: Create Inventory Document
Create: `test-harness/bdd/FEATURE_REFACTORING_INVENTORY.md`

**Contents:**
```markdown
# Feature Refactoring Inventory

## Current State
- File: test-001.feature
- Total lines: 1676
- Total scenarios: 91
- Status: NEEDS SPLITTING

## Scenario Inventory

### By Line Number
1. Line 37-74: Add remote rbee-hive node to registry
2. Line 77-100: EH-001a - SSH connection timeout
3. Line 103-126: EH-001b - SSH authentication failure
... (continue for ALL 91 scenarios)

### By Feature Category
**SSH & Registry Setup (8 scenarios)**
- Add remote rbee-hive node to registry
- SSH connection timeout
- SSH authentication failure
- SSH command execution failure
- Install rbee-hive on remote node
- List registered rbee-hive nodes
- Remove node from rbee-hive registry
- Invalid SSH key path

**Model Provisioning (12 scenarios)**
- Model found in SQLite catalog
- Model not found - download with progress
- Model not found on Hugging Face
- Model repository is private
... (continue)

**Worker Preflight (8 scenarios)**
... (continue)

**Worker Startup (6 scenarios)**
... (continue)

**Inference Execution (10 scenarios)**
... (continue)

**Error Handling (25 scenarios)**
... (continue)

**GGUF Support (4 scenarios)**
... (continue)

**Happy Path (2 scenarios)**
... (continue)
```

**Verification:**
- [ ] All 91 scenarios listed by line number
- [ ] All scenarios categorized by feature
- [ ] No duplicates
- [ ] No gaps in line numbers

### Step 1.3: Identify Feature Boundaries

Analyze the scenarios and identify distinct features:

**Proposed Features:**
1. **SSH Registry Management** - Node registration, SSH validation
2. **Model Provisioning** - Download, catalog, GGUF support
3. **Worker Preflight Checks** - RAM, backend, resource validation
4. **Worker Lifecycle** - Startup, registration, callbacks
5. **Inference Execution** - Request handling, token streaming
6. **Error Handling - SSH** - SSH-specific error scenarios
7. **Error Handling - Network** - HTTP, timeout, retry scenarios
8. **Error Handling - Resources** - RAM, disk, VRAM errors
9. **Happy Path Flows** - End-to-end success scenarios

**Deliverable:** List of proposed feature files with scenario counts

---

## Phase 2: Feature File Design (1-2 hours)

### Step 2.1: Design Feature File Structure

Create: `test-harness/bdd/FEATURE_FILE_DESIGN.md`

**Contents:**
```markdown
# Feature File Design

## Naming Convention
- Format: `{number}-{feature-name}.feature`
- Numbers: 01, 02, 03... (for ordering)
- Names: kebab-case, descriptive

## Proposed Feature Files

### 01-ssh-registry-management.feature
**Purpose:** SSH connection setup and node registry management
**Scenarios:** 8
**Lines:** ~300
**Scenarios:**
- Add remote rbee-hive node to registry
- SSH connection timeout (EH-001a)
- SSH authentication failure (EH-001b)
- SSH command execution failure
- Install rbee-hive on remote node
- List registered rbee-hive nodes
- Remove node from rbee-hive registry
- Invalid SSH key path (EH-011a)
- Duplicate node name (EH-011b)
- Inference fails when node not in registry

### 02-model-provisioning.feature
**Purpose:** Model download, catalog, and GGUF support
**Scenarios:** 16
**Lines:** ~500
**Scenarios:**
- Model found in SQLite catalog
- Model not found - download with progress
- Model not found on Hugging Face (EH-007a)
- Model repository is private (EH-007b)
- Model download timeout (EH-008a)
- Model download fails with retry (EH-008b)
- Downloaded model checksum mismatch (EH-008c)
- Model catalog registration after download
- GGUF model detection by file extension
- GGUF metadata extraction
- GGUF quantization formats supported
- GGUF model size calculation
- Insufficient disk space for download (EH-006a)
- Disk fills up during download (EH-006b)

### 03-worker-preflight-checks.feature
**Purpose:** Resource validation before worker startup
**Scenarios:** 8
**Lines:** ~300
**Scenarios:**
- Worker preflight RAM check passes
- Worker preflight RAM check fails (EH-004a)
- RAM exhausted during model loading (EH-004b)
- Worker preflight backend check passes
- VRAM exhausted on CUDA device (EH-005a)
- Backend not available (EH-009a)
- CUDA not installed (EH-009b)

### 04-worker-lifecycle.feature
**Purpose:** Worker startup, registration, and callbacks
**Scenarios:** 10
**Lines:** ~350
**Scenarios:**
- Worker startup sequence
- Worker ready callback
- Worker binary not found (EH-012a)
- Worker spawn fails (EH-012b)
- Worker crashes during initialization (EH-012c)
- Worker HTTP server binds to port
- Worker sends ready callback
- rbee-hive registers worker in registry
- Model loading begins asynchronously

### 05-inference-execution.feature
**Purpose:** Inference request handling and token streaming
**Scenarios:** 12
**Lines:** ~400
**Scenarios:**
- Worker registry returns empty list
- Worker registry returns matching idle worker
- Worker registry returns matching busy worker
- Inference request sent to worker
- Worker streams tokens via SSE
- Inference completes with token count
- Worker transitions to idle after inference
- Worker transitions from idle to busy to idle
- Inference with max tokens limit
- Inference with temperature parameter

### 06-error-handling-network.feature
**Purpose:** HTTP, timeout, and retry error scenarios
**Scenarios:** 10
**Lines:** ~350
**Scenarios:**
- rbee-hive HTTP connection timeout (EH-002a)
- rbee-hive returns malformed JSON (EH-002b)
- Pool preflight detects version mismatch
- Pool preflight connection timeout with retry
- Retry with exponential backoff
- Circuit breaker after max retries

### 07-error-handling-resources.feature
**Purpose:** RAM, disk, VRAM error scenarios
**Scenarios:** 8
**Lines:** ~300
**Scenarios:**
- Insufficient RAM (EH-004a)
- RAM exhausted during loading (EH-004b)
- VRAM exhausted (EH-005a)
- Insufficient disk space (EH-006a)
- Disk fills up during download (EH-006b)

### 08-happy-path-flows.feature
**Purpose:** End-to-end success scenarios
**Scenarios:** 2
**Lines:** ~150
**Scenarios:**
- Happy path - cold start inference on remote node
- Warm start - reuse existing idle worker

### 09-pool-preflight-checks.feature
**Purpose:** Pool-level health and preflight checks
**Scenarios:** 6
**Lines:** ~200
**Scenarios:**
- Pool preflight health check succeeds
- Pool preflight detects version mismatch
- Pool preflight connection timeout with retry
```

**Verification:**
- [ ] All 91 scenarios accounted for
- [ ] No duplicates across files
- [ ] Logical grouping by feature
- [ ] File sizes reasonable (< 500 lines each)

### Step 2.2: Review Design (CRITICAL)

**STOP AND VERIFY:**
1. Count total scenarios across all proposed files
2. Compare with original 91 scenarios
3. Check for duplicates
4. Check for gaps
5. Verify logical grouping

**DO NOT PROCEED** until counts match exactly.

---

## Phase 3: Create New Feature Files (3-4 hours)

### Step 3.1: Create Feature Files One by One

**Process for EACH feature file:**

1. **Create the file**
   ```bash
   touch test-harness/bdd/tests/features/01-ssh-registry-management.feature
   ```

2. **Copy header and feature description**
   ```gherkin
   # Traceability: [appropriate ID]
   # Architecture: TEAM-037 (queen-rbee orchestration)
   # Components: rbee-keeper, queen-rbee, rbee-hive
   
   Feature: SSH Registry Management
     As a user setting up distributed inference
     I want to register remote nodes with SSH details
     So that queen-rbee can manage remote rbee-hive instances
   ```

3. **Copy Background section if needed**

4. **Copy scenarios from test-001.feature**
   - Copy EXACT text (including line breaks, indentation)
   - Preserve all tags (@setup, @error-handling, etc.)
   - Preserve all comments
   - Preserve all docstrings

5. **Verify scenario count**
   ```bash
   grep "^  Scenario:" 01-ssh-registry-management.feature | wc -l
   # Should match design document
   ```

6. **Test compilation**
   ```bash
   cargo check --bin bdd-runner
   ```

7. **Mark as complete in tracking document**

**Repeat for ALL feature files.**

### Step 3.2: Create Migration Tracking Document

Create: `test-harness/bdd/FEATURE_MIGRATION_TRACKING.md`

**Contents:**
```markdown
# Feature Migration Tracking

## Status: IN PROGRESS

### Completed Files
- [ ] 01-ssh-registry-management.feature (0/8 scenarios)
- [ ] 02-model-provisioning.feature (0/16 scenarios)
- [ ] 03-worker-preflight-checks.feature (0/8 scenarios)
- [ ] 04-worker-lifecycle.feature (0/10 scenarios)
- [ ] 05-inference-execution.feature (0/12 scenarios)
- [ ] 06-error-handling-network.feature (0/10 scenarios)
- [ ] 07-error-handling-resources.feature (0/8 scenarios)
- [ ] 08-happy-path-flows.feature (0/2 scenarios)
- [ ] 09-pool-preflight-checks.feature (0/6 scenarios)

### Total Progress
- Scenarios migrated: 0/91
- Files created: 0/9
- Compilation status: Not tested

### Verification Checklist
- [ ] All 91 scenarios migrated
- [ ] No duplicates
- [ ] No gaps
- [ ] All files compile
- [ ] All tests run
- [ ] test-001.feature can be deleted
```

**Update this document after EACH file is created.**

---

## Phase 4: Verification (1-2 hours)

### Step 4.1: Count Verification

```bash
cd test-harness/bdd/tests/features

# Count scenarios in original
grep "^  Scenario:" test-001.feature | wc -l

# Count scenarios in new files
grep "^  Scenario:" 0*.feature | wc -l

# Should be EQUAL
```

**Verification:**
- [ ] Scenario counts match exactly
- [ ] No scenarios appear in multiple files
- [ ] All scenario names preserved

### Step 4.2: Compilation Verification

```bash
cd test-harness/bdd
cargo check --bin bdd-runner 2>&1 | tee migration-compile-check.log
```

**Verification:**
- [ ] Zero compilation errors
- [ ] All feature files found
- [ ] All step definitions matched

### Step 4.3: Test Execution Verification

```bash
# Run tests on new feature files
cargo test --bin bdd-runner 2>&1 | tee migration-test-run.log

# Compare with baseline
# Should have same number of scenarios
```

**Verification:**
- [ ] All scenarios execute
- [ ] No "step not found" errors
- [ ] Pass/fail rates similar to baseline

### Step 4.4: Content Verification

Create: `test-harness/bdd/FEATURE_MIGRATION_VERIFICATION.md`

**Contents:**
```markdown
# Feature Migration Verification

## Scenario Count Verification
- Original file: 91 scenarios
- New files total: [X] scenarios
- Match: [YES/NO]

## Scenario Name Verification
[List all scenario names from original]
[Check each one exists in new files]

## Tag Verification
- @setup tags: [count in original] vs [count in new]
- @error-handling tags: [count in original] vs [count in new]
- @critical tags: [count in original] vs [count in new]
- @gguf tags: [count in original] vs [count in new]

## Compilation Verification
- Errors: 0
- Warnings: [count]
- Status: PASS

## Test Execution Verification
- Scenarios run: [X]/91
- Scenarios passed: [X]
- Scenarios failed: [X]
- Status: [PASS/FAIL]

## Ready for Deletion
- [ ] All scenarios migrated
- [ ] All tests pass
- [ ] No compilation errors
- [ ] Verification document complete
```

---

## Phase 5: Cleanup (30 minutes)

### Step 5.1: Final Verification

**STOP. DO NOT DELETE test-001.feature UNTIL:**
- [ ] All 91 scenarios verified in new files
- [ ] Compilation passes
- [ ] Tests execute successfully
- [ ] Verification document complete
- [ ] Another team member reviews (if possible)

### Step 5.2: Delete test-001.feature

```bash
cd test-harness/bdd/tests/features

# Backup first (just in case)
cp test-001.feature test-001.feature.backup

# Delete
rm test-001.feature

# Verify compilation still works
cd ../..
cargo check --bin bdd-runner
```

### Step 5.3: Update Documentation

Update: `test-harness/bdd/README.md` (if exists) or create it

**Contents:**
```markdown
# BDD Test Harness

## Feature Files

### 01-ssh-registry-management.feature
SSH connection setup and node registry management (8 scenarios)

### 02-model-provisioning.feature
Model download, catalog, and GGUF support (16 scenarios)

### 03-worker-preflight-checks.feature
Resource validation before worker startup (8 scenarios)

### 04-worker-lifecycle.feature
Worker startup, registration, and callbacks (10 scenarios)

### 05-inference-execution.feature
Inference request handling and token streaming (12 scenarios)

### 06-error-handling-network.feature
HTTP, timeout, and retry error scenarios (10 scenarios)

### 07-error-handling-resources.feature
RAM, disk, VRAM error scenarios (8 scenarios)

### 08-happy-path-flows.feature
End-to-end success scenarios (2 scenarios)

### 09-pool-preflight-checks.feature
Pool-level health and preflight checks (6 scenarios)

**Total:** 91 scenarios across 9 feature files
```

### Step 5.4: Create Completion Summary

Create: `test-harness/bdd/TEAM_077_COMPLETION.md`

**Contents:**
```markdown
# TEAM-077 COMPLETION SUMMARY

## Mission: Feature File Refactoring

### Accomplished
- ✅ Analyzed test-001.feature (1676 lines, 91 scenarios)
- ✅ Created inventory of all scenarios
- ✅ Designed 9 feature files with logical grouping
- ✅ Migrated all 91 scenarios to new files
- ✅ Verified no scenarios lost
- ✅ Verified no duplicates
- ✅ Compilation passes
- ✅ Tests execute successfully
- ✅ Deleted test-001.feature
- ✅ Updated documentation

### Feature Files Created
1. 01-ssh-registry-management.feature (8 scenarios)
2. 02-model-provisioning.feature (16 scenarios)
3. 03-worker-preflight-checks.feature (8 scenarios)
4. 04-worker-lifecycle.feature (10 scenarios)
5. 05-inference-execution.feature (12 scenarios)
6. 06-error-handling-network.feature (10 scenarios)
7. 07-error-handling-resources.feature (8 scenarios)
8. 08-happy-path-flows.feature (2 scenarios)
9. 09-pool-preflight-checks.feature (6 scenarios)

### Verification
- Scenario count: 91/91 ✅
- Compilation: PASS ✅
- Tests: PASS ✅
- No gaps: VERIFIED ✅
- No duplicates: VERIFIED ✅

### Time Breakdown
- Phase 1 (Investigation): X hours
- Phase 2 (Design): X hours
- Phase 3 (Migration): X hours
- Phase 4 (Verification): X hours
- Phase 5 (Cleanup): X hours
Total: X hours
```

---

## Success Criteria

### Minimum Requirements (Must Complete)

- [ ] **Phase 1 Complete** - Inventory document with all 91 scenarios listed
- [ ] **Phase 2 Complete** - Design document with 9 feature files planned
- [ ] **Phase 3 Complete** - All 9 feature files created with scenarios migrated
- [ ] **Phase 4 Complete** - Verification document showing 91/91 scenarios migrated
- [ ] **Phase 5 Complete** - test-001.feature deleted, documentation updated

### Quality Gates

**Gate 1: After Phase 1**
- All 91 scenarios listed by line number
- All scenarios categorized by feature
- No duplicates in inventory

**Gate 2: After Phase 2**
- Feature file design complete
- Scenario count adds up to 91
- Logical grouping verified

**Gate 3: After Phase 3**
- All 9 files created
- All scenarios migrated
- Compilation passes

**Gate 4: After Phase 4**
- Verification document complete
- All checks pass
- Ready for deletion

**Gate 5: After Phase 5**
- test-001.feature deleted
- Documentation updated
- Completion summary created

---

## Critical Rules (MUST FOLLOW)

### BDD Rules (MANDATORY)
1. ✅ **NO SCENARIOS MAY BE LOST** - Every scenario must be migrated
2. ✅ **PRESERVE EXACT TEXT** - Copy scenarios exactly (indentation, tags, comments)
3. ✅ **VERIFY AT EACH STEP** - Don't proceed until current phase complete
4. ✅ **DELETE ONLY AFTER VERIFICATION** - test-001.feature deleted last
5. ✅ **DOCUMENT EVERYTHING** - Create all tracking documents

### Work Methodology (MANDATORY)
1. **Work in phases** - Complete each phase before moving to next
2. **Verify at each step** - Check counts, compilation, tests
3. **Track progress** - Update tracking document after each file
4. **Stop if counts don't match** - Investigate discrepancies immediately
5. **Get review if possible** - Have another team member verify before deletion

### Quality Standards (MANDATORY)
1. **Zero scenarios lost** - All 91 must be in new files
2. **Zero duplicates** - No scenario appears in multiple files
3. **Zero compilation errors** - All files must compile
4. **Logical grouping** - Features make sense
5. **Proper naming** - Files follow naming convention

---

## Recommended Workflow

### Day 1: Investigation & Design (3-4 hours)
1. Read test-001.feature thoroughly
2. Create inventory document
3. Identify feature boundaries
4. Create design document
5. Review and verify design

### Day 2: Migration (4-5 hours)
1. Create feature files one by one
2. Copy scenarios carefully
3. Verify after each file
4. Update tracking document
5. Run compilation checks

### Day 3: Verification & Cleanup (2-3 hours)
1. Count verification
2. Compilation verification
3. Test execution verification
4. Create verification document
5. Delete test-001.feature
6. Update documentation
7. Create completion summary

**Total: 9-12 hours**

---

## Resources Available

### Current State
- **File:** `test-harness/bdd/tests/features/test-001.feature`
- **Size:** 1676 lines
- **Scenarios:** 91
- **Step definitions:** 61 error handling functions + many more

### Documentation
- TEAM_076_FINAL_SUMMARY.md - What was implemented
- test-001.feature - Current monolithic file
- All step definition files in `src/steps/`

### Tools
```bash
# Count scenarios
grep "^  Scenario:" test-001.feature | wc -l

# Extract scenario names
grep "^  Scenario:" test-001.feature

# Check compilation
cargo check --bin bdd-runner

# Run tests (CRITICAL: Use cargo run, NOT cargo test)
cargo run --bin bdd-runner

# Or use the provided script
./run_tests.sh

# Run specific feature
LLORCH_BDD_FEATURE_PATH=tests/features/01-ssh-registry-management.feature cargo run --bin bdd-runner
```

### ⚠️ CRITICAL: Don't Use `cargo test`

**DO NOT RUN:** `cargo test --bin bdd-runner`  
**REASON:** This only runs unit tests and hangs

**CORRECT COMMAND:** `cargo run --bin bdd-runner`  
**REASON:** The BDD runner is a binary that executes cucumber tests

See `CRITICAL_BDD_FIX.md` for details.

---

## Common Pitfalls to Avoid

### ❌ DON'T
1. **Use `cargo test --bin bdd-runner`** - This only runs unit tests and hangs! Use `cargo run --bin bdd-runner`
2. **Delete test-001.feature before verification** - You'll lose scenarios
3. **Skip the inventory phase** - You'll miss scenarios
4. **Rush the migration** - You'll make mistakes
5. **Forget to verify counts** - You'll have gaps
6. **Skip compilation checks** - You'll break tests

### ✅ DO
1. **Use `cargo run --bin bdd-runner`** - This is the correct command to run BDD tests
2. **Work methodically** - Follow phases in order
3. **Verify at each step** - Check counts, compilation, tests
4. **Document everything** - Create all tracking documents
5. **Copy exactly** - Preserve indentation, tags, comments
6. **Test frequently** - Run compilation after each file

---

## Expected Outcomes

### Optimistic Scenario
- 9 feature files created
- All 91 scenarios migrated
- Zero scenarios lost
- Zero duplicates
- Compilation passes
- Tests pass
- test-001.feature deleted
- Documentation complete
- Time: 9-10 hours

### Realistic Scenario
- 9 feature files created
- All 91 scenarios migrated
- Minor issues found and fixed
- Compilation passes after fixes
- Tests pass
- test-001.feature deleted
- Documentation complete
- Time: 11-12 hours

### If Issues Arise
- Stop immediately
- Document the issue
- Investigate discrepancies
- Fix before proceeding
- Don't delete test-001.feature until resolved

---

## Final Notes

**This is critical infrastructure work.** The feature files are the foundation of the BDD test suite. Take your time, work methodically, and verify at each step.

**Remember:** It's better to take an extra hour to verify than to lose scenarios and spend days debugging.

**The goal:** Clean, maintainable feature files that follow BDD best practices.

---

**TEAM-076 says:** Feature file refactoring is CRITICAL! Work METHODICALLY! VERIFY at each step! NO SCENARIOS may be LOST! 🐝

**Good luck, TEAM-077! This is important work. Do it right.**

---

**Handoff Status:** ✅ READY FOR TEAM-077  
**Current State:** test-001.feature needs splitting  
**Target State:** 9 feature files, 91 scenarios preserved  
**Priority:** HIGH - This blocks maintainability
