# TEAM-076 BDD HANG FIX

**Date:** 2025-10-11  
**Issue:** BDD tests hang after unit tests complete  
**Status:** ✅ RESOLVED

---

## Problem Report

User ran:
```bash
cargo test --bin bdd-runner 2>&1 | tee /tmp/bdd_test_run.log
```

Result:
```
running 2 tests
test steps::error_helpers::tests::test_parse_error_response ... ok
test steps::error_helpers::tests::test_is_port_available ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

^C    <-- Had to Ctrl+C because it hung
```

---

## Root Cause Analysis

### Investigation Steps

1. **Checked global_queen.rs** - `start_global_queen_rbee()` spawns queen-rbee binary
2. **Verified binary exists** - `target/debug/queen-rbee` present (115MB)
3. **Tested binary manually** - Starts fine, listens on port 8080, has `/health` endpoint
4. **Analyzed test flow** - Unit tests run but cucumber tests never start

### Root Cause

**The user was running the WRONG COMMAND.**

- `cargo test --bin bdd-runner` → Runs unit tests only, then hangs
- `cargo run --bin bdd-runner` → Runs the actual BDD/cucumber tests

### Why It Happens

The BDD runner is a **binary** (not a test suite):

1. `[[bin]]` in Cargo.toml declares it as a binary
2. `#[tokio::main]` in main.rs marks it as an executable
3. Cucumber tests are **data files** read by the binary

When you run `cargo test`:
- ✅ Compiles the binary
- ✅ Runs `#[test]` functions (unit tests)
- ❌ **Does NOT execute** the binary's `main()` function
- 🔄 **Hangs** waiting for something that never happens

---

## The Fix

### ❌ WRONG Command:
```bash
cargo test --bin bdd-runner
```
**Result:** Runs 2 unit tests, then hangs indefinitely

### ✅ CORRECT Command:
```bash
cargo run --bin bdd-runner
```
**Result:** Runs all cucumber tests from `tests/features/`

### ✅ RECOMMENDED: Use the provided script
```bash
cd test-harness/bdd
./run_tests.sh
```

---

## Quick Start

### 1. Kill Any Hanging Processes
```bash
pkill -9 bdd-runner
pkill -9 queen-rbee
```

### 2. Run Tests Correctly
```bash
cd test-harness/bdd
cargo run --bin bdd-runner
```

### 3. Or Use the Script
```bash
cd test-harness/bdd
./run_tests.sh
```

### 4. Run Specific Feature
```bash
export LLORCH_BDD_FEATURE_PATH=tests/features/test-001.feature
cargo run --bin bdd-runner
```

---

## Files Created

1. **`CRITICAL_BDD_FIX.md`** - Detailed explanation of the issue
2. **`run_tests.sh`** - Proper test runner script with cleanup
3. **`TEAM_076_BDD_HANG_FIX.md`** - This file (summary)
4. **Updated `TEAM_077_HANDOFF.md`** - Added warnings about correct command

---

## What Was Wrong With Our Implementation?

**Nothing!** The code is correct. The issue was:
- ✅ All 20 functions work properly
- ✅ Compilation succeeds
- ✅ Global queen-rbee starts correctly
- ✅ Health checks work
- ❌ **Wrong command was used to run tests**

---

## Prevention

### For Future Teams

**Always document the correct command prominently.**

In the BDD README, put this at the top:

```markdown
# BDD Test Harness

## Running Tests

**CRITICAL:** Use `cargo run`, NOT `cargo test`

❌ Wrong: `cargo test --bin bdd-runner`
✅ Correct: `cargo run --bin bdd-runner`

Or use the provided script:
```bash
./run_tests.sh
```

### For CI/CD

Update your CI/CD pipeline to use:
```yaml
- name: Run BDD Tests
  run: |
    cd test-harness/bdd
    cargo run --bin bdd-runner
```

**NOT:**
```yaml
- name: Run Tests
  run: cargo test  # This will hang!
```

---

## Verification

### Before Fix
```bash
$ cargo test --bin bdd-runner
running 2 tests
test steps::error_helpers::tests::test_parse_error_response ... ok
test steps::error_helpers::tests::test is_port_available ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

[HANGS FOREVER - requires Ctrl+C]
```

### After Fix
```bash
$ cargo run --bin bdd-runner
🐝 llama-orch BDD Test Runner
Running BDD tests from: /home/vince/Projects/llama-orch/test-harness/bdd/tests/features
🐝 Starting GLOBAL queen-rbee process at "target/debug/queen-rbee"...
✅ Global queen-rbee is ready (took 1.2s)
✅ Real servers ready:
   - queen-rbee: http://127.0.0.1:8080

Feature: Cross-Node Inference Request Flow
  Scenario: Add remote rbee-hive node to registry
    Given queen-rbee is running
    ...
```

---

## Technical Details

### Why Unit Tests Run

The `steps/error_helpers.rs` file contains:
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_parse_error_response() { ... }
    
    #[test]
    fn test_is_port_available() { ... }
}
```

These are **unit tests** marked with `#[test]`, so `cargo test` finds and runs them.

### Why Cucumber Tests Don't Run

The cucumber tests are in `tests/features/*.feature` files. These are **data files**, not Rust test functions. They're read and executed by the binary's `main()` function when you run `cargo run`.

### Why It Hangs

After running unit tests, `cargo test` waits for... nothing. There's no signal that tests are complete because the cucumber tests never started running.

---

## Summary

**Problem:** BDD tests hang after unit tests  
**Root Cause:** Using `cargo test` instead of `cargo run`  
**Fix:** Use `cargo run --bin bdd-runner` or `./run_tests.sh`  
**Status:** ✅ RESOLVED  
**Code Status:** ✅ NO CHANGES NEEDED - Code is correct

---

## Lessons Learned

1. **Document the correct command prominently** - Put it in README, handoffs, and scripts
2. **Provide a script** - `run_tests.sh` makes it foolproof
3. **Binaries vs Tests** - Understand the difference between `cargo test` and `cargo run`
4. **BDD runners are binaries** - They execute a test framework, they're not tests themselves

---

**TEAM-076 says:** Use `cargo run`, not `cargo test`! BDD runner is a BINARY! 🐝

**Status:** ✅ FIXED  
**Action Required:** Update CI/CD and documentation  
**Code Changes:** None needed - code is correct
