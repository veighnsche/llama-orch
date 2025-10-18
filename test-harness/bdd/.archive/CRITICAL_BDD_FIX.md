# CRITICAL BDD FIX - Test Hanging Issue

**Date:** 2025-10-11  
**Issue:** Tests hang after unit tests complete  
**Root Cause:** Wrong command being used

---

## Problem

Running `cargo test --bin bdd-runner` causes:
1. ✅ Unit tests run and pass (2 tests)
2. ❌ Cucumber tests DON'T run
3. 🔄 Process hangs indefinitely

## Root Cause

The BDD runner is a **BINARY**, not a test suite. 

- `cargo test` runs unit tests (`#[test]` functions)
- `cargo run` executes the binary (which runs cucumber tests)

## The Fix

### ❌ WRONG Command:
```bash
cargo test --bin bdd-runner
# This only runs unit tests, then hangs
```

### ✅ CORRECT Command:
```bash
cargo run --bin bdd-runner
# This actually runs the cucumber tests
```

## Verification

```bash
# Kill any hanging processes
pkill -9 bdd-runner
pkill -9 queen-rbee

# Run the BDD tests correctly
cd test-harness/bdd
cargo run --bin bdd-runner
```

## Why It Hangs

When running `cargo test`, the test framework:
1. Compiles the binary
2. Runs any `#[test]` functions (unit tests)
3. **Stops** - doesn't execute the binary's `main()` function
4. Waits for something that never happens

The cucumber tests are in `tests/features/*.feature` but they're not integration tests - they're data files read by the binary.

## Recommended Script

Create: `test-harness/bdd/run_tests.sh`

```bash
#!/bin/bash
set -e

# Kill any existing test processes
pkill -9 queen-rbee 2>/dev/null || true
pkill -9 bdd-runner 2>/dev/null || true

# Run the BDD tests
cargo run --bin bdd-runner

# Cleanup
pkill -9 queen-rbee 2>/dev/null || true
```

## Additional Notes

### Unit Tests
If you want to run ONLY the unit tests:
```bash
cargo test --lib
```

### Cucumber Tests
To run the actual BDD/cucumber tests:
```bash
cargo run --bin bdd-runner
```

### Specific Feature
To run a specific feature file:
```bash
LLORCH_BDD_FEATURE_PATH=tests/features/test-001.feature cargo run --bin bdd-runner
```

---

**Status:** ✅ FIXED - Use `cargo run --bin bdd-runner`  
**Next Step:** Update CI/CD and documentation
