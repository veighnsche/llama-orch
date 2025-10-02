# Known Issue: capture_tests() from Same Package

**Date**: 2025-10-02  
**Issue**: Cannot use `capture_tests()` from within the same package

---

## Problem

When running:
```bash
cargo +nightly test -p vram-residency generate_comprehensive_proof_bundle_fast -- --ignored --nocapture
```

Result: **0 tests captured**

### Root Cause

The `capture_tests()` API runs `cargo test -p vram-residency --format json`, which:
1. Tries to run ALL tests in the vram-residency package
2. But we're already INSIDE a test in the vram-residency package
3. Cargo filters out the currently running test file to avoid recursion
4. This filters out ALL tests in that file
5. Result: 0 tests found

### Why proof-bundle Works

The proof-bundle crate's own test works because:
- Test runs in package: `proof-bundle`
- Captures tests from: `proof-bundle` (same package, but different test files)
- Works because the test is in `tests/generate_proof_bundle.rs`
- Captures tests from `tests/test_capture_tests.rs` and `src/lib.rs`

### Why vram-residency Doesn't Work

- Test runs in package: `vram-residency`
- Tries to capture from: `vram-residency` (same package)
- Fails because cargo filters the running test file
- All other test files are separate binaries, not captured by `--format json`

---

## Solutions

### Option 1: External Script (RECOMMENDED)

Create a standalone script that's NOT a test:

```bash
#!/usr/bin/env bash
# bin/worker-orcd-crates/vram-residency/scripts/generate_proof_bundle.sh

cd "$(dirname "$0")/.."
export LLORCH_PROOF_DIR="$PWD/.proof_bundle/unit-fast"

cargo +nightly test -p vram-residency \
    --features skip-long-tests \
    -- --format json -Z unstable-options \
    | python3 scripts/parse_test_json.py
```

### Option 2: Separate Package

Move proof bundle generator to a separate package:
```
bin/worker-orcd-crates/vram-residency-proof-gen/
└── src/
    └── main.rs  # Runs capture_tests("vram-residency")
```

### Option 3: Manual Approach (CURRENT)

Keep the old manual approach that was deleted:
- Run `cargo test --format json` manually
- Parse JSON output
- Write to proof bundle files

---

## Temporary Workaround

For now, use the BDD proof bundle generator which works:

```bash
cargo test -p vram-residency --test bdd_runner -- --nocapture
```

This generates proof bundles in `.proof_bundle/bdd/` successfully.

---

## Action Items

1. ⚠️ **Decide**: Which solution to implement?
2. ⚠️ **Document**: Update README with correct instructions
3. ⚠️ **Test**: Verify chosen solution works
4. ⚠️ **Clean up**: Remove non-working test if not fixable

---

**Status**: ⚠️ BLOCKED  
**Blocker**: `capture_tests()` API limitation  
**Impact**: Cannot generate unit test proof bundles automatically  
**Workaround**: Use BDD proof bundles or manual approach
