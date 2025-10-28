# TEAM-339: Disabled Tests Cleanup

**Status:** ✅ COMPLETE

**Mission:** Remove all disabled test files and references from narration-core crate

## Files Deleted

### Disabled Test Files (5 files)
1. `tests/e2e_job_client_integration.rs.disabled` - E2E integration tests
2. `tests/harness/mod.rs.disabled` - Test harness infrastructure
3. `tests/job_server_basic.rs.disabled` - Basic job server tests
4. `tests/job_server_concurrent.rs.disabled` - Concurrent job server tests
5. `tests/bin.disabled/` - Directory containing fake test binaries:
   - `fake_hive.rs` (6,436 bytes)
   - `fake_queen.rs` (6,579 bytes)
   - `fake_worker.rs` (2,286 bytes)

### Active Test Files Depending on Disabled Code (2 items)
6. `tests/e2e_real_processes.rs` (15,520 bytes) - 7 tests depending on fake binaries
7. `tests/harness/` - Directory with test infrastructure:
   - `README.md` (6,282 bytes)
   - `sse_utils.rs` (6,496 bytes)

**Total removed:** ~43,600 bytes of test code and infrastructure

## Cargo.toml Changes

**Removed binary definitions:**
```toml
# TEAM-303: Test binaries for E2E integration tests
[[bin]]
name = "fake-queen-rbee"
path = "tests/bin/fake_queen.rs"
required-features = ["axum"]

[[bin]]
name = "fake-rbee-hive"
path = "tests/bin/fake_hive.rs"
required-features = ["axum"]

[[bin]]
name = "fake-worker"
path = "tests/bin/fake_worker.rs"
```

**Lines removed:** 14 lines from Cargo.toml

## Verification

✅ No `.disabled` files remain in narration-core
✅ No references to disabled files in code
✅ No references to fake binaries in active tests
✅ No orphaned harness directory
✅ `cargo check -p observability-narration-core` passes
✅ `cargo fmt --package observability-narration-core` passes
✅ No broken references in documentation

## Rationale

These test files were disabled (likely due to circular dependency issues with job-server, as noted in TEAM-337). Rather than keeping dead code in the repository, we've removed them entirely following RULE ZERO: **Delete dead code immediately**.

If these tests are needed in the future, they should be:
1. Moved to integration tests in the job-server crate (to avoid circular dependencies)
2. Or reimplemented without the circular dependency

## Related Work

- **TEAM-337:** Removed job-server and job-client circular dependencies
- **TEAM-303:** Originally created these test binaries for E2E testing
- **TEAM-308:** Fixed hanging tests (these were already disabled by then)

---

**Date:** Oct 28, 2025  
**Team:** TEAM-339  
**Signature:** Disabled test cleanup complete
