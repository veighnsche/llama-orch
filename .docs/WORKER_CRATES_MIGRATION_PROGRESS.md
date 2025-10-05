# Worker Crates Migration Progress

**Date**: 2025-10-05  
**Status**: ⏳ In Progress (2/5 complete)

---

## Summary

Migrating Rust code from `worker-orcd` to shared `worker-crates/` to enable code reuse for `worker-aarmd` (Apple ARM Metal worker).

## Completed Migrations ✅

### 1. worker-gguf ✅
**Status**: Already had stub implementation  
**Tests**: 5 unit tests pass  
**Dependencies**: `thiserror`

### 2. worker-tokenizer ✅
**Commit**: `db8852d`  
**Files Moved**: 11 source files + 3 integration tests  
**Tests**: 80 unit tests + 46 integration tests pass  
**Dependencies**: `thiserror`, `serde`, `serde_json`, `tracing`, `tokenizers`, `tempfile`

**Moved**:
- `src/tokenizer/` → `worker-tokenizer/src/`
- `src/util/` (UTF-8 helpers) → `worker-tokenizer/src/util/`
- `tests/tokenizer_conformance_qwen.rs` → `worker-tokenizer/tests/`
- `tests/phi3_tokenizer_conformance.rs` → `worker-tokenizer/tests/`
- `tests/utf8_edge_cases.rs` → `worker-tokenizer/tests/`

**Import Updates**:
- `use crate::tokenizer::` → `use crate::`
- `use worker_orcd::tokenizer::` → `use worker_tokenizer::`

### 3. worker-models ✅
**Commits**: `60c7c19`, `5bdcf78`, `8786097`, `30d2593`  
**Files Moved**: 6 source files + 2 integration tests + common test module  
**Tests**: 38 unit tests + 8 integration tests pass  
**Dependencies**: `worker-gguf`, `thiserror`, `serde`, `serde_json`, `tracing`, `toml` (dev)

**Moved**:
- `src/models/` → `worker-models/src/`
- `tests/adapter_integration.rs` → `worker-models/tests/`
- `tests/adapter_factory_integration.rs` → `worker-models/tests/`
- `tests/common/` → `worker-models/tests/common/`

**Import Updates**:
- `use crate::models::` → `use crate::`
- `use crate::gguf::` → `use worker_gguf::`
- `use worker_orcd::models::` → `use worker_models::`

**Fixes**:
- Stubbed `CudaError` in GPT module (GPT is currently a stub)
- Moved `announce_stub_mode!` macro with common test module

---

## Remaining Migrations ⏳

### 4. worker-common (Pending)
**Source Files**:
- `bin/worker-orcd/src/error.rs`
- `bin/worker-orcd/src/sampling_config.rs`
- `bin/worker-orcd/src/inference_result.rs`
- `bin/worker-orcd/src/startup.rs`

**Estimated Time**: 1-2 hours  
**Complexity**: Low (simple data types)

### 5. worker-http (Pending)
**Source Files**:
- `bin/worker-orcd/src/http/` (7 files)
  - `execute.rs`
  - `health.rs`
  - `mod.rs`
  - `routes.rs`
  - `server.rs`
  - `sse.rs`
  - `validation.rs`

**Estimated Time**: 2-3 hours  
**Complexity**: Medium (depends on worker-common)

---

## Migration Scripts Created

### Completed
- ✅ `tools/worker-crates-migration/migrate-worker-gguf.sh` (not needed, already had stub)
- ✅ `tools/worker-crates-migration/migrate-worker-tokenizer-v2.sh`
- ✅ `tools/worker-crates-migration/migrate-worker-models-v2.sh`

### TODO
- ⏳ `tools/worker-crates-migration/migrate-worker-common-v2.sh`
- ⏳ `tools/worker-crates-migration/migrate-worker-http-v2.sh`

---

## Verification Status

### worker-gguf ✅
```bash
$ cargo test -p worker-gguf
test result: ok. 5 passed; 0 failed
```

### worker-tokenizer ✅
```bash
$ cargo test -p worker-tokenizer --lib
test result: ok. 80 passed; 0 failed; 4 ignored

$ cargo test -p worker-tokenizer --tests
test result: ok. 46 passed; 0 failed
```

### worker-models ✅
```bash
$ cargo test -p worker-models --lib
test result: ok. 38 passed; 0 failed

$ cargo test -p worker-models --tests
test result: ok. 8 passed; 0 failed
```

### worker-orcd ✅
```bash
$ cargo check -p worker-orcd
Finished `dev` profile [unoptimized + debuginfo] target(s) in 3.69s
```

---

## Workspace Integration

### Cargo.toml Updates

**worker-orcd dependencies added**:
```toml
worker-gguf = { path = "../worker-crates/worker-gguf" }
worker-tokenizer = { path = "../worker-crates/worker-tokenizer" }
worker-models = { path = "../worker-crates/worker-models" }
```

**worker-orcd lib.rs updates**:
- Removed `pub mod tokenizer;`
- Removed `pub mod models;`
- Removed `pub mod util;`

**worker-orcd test declarations removed**:
- `tokenizer_conformance_qwen`
- `phi3_tokenizer_conformance`
- `utf8_edge_cases`
- `adapter_integration`
- `adapter_factory_integration`

---

## Git History Preservation

All migrations used `git mv` to preserve full commit history:

```bash
$ git log --follow bin/worker-crates/worker-tokenizer/src/lib.rs
# Shows full history from worker-orcd/src/tokenizer/mod.rs

$ git log --follow bin/worker-crates/worker-models/src/lib.rs
# Shows full history from worker-orcd/src/models/mod.rs
```

---

## Next Steps

### Immediate (Next Session)
1. **Migrate worker-common** (~1-2 hours)
   - Move `error.rs`, `sampling_config.rs`, `inference_result.rs`, `startup.rs`
   - Update imports in worker-orcd
   - Verify compilation

2. **Migrate worker-http** (~2-3 hours)
   - Move `src/http/` directory
   - Add dependencies: `axum`, `tower`, `tokio`, `futures`
   - Update imports to use worker-common
   - Verify compilation

3. **Final Verification**
   - Run full test suite: `cargo test --workspace`
   - Verify worker-orcd still compiles with CUDA
   - Check binary size (should be similar)

### Future (After All Migrations)
1. **Refactor worker-orcd**
   - Implement `ComputeBackend` trait for CUDA
   - Clean up remaining CUDA-specific code
   - Verify all integration tests pass

2. **Begin worker-aarmd**
   - Implement `ComputeBackend` trait for Metal
   - Reuse all shared crates
   - Verify 85% code reuse achieved

---

## Lessons Learned

### What Went Well ✅
- `git mv` preserved full commit history
- Migration scripts provided safety with dry-run mode
- Incremental approach (one crate at a time) caught issues early
- Test coverage validated correctness after each migration

### Challenges Encountered ⚠️
1. **Import path updates**: Required careful sed replacements
   - `use crate::tokenizer::` → `use crate::`
   - `use worker_orcd::` → `use worker_tokenizer::`

2. **Missing dependencies**: Had to add to Cargo.toml
   - `tracing`, `tokenizers`, `tempfile` for worker-tokenizer
   - `toml` for worker-models tests

3. **Shared test utilities**: `announce_stub_mode!` macro needed migration
   - Moved `tests/common/` module to worker-models

4. **Stub implementations**: GPT model referenced `CudaError`
   - Created stub error type until CUDA implementation

### Improvements for Remaining Migrations
1. **Pre-check dependencies**: Scan for `use` statements before migration
2. **Move test utilities first**: Identify shared test code early
3. **Verify imports**: Run `cargo check` immediately after file moves
4. **Update Cargo.toml atomically**: Add all dependencies in one step

---

## Timeline

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| 0 | Scaffold worker-crates | 1 hour | ✅ Complete |
| 1.1 | Migrate worker-gguf | N/A | ✅ Already done |
| 1.2 | Migrate worker-tokenizer | 2 hours | ✅ Complete |
| 1.3 | Migrate worker-models | 2 hours | ✅ Complete |
| 1.4 | Migrate worker-common | 1-2 hours | ⏳ Pending |
| 1.5 | Migrate worker-http | 2-3 hours | ⏳ Pending |
| 2.0 | Refactor worker-orcd | 1 day | ⏳ Pending |
| **TOTAL** | **Phases 0-1** | **~5-7 hours** | **60% Complete** |

---

## References

- **Development Plan**: `.docs/WORKER_AARMD_DEVELOPMENT_PLAN.md`
- **Scaffold Complete**: `.docs/WORKER_CRATES_SCAFFOLD_COMPLETE.md`
- **Migration Tools**: `.docs/MIGRATION_TOOLS_COMPLETE.md`
- **Test Strategy**: `.docs/TEST_MIGRATION_STRATEGY.md`
- **BDD Decision**: `.docs/BDD_MIGRATION_DECISION.md`
- **Migration Scripts**: `tools/worker-crates-migration/`

---

**Status**: ✅ 3/5 crates migrated (60% complete)  
**Next Action**: Migrate worker-common (~1-2 hours)  
**Estimated Completion**: 1-2 more sessions
