# Test Migration Strategy

**Date**: 2025-10-05  
**Status**: Documented  
**Issue**: Tests must be migrated along with source code

---

## Problem

Initial migration scripts only moved source code, **not tests**. This would result in:
- ❌ Lost test coverage in extracted crates
- ❌ Duplicate tests in worker-orcd
- ❌ Incomplete verification

## Solution

Tests are migrated based on their location:

### 1. Embedded Tests (in source files)

**Location**: `#[cfg(test)]` modules within source files  
**Migration**: Automatic (moves with source file)

**Example: worker-gguf**
```rust
// bin/worker-orcd/src/gguf/mod.rs (lines 219-273)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen_metadata() { /* ... */ }
    
    #[test]
    fn test_phi3_metadata() { /* ... */ }
    
    // ... 5 tests total
}
```

**Migration**:
```bash
# When we move mod.rs, tests move automatically
git mv bin/worker-orcd/src/gguf/mod.rs \
       bin/worker-crates/worker-gguf/src/lib.rs
# ✅ Tests included automatically
```

### 2. Separate Test Files

**Location**: `src/<module>/tests/` or `src/tests/<module>/`  
**Migration**: Explicit move required

**Example: worker-tokenizer** (hypothetical)
```
bin/worker-orcd/src/tokenizer/
├── mod.rs
├── encoder.rs
├── decoder.rs
└── tests/
    ├── encoder_tests.rs
    └── decoder_tests.rs
```

**Migration**:
```bash
# Move entire directory
git mv bin/worker-orcd/src/tokenizer \
       bin/worker-crates/worker-tokenizer/src/

# Tests move with directory
# ✅ src/tokenizer/tests/ → worker-tokenizer/src/tests/
```

### 3. Integration Tests

**Location**: `tests/` directory (crate root)  
**Migration**: Case-by-case decision

**Example: worker-orcd integration tests**
```
bin/worker-orcd/tests/
├── qwen_integration.rs          # Uses GGUF + tokenizer + model
├── phi3_integration.rs          # Uses GGUF + tokenizer + model
├── gpt_integration.rs           # Uses GGUF + tokenizer + model
└── tokenizer_conformance_qwen.rs # Uses tokenizer only
```

**Decision Matrix**:

| Test File | Tests What | Action |
|-----------|------------|--------|
| `qwen_integration.rs` | Full worker (CUDA + GGUF + tokenizer) | ❌ Keep in worker-orcd |
| `phi3_integration.rs` | Full worker (CUDA + GGUF + tokenizer) | ❌ Keep in worker-orcd |
| `gpt_integration.rs` | Full worker (CUDA + GGUF + tokenizer) | ❌ Keep in worker-orcd |
| `tokenizer_conformance_qwen.rs` | Tokenizer only | ✅ Move to worker-tokenizer |

**Rationale**: Integration tests that require CUDA/FFI stay in worker-orcd. Pure Rust tests move to shared crates.

### 4. Benchmarks

**Location**: `benches/` directory  
**Migration**: Move if benchmark is pure Rust

**Example**:
```bash
# If benchmark tests GGUF parsing only
git mv bin/worker-orcd/benches/gguf_parsing.rs \
       bin/worker-crates/worker-gguf/benches/parsing.rs

# If benchmark tests full inference (CUDA)
# ❌ Keep in worker-orcd
```

---

## Migration Rules

### Rule 1: Embedded Tests Always Move
```rust
// These move automatically with source file
#[cfg(test)]
mod tests { /* ... */ }
```

### Rule 2: Pure Rust Tests Move
```bash
# If test has no FFI dependencies, move it
git mv bin/worker-orcd/src/tokenizer/tests/ \
       bin/worker-crates/worker-tokenizer/src/tests/
```

### Rule 3: FFI Tests Stay
```bash
# If test requires CUDA/Metal/FFI, keep in worker binary
# Example: tests/qwen_integration.rs (needs CUDA)
# ❌ Do NOT move
```

### Rule 4: Update Test Imports
```rust
// Before (in worker-orcd)
use crate::gguf::GGUFMetadata;

// After (in worker-gguf)
use super::GGUFMetadata;  // or use crate::GGUFMetadata;
```

---

## Per-Crate Test Migration

### worker-gguf

**Tests to migrate**:
- ✅ `#[cfg(test)] mod tests` in `mod.rs` (5 tests)
  - `test_qwen_metadata`
  - `test_phi3_metadata`
  - `test_gpt2_metadata`
  - `test_rope_freq_base`
  - `test_context_length`

**Tests to keep in worker-orcd**:
- ❌ None (GGUF parsing is pure Rust)

**Action**: Automatic (tests embedded in source file)

### worker-tokenizer

**Tests to migrate**:
- ✅ `#[cfg(test)]` modules in all tokenizer files
- ✅ `tests/tokenizer_conformance_qwen.rs` (pure Rust)
- ✅ `tests/phi3_tokenizer_conformance.rs` (pure Rust)
- ✅ `tests/utf8_edge_cases.rs` (pure Rust)

**Tests to keep in worker-orcd**:
- ❌ Integration tests that use CUDA (e.g., full inference with tokenization)

**Action**: Move directory + cherry-pick integration tests

### worker-models

**Tests to migrate**:
- ✅ `#[cfg(test)]` modules in model adapter files
- ✅ `tests/adapter_factory_integration.rs` (pure Rust)
- ✅ `tests/adapter_integration.rs` (pure Rust)

**Tests to keep in worker-orcd**:
- ❌ `tests/qwen_integration.rs` (needs CUDA)
- ❌ `tests/phi3_integration.rs` (needs CUDA)
- ❌ `tests/gpt_integration.rs` (needs CUDA)

**Action**: Move directory + keep CUDA integration tests

### worker-common

**Tests to migrate**:
- ✅ `#[cfg(test)]` modules in common files
- ✅ Any pure Rust tests for sampling, errors, etc.

**Tests to keep in worker-orcd**:
- ❌ None (common types are pure Rust)

**Action**: Automatic (tests embedded in source files)

### worker-http

**Tests to migrate**:
- ✅ `#[cfg(test)]` modules in HTTP files
- ✅ `tests/http_server_integration.rs` (pure Rust, uses mock backend)
- ✅ `tests/sse_streaming_integration.rs` (pure Rust)
- ✅ `tests/execute_endpoint_integration.rs` (if mockable)

**Tests to keep in worker-orcd**:
- ❌ Tests that require real CUDA inference

**Action**: Move directory + cherry-pick integration tests

---

## Updated Migration Scripts

### migrate-worker-gguf.sh (Updated)

```bash
# Step 3: Move source file (tests embedded)
git mv bin/worker-orcd/src/gguf/mod.rs \
       bin/worker-crates/worker-gguf/src/lib.rs
# ✅ 5 tests move automatically

# Step 8: Verify tests
cargo test -p worker-gguf
# Expected: 5 tests pass
```

### migrate-worker-tokenizer.sh (TODO)

```bash
# Step 3: Move entire directory (includes tests/)
git mv bin/worker-orcd/src/tokenizer \
       bin/worker-crates/worker-tokenizer/src/

# Step 4: Move integration tests (pure Rust only)
git mv bin/worker-orcd/tests/tokenizer_conformance_qwen.rs \
       bin/worker-crates/worker-tokenizer/tests/conformance_qwen.rs

git mv bin/worker-orcd/tests/phi3_tokenizer_conformance.rs \
       bin/worker-crates/worker-tokenizer/tests/conformance_phi3.rs

git mv bin/worker-orcd/tests/utf8_edge_cases.rs \
       bin/worker-crates/worker-tokenizer/tests/utf8_edge_cases.rs

# Step 8: Verify tests
cargo test -p worker-tokenizer
# Expected: All unit tests + 3 integration tests pass
```

### migrate-worker-models.sh (TODO)

```bash
# Step 3: Move directory
git mv bin/worker-orcd/src/models \
       bin/worker-crates/worker-models/src/

# Step 4: Move pure Rust integration tests
git mv bin/worker-orcd/tests/adapter_factory_integration.rs \
       bin/worker-crates/worker-models/tests/factory_integration.rs

git mv bin/worker-orcd/tests/adapter_integration.rs \
       bin/worker-crates/worker-models/tests/adapter_integration.rs

# Keep CUDA integration tests in worker-orcd:
# - qwen_integration.rs (needs CUDA)
# - phi3_integration.rs (needs CUDA)
# - gpt_integration.rs (needs CUDA)

# Step 8: Verify tests
cargo test -p worker-models
# Expected: Unit tests + 2 integration tests pass
```

---

## Verification Checklist

After each migration:

- [ ] Source code moved with `git mv`
- [ ] Embedded tests moved automatically
- [ ] Separate test files moved explicitly
- [ ] Integration tests categorized (pure Rust vs FFI)
- [ ] Pure Rust integration tests moved
- [ ] FFI integration tests kept in worker-orcd
- [ ] Test imports updated
- [ ] `cargo test -p <crate>` passes
- [ ] Test count verified (expected vs actual)
- [ ] Git history preserved for test files

---

## Test Count Tracking

| Crate | Unit Tests | Integration Tests | Total | Status |
|-------|------------|-------------------|-------|--------|
| worker-gguf | 5 (embedded) | 0 | 5 | ✅ Documented |
| worker-tokenizer | ~20 (embedded) | 3 (moved) | ~23 | ⏳ TODO |
| worker-models | ~10 (embedded) | 2 (moved) | ~12 | ⏳ TODO |
| worker-common | ~5 (embedded) | 0 | ~5 | ⏳ TODO |
| worker-http | ~8 (embedded) | 2 (moved) | ~10 | ⏳ TODO |
| **worker-orcd** | **Remaining** | **~15 (CUDA)** | **~15** | **Keep** |

---

## BDD Tests — Do NOT Migrate

**Critical Decision**: ❌ **BDD tests stay in worker-orcd**, do NOT create BDD for worker-crates.

**Rationale**:
- BDD tests verify integration behavior (CUDA + GGUF + tokenizer + HTTP)
- worker-crates are low-level libraries (unit tests sufficient)
- BDD setup is complex (~2-4 hours per crate)
- worker-orcd BDD can use extracted crates

**See**: `.docs/BDD_MIGRATION_DECISION.md` for full analysis.

---

## Summary

**Tests ARE migrated**, but strategy depends on test type:

1. ✅ **Embedded tests** (`#[cfg(test)]`) — Move automatically with source
2. ✅ **Pure Rust integration tests** — Move explicitly with `git mv`
3. ❌ **FFI integration tests** (CUDA/Metal) — Keep in worker binary
4. ❌ **BDD tests** — Keep in worker binary (do NOT migrate)
5. ✅ **Git history preserved** — All test moves use `git mv`

**Result**: Shared crates have unit tests, worker binaries keep integration/BDD tests.

---

## References

- **Migration Scripts**: `tools/worker-crates-migration/`
- **Development Plan**: `.docs/WORKER_AARMD_DEVELOPMENT_PLAN.md`
- **worker-orcd Tests**: `bin/worker-orcd/tests/` (24 integration tests)
