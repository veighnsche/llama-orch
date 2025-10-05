# BDD Migration Decision — Do NOT Migrate BDD Tests

**Date**: 2025-10-05  
**Status**: ⚠️ Critical Decision  
**Issue**: BDD tests should NOT be migrated to worker-crates

---

## Problem

BDD (Behavior-Driven Development) tests in llama-orch follow a specific pattern:

```
bin/<crate>/
├── src/
│   └── lib.rs
├── bdd/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs
│   │   └── steps/
│   │       ├── world.rs
│   │       ├── assertions.rs
│   │       └── validation.rs
│   └── tests/
│       └── features/
│           └── *.feature
└── Cargo.toml
```

**Current BDD crates in workspace:**
- `bin/orchestratord/bdd`
- `bin/pool-managerd/bdd`
- `bin/worker-orcd/bdd`
- `bin/orchestratord-crates/orchestrator-core/bdd`
- `bin/pool-managerd-crates/model-catalog/bdd`
- `bin/pool-managerd-crates/model-provisioner/bdd`
- `bin/shared-crates/audit-logging/bdd`
- `bin/shared-crates/input-validation/bdd`
- `bin/shared-crates/narration-core/bdd`
- `bin/shared-crates/secrets-management/bdd`

---

## Why NOT Migrate BDD Tests

### 1. **BDD Tests Are Integration Tests**

BDD tests verify **behavior across components**, not isolated units:

```gherkin
# Example: worker-orcd BDD test (hypothetical)
Feature: Worker GGUF Loading
  Scenario: Load Qwen model
    Given a worker-orcd instance
    And a GGUF file "qwen-0.5b.gguf"
    When I load the model
    Then the model should be in VRAM
    And the worker should report ready
```

This test requires:
- ❌ CUDA context (worker-orcd specific)
- ❌ HTTP server (worker-orcd specific)
- ❌ Full worker lifecycle (worker-orcd specific)
- ✅ GGUF parsing (worker-gguf)

**Conclusion**: This is a **worker-orcd integration test**, not a worker-gguf unit test.

### 2. **BDD Setup Is Complex**

Each BDD crate requires:
- Separate `Cargo.toml` with cucumber dependency
- `src/main.rs` runner
- `src/steps/world.rs` (World state management)
- `src/steps/assertions.rs` (Custom assertions)
- `tests/features/*.feature` (Gherkin scenarios)

**Cost**: Creating BDD setup for each worker-crate = ~2-4 hours per crate × 6 crates = **12-24 hours**

**Benefit**: Minimal (unit tests already cover isolated functionality)

### 3. **BDD Tests Belong to Binaries**

Looking at existing patterns:

```
bin/orchestratord/bdd/           ✅ Tests orchestratord binary
bin/pool-managerd/bdd/           ✅ Tests pool-managerd binary
bin/worker-orcd/bdd/             ✅ Tests worker-orcd binary

bin/shared-crates/audit-logging/bdd/  ✅ Tests audit-logging library
bin/shared-crates/input-validation/bdd/ ✅ Tests input-validation library
```

**Pattern**: BDD tests exist for:
- ✅ **Binaries** (orchestratord, pool-managerd, worker-orcd)
- ✅ **Critical shared libraries** (audit-logging, input-validation)
- ❌ **Not for every crate**

### 4. **worker-crates Are Low-Level Libraries**

```
worker-gguf       → GGUF parser (pure data structure)
worker-tokenizer  → Tokenization (pure algorithm)
worker-models     → Model adapters (pure logic)
worker-common     → Common types (pure data)
worker-http       → HTTP server (framework wrapper)
worker-compute    → Trait definition (no implementation)
```

**These are low-level utilities**, not user-facing behaviors. BDD is overkill.

### 5. **Unit Tests Are Sufficient**

```rust
// worker-gguf has 5 unit tests (sufficient)
#[test]
fn test_qwen_metadata() { /* ... */ }

#[test]
fn test_phi3_metadata() { /* ... */ }

// No need for BDD:
// Feature: GGUF Parsing
//   Scenario: Parse Qwen metadata
//     Given a GGUF file "qwen.gguf"
//     When I parse the metadata
//     Then the architecture should be "llama"
// (This is just a verbose unit test!)
```

---

## Decision: Keep BDD in worker-orcd Only

### ✅ DO: Keep BDD Tests in worker-orcd

```
bin/worker-orcd/
├── src/
│   └── (uses worker-crates)
├── bdd/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs
│   │   └── steps/
│   │       └── world.rs
│   └── tests/
│       └── features/
│           ├── model_loading.feature
│           ├── inference_execution.feature
│           └── sse_streaming.feature
└── Cargo.toml
```

**BDD tests verify**:
- Full worker lifecycle (CUDA + GGUF + tokenizer + HTTP)
- End-to-end inference flow
- SSE streaming behavior
- Error handling across components

### ❌ DON'T: Create BDD for worker-crates

```
bin/worker-crates/worker-gguf/
├── src/
│   └── lib.rs (with #[cfg(test)] unit tests)
├── bdd/  ❌ DON'T CREATE THIS
└── Cargo.toml
```

**Reason**: Unit tests are sufficient for isolated libraries.

---

## Migration Strategy (Updated)

### Phase 1: Extract Source + Unit Tests

```bash
# Extract worker-gguf
git mv bin/worker-orcd/src/gguf/mod.rs \
       bin/worker-crates/worker-gguf/src/lib.rs
# ✅ Unit tests move automatically

# Update worker-orcd to use worker-gguf
# ✅ BDD tests stay in worker-orcd/bdd/
```

### Phase 2: Update worker-orcd BDD Tests

```toml
# bin/worker-orcd/bdd/Cargo.toml
[dependencies]
worker-gguf = { path = "../../worker-crates/worker-gguf" }
worker-tokenizer = { path = "../../worker-crates/worker-tokenizer" }
worker-models = { path = "../../worker-crates/worker-models" }
worker-common = { path = "../../worker-crates/worker-common" }
worker-http = { path = "../../worker-crates/worker-http" }
worker-orcd = { path = ".." }  # Main binary
```

```rust
// bin/worker-orcd/bdd/src/steps/world.rs
use worker_gguf::GGUFMetadata;  // ✅ Can use extracted crates
use worker_orcd::WorkerOrcd;    // ✅ Still tests full worker
```

**Result**: BDD tests can use extracted crates but test full worker integration.

---

## Comparison: Unit Tests vs BDD Tests

### Unit Tests (worker-gguf)

```rust
#[test]
fn test_qwen_metadata() {
    let metadata = GGUFMetadata::from_file("qwen.gguf").unwrap();
    assert_eq!(metadata.architecture().unwrap(), "llama");
    assert_eq!(metadata.vocab_size().unwrap(), 151936);
}
```

**Scope**: Isolated GGUF parsing  
**Dependencies**: None  
**Speed**: Fast (~1ms)  
**Purpose**: Verify data structure correctness

### BDD Tests (worker-orcd)

```gherkin
Feature: Worker Model Loading
  Scenario: Load Qwen model to VRAM
    Given a CUDA device is available
    And a GGUF file "qwen-0.5b.gguf" exists
    When I start worker-orcd with the model
    Then the model should be loaded to VRAM
    And the worker should report ready
    And the VRAM usage should be ~500MB
```

**Scope**: Full worker integration (CUDA + GGUF + HTTP)  
**Dependencies**: CUDA, filesystem, HTTP server  
**Speed**: Slow (~5-10s)  
**Purpose**: Verify end-to-end behavior

---

## When to Create BDD Tests

### ✅ Create BDD for:
- **Binaries** (orchestratord, pool-managerd, worker-orcd)
- **Critical shared libraries** (audit-logging, input-validation)
- **Complex workflows** (multi-step processes)
- **User-facing behavior** (API contracts, error messages)

### ❌ Don't Create BDD for:
- **Low-level utilities** (parsers, tokenizers, adapters)
- **Data structures** (types, configs, errors)
- **Trait definitions** (worker-compute)
- **Simple wrappers** (HTTP helpers)

---

## Updated Test Strategy

### worker-gguf
- ✅ **Unit tests**: 5 tests in `#[cfg(test)]` module
- ❌ **BDD tests**: None (not needed)
- ✅ **Integration tests**: Via worker-orcd BDD

### worker-tokenizer
- ✅ **Unit tests**: ~20 tests in `#[cfg(test)]` modules
- ✅ **Integration tests**: 3 pure Rust tests in `tests/`
- ❌ **BDD tests**: None (not needed)
- ✅ **Integration tests**: Via worker-orcd BDD

### worker-models
- ✅ **Unit tests**: ~10 tests in `#[cfg(test)]` modules
- ✅ **Integration tests**: 2 pure Rust tests in `tests/`
- ❌ **BDD tests**: None (not needed)
- ✅ **Integration tests**: Via worker-orcd BDD

### worker-common
- ✅ **Unit tests**: ~5 tests in `#[cfg(test)]` modules
- ❌ **BDD tests**: None (not needed)

### worker-http
- ✅ **Unit tests**: ~8 tests in `#[cfg(test)]` modules
- ✅ **Integration tests**: 2 pure Rust tests in `tests/`
- ❌ **BDD tests**: None (not needed)
- ✅ **Integration tests**: Via worker-orcd BDD

### worker-orcd (binary)
- ✅ **Unit tests**: Remaining CUDA-specific tests
- ✅ **Integration tests**: ~15 CUDA integration tests
- ✅ **BDD tests**: Full worker behavior tests
  - Model loading
  - Inference execution
  - SSE streaming
  - Error handling
  - Cancellation
  - VRAM management

---

## Summary

**Decision**: ❌ **Do NOT migrate BDD tests to worker-crates**

**Rationale**:
1. BDD tests are for integration, not isolated units
2. BDD setup is complex and time-consuming
3. Unit tests are sufficient for low-level libraries
4. BDD tests belong to binaries (worker-orcd)
5. worker-orcd BDD can use extracted crates

**Result**:
- ✅ worker-crates have unit tests (sufficient)
- ✅ worker-orcd keeps BDD tests (integration)
- ✅ worker-orcd BDD uses extracted crates
- ✅ No duplicate BDD setup needed
- ✅ Saves 12-24 hours of work

**Updated Timeline**: No change (BDD not migrated)

---

## References

- **Test Migration Strategy**: `.docs/TEST_MIGRATION_STRATEGY.md`
- **BDD Pattern**: `bin/shared-crates/audit-logging/bdd/` (example)
- **worker-orcd BDD**: `bin/worker-orcd/bdd/` (placeholder)
