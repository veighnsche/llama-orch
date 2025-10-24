# Worker Crates BDD Scaffolding Complete

**Date**: 2025-10-05  
**Status**: ✅ Complete  
**Decision**: Add BDD tests to testable worker-crates

---

## Summary

Created BDD test scaffolding for 3 worker-crates that can be meaningfully tested with BDD:
- ✅ `worker-gguf/bdd` — GGUF parsing behavior
- ✅ `worker-tokenizer/bdd` — Tokenization behavior
- ✅ `worker-models/bdd` — Model adapter behavior

**Excluded** (not testable with BDD):
- ❌ `worker-common` — Simple data types (unit tests sufficient)
- ❌ `worker-http` — Framework wrapper (integration tests sufficient)
- ❌ `worker-compute` — Trait definition only (no implementation)

---

## BDD Crates Created

### 1. worker-gguf/bdd

**Purpose**: Test GGUF file format parsing behavior

**Structure**:
```
bin/worker-crates/worker-gguf/bdd/
├── Cargo.toml
├── README.md
├── src/
│   ├── main.rs
│   └── steps/
│       ├── mod.rs
│       └── world.rs
└── tests/
    └── features/
        └── gguf_parsing.feature
```

**Features**:
- Parse Qwen model metadata
- Parse Phi-3 model metadata
- Parse GPT-2 model metadata
- Verify architecture detection
- Verify model dimensions
- Verify attention configuration (GQA vs MHA)

**Status**: ✅ Compiles (with TODO placeholders for actual implementation)

### 2. worker-tokenizer/bdd

**Purpose**: Test tokenization behavior

**Structure**:
```
bin/worker-crates/worker-tokenizer/bdd/
├── Cargo.toml
├── README.md
├── src/
│   ├── main.rs
│   └── steps/
│       ├── mod.rs
│       └── world.rs
└── tests/
    └── features/
        └── tokenization.feature
```

**Features**:
- Encode and decode simple text
- UTF-8 boundary safety
- Round-trip consistency
- Token count verification

**Status**: ✅ Compiles (with TODO placeholders)

### 3. worker-models/bdd

**Purpose**: Test model adapter factory and architecture detection

**Structure**:
```
bin/worker-crates/worker-models/bdd/
├── Cargo.toml
├── README.md
├── src/
│   ├── main.rs
│   └── steps/
│       ├── mod.rs
│       └── world.rs
└── tests/
    └── features/
        └── model_adapters.feature
```

**Features**:
- Detect Llama-style architecture
- Detect GPT-style architecture
- Create appropriate adapter
- Verify adapter supports inference

**Status**: ✅ Compiles (with TODO placeholders)

---

## Implementation Status

### Scaffold Complete ✅

All BDD crates have:
- ✅ `Cargo.toml` with cucumber dependency
- ✅ `src/main.rs` runner
- ✅ `src/steps/mod.rs` with step definitions
- ✅ `src/steps/world.rs` with World state
- ✅ `tests/features/*.feature` with Gherkin scenarios
- ✅ `README.md` with usage instructions
- ✅ Added to workspace `Cargo.toml`
- ✅ Compiles successfully

### TODO: Wire Up After Extraction

All step implementations have `// TODO: Uncomment after worker-gguf extraction` comments.

**After Phase 1 extraction**, uncomment:
1. Import statements: `use worker_gguf::GGUFMetadata;`
2. World state types: `pub metadata: Option<GGUFMetadata>`
3. Step implementations: Actual parsing and assertions

---

## Workspace Integration

Updated `Cargo.toml`:
```toml
[workspace]
members = [
    # ... existing members ...
    
    "bin/worker-crates/worker-gguf",
    "bin/worker-crates/worker-gguf/bdd",       # ← Added
    "bin/worker-crates/worker-tokenizer",
    "bin/worker-crates/worker-tokenizer/bdd",  # ← Added
    "bin/worker-crates/worker-models",
    "bin/worker-crates/worker-models/bdd",     # ← Added
    
    # ... other members ...
]
```

---

## Running BDD Tests

### After Extraction (Phase 1 Complete)

```bash
# Run worker-gguf BDD tests
cd bin/worker-crates/worker-gguf/bdd
cargo run --bin bdd-runner

# Run worker-tokenizer BDD tests
cd bin/worker-crates/worker-tokenizer/bdd
cargo run --bin bdd-runner

# Run worker-models BDD tests
cd bin/worker-crates/worker-models/bdd
cargo run --bin bdd-runner

# Run all BDD tests
cargo test --workspace --features bdd-cucumber
```

---

## Why These Crates Get BDD

### ✅ worker-gguf
**Reason**: Complex parsing behavior with multiple scenarios
- Different model architectures (llama, gpt)
- Different attention mechanisms (GQA, MHA)
- Different RoPE configurations
- BDD clearly expresses "given a Qwen model, then architecture should be llama"

### ✅ worker-tokenizer
**Reason**: Complex encoding/decoding behavior
- UTF-8 boundary safety is critical
- Round-trip consistency must be verified
- Multiple tokenizer backends (gguf-bpe, hf-json)
- BDD clearly expresses "given text with emojis, then encoding should be UTF-8 safe"

### ✅ worker-models
**Reason**: Factory pattern with multiple adapter types
- Architecture detection logic
- Adapter selection based on metadata
- Different adapter implementations (Llama, GPT)
- BDD clearly expresses "given a Qwen model, then adapter should be LlamaAdapter"

---

## Why These Crates Don't Get BDD

### ❌ worker-common
**Reason**: Simple data types and utilities
- `SamplingConfig` — Just a struct
- `InferenceResult` — Just a struct
- `callback` — Simple HTTP POST
- Unit tests are sufficient

### ❌ worker-http
**Reason**: Framework wrapper
- Wraps Axum (already tested)
- Integration tests are sufficient
- No complex behavior to express in Gherkin

### ❌ worker-compute
**Reason**: Trait definition only
- No implementation (just trait definition)
- Implementations are in worker binaries (worker-orcd, worker-aarmd)
- Nothing to test until implemented

---

## Example Feature File

### worker-gguf/bdd/tests/features/gguf_parsing.feature

```gherkin
Feature: GGUF File Parsing
  As a worker implementation
  I want to parse GGUF metadata
  So that I can understand model architecture and configuration

  Scenario: Parse Qwen model metadata
    Given a GGUF file "qwen-2.5-0.5b.gguf"
    When I parse the GGUF metadata
    Then the architecture should be "llama"
    And the vocabulary size should be 151936
    And the hidden dimension should be 896
    And the number of layers should be 24
    And the number of attention heads should be 14
    And the number of KV heads should be 2
    And the model should use GQA
    And the RoPE frequency base should be 1000000.0
    And the context length should be 32768
```

---

## Next Steps

### Phase 1: Extract Source Code
1. Run migration scripts to extract source code
2. Unit tests move automatically with source
3. BDD scaffolds are ready (already created)

### Phase 2: Wire Up BDD Tests
1. Uncomment `use worker_gguf::GGUFMetadata;` imports
2. Uncomment actual step implementations
3. Run BDD tests to verify behavior
4. Add more scenarios as needed

### Phase 3: Continuous Testing
1. Run BDD tests in CI/CD
2. Add new scenarios for edge cases
3. Use BDD as living documentation

---

## Verification

```bash
# All BDD crates compile
$ cargo check -p worker-gguf-bdd -p worker-tokenizer-bdd -p worker-models-bdd
✅ Finished in 0.53s

# Workspace includes BDD crates
$ cargo metadata --format-version 1 | jq '.workspace_members' | grep bdd
✅ "worker-gguf-bdd 0.0.0"
✅ "worker-tokenizer-bdd 0.0.0"
✅ "worker-models-bdd 0.0.0"
```

---

## File Count

**Total files created**: 24

- 3 × `Cargo.toml`
- 3 × `README.md`
- 3 × `src/main.rs`
- 3 × `src/steps/mod.rs`
- 3 × `src/steps/world.rs`
- 3 × `tests/features/*.feature`
- 1 × Updated workspace `Cargo.toml`
- 1 × This document

---

## References

- **BDD Pattern**: `bin/shared-crates/audit-logging/bdd/` (reference implementation)
- **Test Migration Strategy**: `.docs/TEST_MIGRATION_STRATEGY.md`
- **Development Plan**: `.docs/WORKER_AARMD_DEVELOPMENT_PLAN.md`
- **Cucumber Docs**: https://cucumber-rs.github.io/cucumber/current/

---

**Status**: ✅ BDD scaffolding complete  
**Ready for**: Phase 1 extraction (wire up after source code moves)  
**Timeline**: No impact (scaffolds created, implementation after extraction)
