# Sprint 7 Test Guide

Quick reference for running Sprint 7 tests.

---

## Quick Start

### Run All Tests (Fast - No GPU Required)
```bash
cd bin/worker-orcd
cargo test
```

### Run Integration Tests with Real Models (Requires GPU)
```bash
cargo test --tests --features cuda -- --ignored
```

### Run Specific Test Suites
```bash
# Haiku Anti-Cheat (M0 Success Criteria)
REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat -- --ignored

# Performance Baseline
cargo test --test performance_baseline -- --ignored

# UTF-8 Edge Cases
cargo test --test utf8_streaming_edge_cases -- --ignored

# All Models Integration
cargo test --test all_models_integration -- --ignored

# OOM Recovery
cargo test --test oom_recovery -- --ignored

# Cancellation
cargo test --test cancellation_integration -- --ignored

# Final Validation
cargo test --test final_validation -- --ignored

# Gate 4 Checkpoint
cargo test --test gate4_checkpoint
```

---

## Test Categories

### Unit Tests (Fast)
Run without GPU, test logic only:
```bash
cargo test --lib
cargo test minute_to_words
cargo test utf8_validation
cargo test performance_calculation
```

### Integration Tests (Slow)
Require real models and GPU:
```bash
cargo test --test haiku_generation_anti_cheat -- --ignored
cargo test --test performance_baseline -- --ignored
```

### Validation Tests
Check M0 requirements:
```bash
cargo test --test final_validation -- --ignored
cargo test --test gate4_checkpoint
```

---

## Test Models Required

Place models in `.test-models/`:
```
.test-models/
├── qwen/
│   └── qwen2.5-0.5b-instruct-q4_k_m.gguf
└── gpt/
    └── gpt-oss-20b-mxfp4.gguf
```

---

## Environment Variables

```bash
# Require real GPU (no mocks)
export REQUIRE_REAL_LLAMA=1

# Set run ID for artifacts
export LLORCH_RUN_ID=test-run-001

# Enable verbose output
export RUST_LOG=debug
```

---

## Test Artifacts

Artifacts are saved to `.test-results/`:

```bash
# View haiku test results
cat .test-results/haiku/*/test_report.md

# View performance baselines
cat .test-results/performance/*.json

# View Gate 4 report
cat .test-results/gate4/gate4_report.md
```

---

## CI/CD

Tests run automatically in CI:
- **On Push**: All unit tests + integration tests (stub mode)
- **On Main**: + Performance benchmarks
- **Manual**: Full integration tests with real models

---

## Troubleshooting

### Tests Hang
- Check GPU availability: `nvidia-smi`
- Reduce timeout in test
- Check model paths

### OOM Errors
- Reduce batch size
- Use smaller model
- Check VRAM: `nvidia-smi`

### Compilation Errors
- Update dependencies: `cargo update`
- Clean build: `cargo clean && cargo build`
- Check Rust version: `rustc --version`

---

## M0 Validation Checklist

Run these tests to validate M0:

```bash
# 1. Unit tests
cargo test --lib

# 2. Integration tests
cargo test --tests

# 3. Haiku anti-cheat (M0 success criteria)
REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat -- --ignored

# 4. Performance baseline
cargo test --test performance_baseline -- --ignored

# 5. Final validation
cargo test --test final_validation -- --ignored

# 6. Gate 4 checkpoint
cargo test --test gate4_checkpoint

# 7. Check report
cat .test-results/gate4/gate4_report.md
```

If all pass: **M0 COMPLETE** ✅

---

Built by Foundation-Alpha 🏗️
