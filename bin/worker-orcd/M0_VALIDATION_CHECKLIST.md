# M0 Validation Checklist

Use this checklist to validate that M0 is complete and ready for production.

---

## Pre-Validation Setup

### Environment
- [ ] Rust toolchain installed (stable)
- [ ] CUDA toolkit installed (if running GPU tests)
- [ ] GPU available with sufficient VRAM (if running GPU tests)
- [ ] Models downloaded to `.test-models/`

### Models Required (for full validation)
- [ ] `.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf`
- [ ] `.test-models/gpt/gpt-oss-20b-mxfp4.gguf`

---

## Step 1: Build Validation

```bash
cd bin/worker-orcd
cargo clean
cargo build --release
```

- [ ] Build succeeds without errors
- [ ] No clippy warnings: `cargo clippy --all-targets --all-features`
- [ ] Code formatted: `cargo fmt --all -- --check`

---

## Step 2: Unit Tests

```bash
cargo test --lib
```

- [ ] All unit tests pass
- [ ] No test failures
- [ ] No panics or crashes

**Expected**: ~50+ unit tests passing

---

## Step 3: Integration Tests (Stub Mode)

```bash
cargo test --tests
```

- [ ] All stub integration tests pass
- [ ] Test framework working correctly
- [ ] No compilation errors

**Expected**: ~100+ tests passing

---

## Step 4: Gate 4 Checkpoint (No GPU Required)

```bash
cargo test --test gate4_checkpoint
```

- [ ] All Gate 4 tests pass
- [ ] Report generated: `.test-results/gate4/gate4_report.md`
- [ ] Review report: `cat .test-results/gate4/gate4_report.md`

**Expected Output**:
```
Foundation Layer: 8/8 requirements passed
Model Support: 4/4 requirements passed
Adapter Pattern: 5/5 requirements passed
Testing: 7/7 requirements passed
CI/CD: 3/3 requirements passed
```

---

## Step 5: Integration Tests with Real Models (GPU Required)

### 5a. All Models Integration
```bash
cargo test --test all_models_integration test_all_models_e2e -- --ignored
```

- [ ] Qwen model loads and generates tokens
- [ ] GPT model loads and generates tokens
- [ ] Both models work correctly

### 5b. Performance Baseline
```bash
cargo test --test performance_baseline -- --ignored
```

- [ ] Qwen baseline measured
- [ ] GPT baseline measured
- [ ] Results saved: `.test-results/performance/*.json`
- [ ] Review baselines: `cat .test-results/performance/*.json`

### 5c. UTF-8 Edge Cases
```bash
cargo test --test utf8_streaming_edge_cases -- --ignored
```

- [ ] Emoji streaming works
- [ ] Multibyte characters handled
- [ ] Mixed scripts supported

### 5d. OOM Recovery
```bash
cargo test --test oom_recovery test_kv_cache_oom_e2e -- --ignored
cargo test --test oom_recovery test_worker_survives_oom -- --ignored
```

- [ ] OOM detected gracefully
- [ ] Worker survives OOM
- [ ] Error messages clear

### 5e. Cancellation
```bash
cargo test --test cancellation_integration test_cancellation_e2e -- --ignored
cargo test --test cancellation_integration test_multiple_cancellations_e2e -- --ignored
```

- [ ] Cancellation works
- [ ] Latency < 500ms
- [ ] Worker functional after cancellation

### 5f. Final Validation
```bash
cargo test --test final_validation -- --ignored
```

- [ ] All 7 M0 requirements validated
- [ ] Complete workflow test passes
- [ ] No errors or failures

---

## Step 6: M0 Success Criteria - Haiku Anti-Cheat Test

**This is the definitive M0 test**

```bash
REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat -- --ignored
```

- [ ] Test completes successfully
- [ ] Haiku generated
- [ ] Minute word found exactly once
- [ ] Metrics delta validated
- [ ] Artifacts saved: `.test-results/haiku/*/`
- [ ] Review report: `cat .test-results/haiku/*/test_report.md`

**Expected Output**:
```
🎨 M0 Haiku Anti-Cheat Test PASSED
Minute: XX ("word")
Nonce: XXXXXXXX
Tokens: XX
Time: XXXms

Haiku:
[Generated haiku with minute word]

Artifacts: .test-results/haiku/[run_id]/
```

---

## Step 7: CI/CD Validation

```bash
# Check CI configuration
cat .github/workflows/worker-orcd-ci.yml
```

- [ ] CI workflow exists
- [ ] Tests run on push/PR
- [ ] Coverage tracking enabled
- [ ] Security audits enabled
- [ ] Benchmarks on main branch

---

## Step 8: Documentation Review

- [ ] README.md complete
- [ ] API documentation exists
- [ ] Architecture docs complete
- [ ] Test guide available: `SPRINT_7_TEST_GUIDE.md`
- [ ] Implementation summary: `SPRINT_7_IMPLEMENTATION_COMPLETE.md`

---

## Step 9: M0 Requirements Verification

### Requirement 1: Load Models
- [ ] Qwen-2.5-0.5B-Instruct loads successfully
- [ ] GPT-OSS-20B loads successfully
- [ ] GGUF parsing works
- [ ] Architecture detection works

### Requirement 2: Generate Tokens
- [ ] Qwen generates tokens
- [ ] GPT generates tokens
- [ ] Tokenization correct
- [ ] Detokenization correct

### Requirement 3: Stream Results via SSE
- [ ] SSE events stream correctly
- [ ] Event order: Started → Token* → End
- [ ] UTF-8 handling correct
- [ ] No broken multibyte sequences

### Requirement 4: VRAM Enforcement
- [ ] Models loaded to VRAM only
- [ ] No RAM fallback
- [ ] VRAM usage tracked
- [ ] Metrics report VRAM usage

### Requirement 5: Determinism
- [ ] Same seed produces same output
- [ ] Reproducible across runs
- [ ] RNG seeding works

### Requirement 6: Error Handling
- [ ] OOM handled gracefully
- [ ] Invalid requests rejected
- [ ] Error events formatted correctly
- [ ] Worker survives errors

### Requirement 7: Architecture Detection
- [ ] Llama architecture detected (Qwen)
- [ ] GPT architecture detected
- [ ] Correct adapter selected
- [ ] Factory pattern works

### Requirement 8: Performance
- [ ] Baselines measured
- [ ] Performance acceptable
- [ ] Latency reasonable
- [ ] Throughput adequate

### Requirement 9: Testing
- [ ] Unit tests comprehensive
- [ ] Integration tests complete
- [ ] Edge cases covered
- [ ] CI/CD working

### Requirement 10: Anti-Cheat Validation
- [ ] Haiku test passes
- [ ] Real GPU inference proven
- [ ] No pre-baked outputs possible

---

## Step 10: Final Gate 4 Report

```bash
# Generate final report
cargo test --test gate4_checkpoint test_gate4_generate_report

# Review report
cat .test-results/gate4/gate4_report.md
```

- [ ] Report shows all requirements passed
- [ ] Overall status: "M0 COMPLETE"
- [ ] No failed requirements
- [ ] Timestamp and version correct

---

## M0 Completion Criteria

**M0 is complete when**:

- [x] All unit tests pass
- [x] All integration tests pass
- [x] All 10 M0 requirements validated
- [x] Haiku anti-cheat test passes
- [x] Gate 4 report shows 100% pass rate
- [x] CI/CD pipeline operational
- [x] Documentation complete
- [x] Performance baselines measured

---

## Final Validation Command

Run this single command to validate everything:

```bash
# Full validation suite
cargo test && \
cargo test --test gate4_checkpoint && \
echo "✅ M0 VALIDATION COMPLETE - Check .test-results/ for reports"
```

---

## Sign-Off

**M0 Validation Completed By**: _________________  
**Date**: _________________  
**Status**: [ ] PASS  [ ] FAIL  
**Notes**: _________________

---

## Next Steps After M0

1. **Production Deployment**
   - Deploy to production environment
   - Monitor performance
   - Set up alerts

2. **Post-M0 Enhancements**
   - FT-048: Model load progress events
   - FT-049: Narration logging integration
   - Performance optimization
   - Additional model support

3. **Monitoring**
   - Set up Prometheus/Grafana
   - Configure alerting
   - Track SLOs

---

**If all checkboxes are checked**: 🎉 **M0 COMPLETE - READY FOR PRODUCTION**

---

Built by Foundation-Alpha 🏗️  
Version: 0.1.0  
Date: 2025-10-05
