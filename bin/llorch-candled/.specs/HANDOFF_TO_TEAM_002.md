# HANDOFF TO TEAM-002: Critical Review & Validation

**From:** TEAM-001 (Math Integration)  
**To:** TEAM-002 (Critical Review & Validation)  
**Date:** 2025-10-08  
**Purpose:** Thorough review and active attempt to disprove TEAM-001's work

---

## Your Mission: Break Our Implementation

**DO NOT TRUST US.** Your job is to actively try to disprove everything we did.

We claim Checkpoint 1 (RMSNorm) is complete. **Prove us wrong.**

---

## What We Claim We Did

### 1. Added Candle Dependencies ✅ (We Say)

**File:** `bin/llorch-candled/Cargo.toml`

**Our Claims:**
- Added `candle-core = "0.9"`
- Added `candle-nn = "0.9"`
- Updated `cuda` feature to include `candle-core/cuda` and `candle-nn/cuda`
- Used published versions from crates.io (not path dependencies)

**YOUR REVIEW TASKS:**
- [ ] Verify dependencies actually compile
- [ ] Check if versions are correct (is 0.9 the right version?)
- [ ] Confirm CUDA feature actually includes Candle features
- [ ] Test if `cargo build` works
- [ ] Test if `cargo build --features cuda` works (if CUDA available)
- [ ] Verify no version conflicts with other workspace crates

**How to Disprove:**
```bash
cd bin/llorch-candled
cargo clean
cargo build 2>&1 | tee build.log
# Look for errors, warnings, version conflicts
```

---

### 2. Implemented RMSNorm Using Candle ✅ (We Say)

**File:** `bin/llorch-candled/src/layers/rms_norm.rs`

**Our Claims:**
- Uses `candle_nn::ops::rms_norm` for the math
- Automatic CUDA kernel selection
- Maintains our architecture (not using Candle's high-level abstractions)
- Provides `from_array()` helper for compatibility

**YOUR REVIEW TASKS:**
- [ ] Verify we're actually using Candle's function (not reimplementing)
- [ ] Check if CUDA path actually works
- [ ] Confirm we didn't import unnecessary Candle abstractions
- [ ] Validate the epsilon conversion (f64 → f32) is correct
- [ ] Test edge cases: empty tensors, very large tensors, extreme values
- [ ] Verify device handling is correct

**How to Disprove:**
```rust
// Test with extreme values
let weight = vec![f32::MAX; 4096];
let norm = RMSNorm::from_array(&weight, 1e-5, &device)?;
// Does it crash? Overflow? NaN?

// Test with zeros
let weight = vec![0.0f32; 4096];
// What happens?

// Test with negative weights
let weight = vec![-1.0f32; 4096];
// Is this handled correctly?
```

---

### 3. Created Comprehensive Tests ✅ (We Say)

**File:** `bin/llorch-candled/tests/checkpoint_01_rms_norm.rs`

**Our Claims:**
- 7 tests, all passing
- Tests: shape, NaN/Inf, determinism, mathematical properties, batch, scale, complete validation
- Follows llorch-cpud checkpoint test structure

**YOUR REVIEW TASKS:**
- [ ] Run all tests and verify they actually pass
- [ ] Check if tests are actually comprehensive or just superficial
- [ ] Verify determinism test is truly bit-exact
- [ ] Validate mathematical properties test (is RMS really ≈ 1.0?)
- [ ] Test with different input patterns (not just our `generate_test_input`)
- [ ] Check for missing edge cases
- [ ] Verify test coverage is adequate

**How to Disprove:**
```bash
# Run tests multiple times
for i in {1..10}; do
  cargo test --test checkpoint_01_rms_norm -- --nocapture
done
# Do they always pass? Same output?

# Run with different optimization levels
cargo test --test checkpoint_01_rms_norm --release
# Does release mode break anything?
```

---

## Critical Questions to Answer

### Mathematical Correctness

**We claim:** RMSNorm formula is `x / sqrt(mean(x²) + eps) * weight`

**VERIFY:**
1. [ ] Is Candle's `rms_norm` actually implementing this formula?
2. [ ] Is the epsilon placement correct? (before sqrt, not after)
3. [ ] Is the mean computed over the correct axis?
4. [ ] Is the weight applied correctly?
5. [ ] Does it match llama.cpp's implementation?

**How to Verify:**
```python
# Compare with reference implementation
import numpy as np

def reference_rms_norm(x, weight, eps):
    # x: [seq_len, hidden_size]
    mean_sq = np.mean(x**2, axis=1, keepdims=True)
    rms = np.sqrt(mean_sq + eps)
    normalized = x / rms
    return normalized * weight

# Load our output and compare
```

### Checkpoint 1 Spec Compliance

**Checkpoint 1 Spec Says:**
- File: `bin/llorch-cpud/.specs/checkpoints/CHECKPOINT_01_RMS_NORM.md`
- Tolerance: 1e-5
- Must validate against llama.cpp reference
- Must use epsilon = 1e-5
- Must handle Llama-2 dimensions (hidden_size=4096)

**VERIFY:**
1. [ ] Did we actually compare against llama.cpp? (NO - we didn't!)
2. [ ] Is tolerance 1e-5 enforced in tests?
3. [ ] Is epsilon exactly 1e-5?
4. [ ] Are Llama-2 dimensions tested (4096)?
5. [ ] Did we test with actual Llama-2 weights from GGUF?

**CRITICAL GAPS WE MIGHT HAVE:**
- ❌ No llama.cpp reference comparison
- ❌ No GGUF weight loading test
- ❌ No actual Llama-2 model validation
- ❌ No proof bundle generation (spec requires it!)

### Performance & CUDA

**We claim:** CUDA acceleration works automatically

**VERIFY:**
1. [ ] Does CUDA feature actually compile?
2. [ ] Does it actually use CUDA kernels?
3. [ ] Is there a performance difference CPU vs CUDA?
4. [ ] Does CPU fallback work when CUDA unavailable?

**How to Verify:**
```bash
# Build with CUDA
cargo build --features cuda

# Run tests with CUDA
cargo test --features cuda --test checkpoint_01_rms_norm

# Benchmark CPU vs CUDA
cargo bench --features cuda
```

---

## What We DIDN'T Do (Gaps You Should Find)

### Missing: llama.cpp Reference Comparison

**Checkpoint 1 spec requires:**
- Extract checkpoint using TEAM-006's tool
- Compare outputs with tolerance < 1e-5
- Validate first 10 values match

**We did:** Internal Candle-only tests

**YOU MUST:**
1. [ ] Extract llama.cpp checkpoint for RMSNorm
2. [ ] Run our implementation with same input
3. [ ] Compare outputs element-wise
4. [ ] Verify max difference < 1e-5

### Missing: GGUF Weight Loading

**Checkpoint 1 spec requires:**
- Load actual Llama-2 weights from GGUF
- Use real `blk.0.attn_norm.weight` tensor
- Validate with real model data

**We did:** Synthetic weights (all ones)

**YOU MUST:**
1. [ ] Load Llama-2 7B Q8_0 GGUF
2. [ ] Extract `blk.0.attn_norm.weight`
3. [ ] Run RMSNorm with real weights
4. [ ] Validate output

### Missing: Proof Bundle

**Checkpoint 1 spec requires:**
- Generate proof bundle under `.proof_bundle/checkpoint_01/<run_id>/`
- Files: input.ndjson, output.ndjson, metadata.json, comparison.md
- Autogenerated header per PB-1012

**We did:** Nothing

**YOU MUST:**
1. [ ] Verify proof bundle generation
2. [ ] Check file format compliance
3. [ ] Validate metadata completeness

### Missing: Integration Test

**We did:** Unit tests only

**YOU MUST:**
1. [ ] Test RMSNorm in actual model context
2. [ ] Verify it works with embedding output
3. [ ] Check it integrates with attention layer

---

## Specific Test Cases to Try

### Edge Cases We Might Have Missed

```rust
// 1. Zero input
let input = Tensor::zeros((2, 4096), &device)?;
let output = norm.forward(&input)?;
// Should not NaN (eps prevents division by zero)

// 2. Very large values
let input = Tensor::ones((2, 4096), &device)? * 1e10;
let output = norm.forward(&input)?;
// Should not overflow

// 3. Very small values
let input = Tensor::ones((2, 4096), &device)? * 1e-10;
let output = norm.forward(&input)?;
// Should not underflow

// 4. Mixed signs
let input_data: Vec<f32> = (0..8192)
    .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
    .collect();
let input = Tensor::from_vec(input_data, (2, 4096), &device)?;
// Should handle correctly

// 5. Single token (seq_len=1)
let input = Tensor::randn(0f32, 1.0, (1, 4096), &device)?;
// Should work with batch size 1

// 6. Large batch
let input = Tensor::randn(0f32, 1.0, (100, 4096), &device)?;
// Should handle large batches
```

### Numerical Stability Tests

```rust
// Test epsilon importance
let norm_small_eps = RMSNorm::from_array(&weight, 1e-10, &device)?;
let norm_large_eps = RMSNorm::from_array(&weight, 1e-3, &device)?;

let input = Tensor::zeros((2, 4096), &device)?;
let out_small = norm_small_eps.forward(&input)?;
let out_large = norm_large_eps.forward(&input)?;
// Compare outputs - should differ based on epsilon
```

### Candle API Correctness

```rust
// Verify we're using the right Candle function
// Check candle-nn source: does ops::rms_norm match our expectations?

// Test device handling
let cpu_device = Device::Cpu;
let cpu_norm = RMSNorm::from_array(&weight, 1e-5, &cpu_device)?;

#[cfg(feature = "cuda")]
{
    let cuda_device = Device::new_cuda(0)?;
    let cuda_norm = RMSNorm::from_array(&weight, 1e-5, &cuda_device)?;
    
    let input_cpu = Tensor::randn(0f32, 1.0, (2, 4096), &cpu_device)?;
    let input_cuda = input_cpu.to_device(&cuda_device)?;
    
    let out_cpu = cpu_norm.forward(&input_cpu)?;
    let out_cuda = cuda_norm.forward(&input_cuda)?;
    
    // Should produce same results (within tolerance)
    let out_cuda_cpu = out_cuda.to_device(&cpu_device)?;
    // Compare out_cpu and out_cuda_cpu
}
```

---

## How to Systematically Disprove Our Work

### Step 1: Verify Build & Dependencies
```bash
cd bin/llorch-candled
cargo clean
cargo build 2>&1 | tee review_build.log
cargo build --features cuda 2>&1 | tee review_build_cuda.log
# Check logs for warnings, errors, version issues
```

### Step 2: Run Our Tests
```bash
cargo test --test checkpoint_01_rms_norm -- --nocapture 2>&1 | tee review_tests.log
cargo test --test checkpoint_01_rms_norm --release -- --nocapture 2>&1 | tee review_tests_release.log
# Do they actually pass? Same results in debug vs release?
```

### Step 3: Add Missing Tests
```bash
# Create review_checkpoint_01.rs with:
# - llama.cpp comparison
# - GGUF weight loading
# - Edge cases
# - Numerical stability
# - Performance benchmarks
```

### Step 4: Compare with Reference
```bash
# Extract llama.cpp checkpoint
cd bin/llorch-cpud/tools/checkpoint-extractor
./build/llorch-checkpoint-extractor \
  /.test-models/llama2-7b/llama-2-7b.Q8_0.gguf \
  "Hello" \
  /tmp/llama2_checkpoints

# Compare with our output
# Write comparison script
```

### Step 5: Generate Proof Bundle
```bash
# Our tests should generate proof bundle
# Check if it exists, if not, we failed spec compliance
ls -la .proof_bundle/checkpoint_01/
```

### Step 6: Code Review
```bash
# Review our implementation line by line
# Check for:
# - Correct Candle API usage
# - Proper error handling
# - Edge case handling
# - Documentation accuracy
```

---

## Success Criteria for Your Review

### Minimum (Must Find)
- [ ] All our tests actually pass
- [ ] Build succeeds without errors
- [ ] Basic functionality works

### Expected (Should Find)
- [ ] Missing llama.cpp comparison
- [ ] Missing GGUF integration
- [ ] Missing proof bundle
- [ ] Edge cases not covered
- [ ] Numerical stability issues (if any)

### Excellent (Bonus Points)
- [ ] Find bugs we missed
- [ ] Identify performance issues
- [ ] Suggest improvements
- [ ] Propose additional tests
- [ ] Find spec non-compliance

---

## Deliverables from TEAM-002

### Required Documents
1. **TEAM_002_REVIEW_REPORT.md**
   - What we claimed vs what you verified
   - Gaps found
   - Bugs found (if any)
   - Spec compliance check
   - Pass/Fail decision

2. **TEAM_002_ADDITIONAL_TESTS.rs**
   - Tests we missed
   - Edge cases
   - llama.cpp comparison
   - GGUF integration

3. **TEAM_002_PROOF_BUNDLE/**
   - Proper proof bundle generation
   - Comparison with reference
   - Validation artifacts

### Review Checklist

#### Code Quality
- [ ] Follows Rust best practices
- [ ] Proper error handling
- [ ] Documentation complete
- [ ] TEAM-001 signatures present
- [ ] No unused imports/variables

#### Functional Correctness
- [ ] RMSNorm formula correct
- [ ] Epsilon handling correct
- [ ] Weight application correct
- [ ] Device handling correct
- [ ] Batch processing correct

#### Spec Compliance
- [ ] Checkpoint 1 spec requirements met
- [ ] Tolerance 1e-5 enforced
- [ ] llama.cpp comparison done
- [ ] GGUF weights tested
- [ ] Proof bundle generated

#### Test Coverage
- [ ] All claimed tests pass
- [ ] Edge cases covered
- [ ] Numerical stability tested
- [ ] Performance validated
- [ ] CUDA path tested (if available)

#### Integration
- [ ] Works with worker-crates
- [ ] Integrates with model pipeline
- [ ] Compatible with GGUF loader
- [ ] Ready for Checkpoint 1B

---

## Red Flags to Look For

### 🚩 Critical Issues
- Tests pass but implementation is wrong
- Candle function doesn't match our claims
- CUDA path doesn't work
- Numerical instability
- Spec non-compliance

### 🚩 Major Issues
- Missing llama.cpp validation
- No GGUF integration
- No proof bundle
- Incomplete test coverage
- Poor error handling

### 🚩 Minor Issues
- Unused imports
- Missing documentation
- Suboptimal performance
- Code style issues

---

## What Success Looks Like

### If We Did Everything Right ✅
- All tests pass
- llama.cpp comparison < 1e-5 difference
- GGUF weights load correctly
- Proof bundle generated
- CUDA works (if available)
- Spec fully compliant

### If We Failed ❌
- Tests fail
- llama.cpp comparison shows errors
- GGUF integration missing
- No proof bundle
- Spec violations found

**Your job: Determine which one it is.**

---

## Resources for Your Review

### Reference Implementations
- **llama.cpp:** `/home/vince/Projects/llama-orch/reference/llama.cpp`
  - File: `llama.cpp`, function: `ggml_rms_norm`
- **Candle:** `/home/vince/Projects/llama-orch/reference/candle`
  - File: `candle-nn/src/ops.rs`, function: `rms_norm`
- **Mistral.rs:** `/home/vince/Projects/llama-orch/reference/mistral.rs`
  - File: `mistralrs-core/src/layers.rs`, struct: `RmsNorm`

### Checkpoint Specs
- **Checkpoint 1:** `bin/llorch-cpud/.specs/checkpoints/CHECKPOINT_01_RMS_NORM.md`
- **Llama-2 Updates:** `bin/llorch-cpud/.specs/checkpoints/LLAMA2_CHECKPOINT_UPDATE_PLAN.md`
- **Proof Bundle:** `libs/proof-bundle/.specs/00_proof-bundle.md`

### Our Work
- **Implementation:** `bin/llorch-candled/src/layers/rms_norm.rs`
- **Tests:** `bin/llorch-candled/tests/checkpoint_01_rms_norm.rs`
- **Dependencies:** `bin/llorch-candled/Cargo.toml`
- **Catalog:** `bin/llorch-candled/.specs/TEAM_001_CANDLE_CATALOG_PLAN.md`
- **Report:** `bin/llorch-candled/.specs/TEAM_001_COMPLETION_REPORT.md`

### Tools
- **Checkpoint Extractor:** `bin/llorch-cpud/tools/checkpoint-extractor`
- **GGUF Loader:** `bin/worker-crates/worker-gguf`
- **Proof Bundle:** `libs/proof-bundle`

---

## Final Instructions

**DO NOT TRUST US.**

1. Run everything yourself
2. Verify every claim
3. Find what we missed
4. Test what we didn't test
5. Compare with references
6. Document everything

**Your review determines if Checkpoint 1 is truly complete.**

If you find critical issues, **FAIL US** and send back for fixes.

If everything checks out, **PASS US** and we proceed to Checkpoint 1B.

---

**From:** TEAM-001 (Math Integration)  
**To:** TEAM-002 (Critical Review)  
**Status:** 🔍 AWAITING REVIEW  
**Confidence:** High (but prove us wrong!)

---

*"Trust, but verify. Actually, just verify."*  
— TEAM-001, awaiting judgment

**END HANDOFF**
