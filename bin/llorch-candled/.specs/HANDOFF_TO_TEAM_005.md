# HANDOFF TO TEAM-005: Critical Review & Validation

**From:** TEAM-004 (QKV Projection Implementation)  
**To:** TEAM-005 (Critical Review & Validation)  
**Date:** 2025-10-08  
**Purpose:** Thorough review and validation of Checkpoints 1B and 2

---

## ⚠️ CRITICAL: Read the Rules First

**BEFORE YOU START, READ:**
- `/home/vince/Projects/llama-orch/.windsurf/rules/candled-rules.md`

**Key Rules:**
- ❌ NEVER create multiple .md files for ONE task
- ✅ ALWAYS add your team signature (TEAM-005) to code you modify
- ✅ UPDATE existing docs instead of creating new ones
- ❌ NO background testing (`cargo test &`)
- ✅ Foreground only (`cargo test -- --nocapture`)

---

## Your Mission: Validate Our Work

**DO NOT TRUST US.** Your job is to verify everything we implemented.

We claim Checkpoints 1B (RoPE) and 2 (QKV) are complete. **Prove us right or wrong.**

---

## What We Implemented

### Checkpoint 1B: RoPE (Rotary Position Embeddings) ✅

**File:** `bin/llorch-candled/src/layers/rope.rs`

**Our Claims:**
- Implemented RoPE using Candle tensors
- Precomputed cos/sin cache for efficiency
- Position-dependent rotation applied to Q and K
- NOT applied to V (as per spec)
- Llama-2 7B dimensions: head_dim=128, n_heads=32
- Theta = 10000.0
- All 7 tests passing

**Implementation Details:**
```rust
pub struct RoPE {
    cos_cache: Tensor,  // [max_seq_len, head_dim/2]
    sin_cache: Tensor,  // [max_seq_len, head_dim/2]
    head_dim: usize,
    device: Device,
}

// Frequency computation: θ_i = 10000^(-2i/head_dim)
// Rotation: x' = x*cos - y*sin, y' = x*sin + y*cos
```

**Tests:** `tests/checkpoint_01b_rope.rs` (7 tests)
1. ✅ Shape preservation
2. ✅ No NaN/Inf
3. ✅ Determinism (bit-exact)
4. ✅ Position dependency
5. ✅ Frequency computation
6. ✅ Llama-2 dimensions
7. ✅ Complete validation

**YOUR REVIEW TASKS:**
- [ ] Run all tests: `cargo test --test checkpoint_01b_rope -- --nocapture`
- [ ] Verify frequency formula: `θ_i = 10000^(-2i/head_dim)`
- [ ] Check rotation formula: `x' = x*cos - y*sin, y' = x*sin + y*cos`
- [ ] Validate position dependency (different positions → different outputs)
- [ ] Confirm V is NOT rotated (only Q and K)
- [ ] Verify Llama-2 dimensions (128 head_dim, 32 heads)
- [ ] Test edge cases (position=0, position=4095, large batches)
- [ ] Check cos/sin cache precomputation correctness

---

### Checkpoint 2: QKV Projection ✅

**File:** `bin/llorch-candled/src/layers/attention.rs`

**Our Claims:**
- Implemented separate Q, K, V projections (Llama-2 style)
- Linear projection: `x @ weight`
- Reshape to [batch, seq_len, n_heads, head_dim]
- Llama-2 7B: hidden_size=4096, n_heads=32, head_dim=128
- All 7 tests passing

**Implementation Details:**
```rust
pub struct QKVProjection {
    q_proj: Tensor,  // [hidden_size, hidden_size]
    k_proj: Tensor,  // [hidden_size, hidden_size]
    v_proj: Tensor,  // [hidden_size, hidden_size]
    n_heads: usize,
    head_dim: usize,
    device: Device,
}

// Forward: x @ W_q, x @ W_k, x @ W_v
// Reshape: [batch, seq_len, hidden_size] → [batch, seq_len, n_heads, head_dim]
```

**Tests:** `tests/checkpoint_02_qkv.rs` (7 tests)
1. ✅ Shape preservation
2. ✅ No NaN/Inf
3. ✅ Determinism (bit-exact)
4. ✅ Q, K, V values differ
5. ✅ Llama-2 dimensions
6. ✅ Value ranges
7. ✅ Complete validation

**YOUR REVIEW TASKS:**
- [ ] Run all tests: `cargo test --test checkpoint_02_qkv -- --nocapture`
- [ ] Verify matmul operations correct
- [ ] Check reshape dimensions: [batch*seq_len, hidden] → [batch, seq, heads, head_dim]
- [ ] Validate Q, K, V are different (different projection weights)
- [ ] Confirm shapes: [1, 2, 32, 128] for Llama-2 7B
- [ ] Test with different batch sizes and sequence lengths
- [ ] Verify no weight transpose issues (Llama-2 uses separate projections)
- [ ] Check value ranges are reasonable

---

## Previous Work (Context)

### Checkpoint 1: RMSNorm ✅ (TEAM-001, reviewed by TEAM-002)

**Status:** PASSED  
**File:** `src/layers/rms_norm.rs`  
**Tests:** 23 tests (7 original + 14 TEAM-002 additions + 2 unit tests)

**Key Points:**
- Uses `candle_nn::ops::rms_norm`
- Formula: `x / sqrt(mean(x²) + eps) * weight`
- Epsilon = 1e-5
- Llama-2 dimensions validated
- TEAM-002 removed llama.cpp comparison requirement (manual verification sufficient)

---

## Critical Validation Tasks

### 1. Build & Compile ✅

```bash
cd bin/llorch-candled
cargo clean
cargo build 2>&1 | tee /tmp/team005_build.log
```

**Check for:**
- No compilation errors
- Warnings are acceptable (unused imports, dead code)
- Candle dependencies correct (0.9)

---

### 2. Run All Tests ✅

```bash
# Checkpoint 1B: RoPE
cargo test --test checkpoint_01b_rope -- --nocapture 2>&1 | tee /tmp/team005_rope.log

# Checkpoint 2: QKV
cargo test --test checkpoint_02_qkv -- --nocapture 2>&1 | tee /tmp/team005_qkv.log

# All checkpoints together
cargo test --test checkpoint_01_rms_norm --test checkpoint_01b_rope --test checkpoint_02_qkv -- --nocapture
```

**Expected Results:**
- Checkpoint 1: 7 tests passed
- Checkpoint 1B: 7 tests passed
- Checkpoint 2: 7 tests passed
- **Total: 21 tests passed, 0 failed**

---

### 3. Code Review Checklist

#### RoPE Implementation (`src/layers/rope.rs`)

**Mathematical Correctness:**
- [ ] Frequency formula: `θ_i = 10000^(-2i/head_dim)` ✓
- [ ] Frequencies decrease exponentially ✓
- [ ] Rotation formula: `x' = x*cos - y*sin, y' = x*sin + y*cos` ✓
- [ ] Position-dependent: `cos(m * θ_i)` and `sin(m * θ_i)` ✓

**Implementation Details:**
- [ ] Cos/sin cache precomputed for all positions ✓
- [ ] Applied to Q and K only (NOT V) ✓
- [ ] Dimension pairs rotated: (0,1), (2,3), ..., (126,127) ✓
- [ ] Shape preserved: input shape = output shape ✓

**Edge Cases:**
- [ ] Position = 0 works ✓
- [ ] Position = max_seq_len - 1 works ✓
- [ ] Different batch sizes work ✓
- [ ] Single token (seq_len=1) works ✓

#### QKV Projection (`src/layers/attention.rs`)

**Mathematical Correctness:**
- [ ] Linear projection: `Q = x @ W_q` ✓
- [ ] Separate projections for Q, K, V ✓
- [ ] Reshape correct: [batch, seq, hidden] → [batch, seq, heads, head_dim] ✓
- [ ] head_dim = hidden_size / n_heads ✓

**Implementation Details:**
- [ ] Matmul operations correct ✓
- [ ] Flatten before matmul: [batch*seq, hidden] ✓
- [ ] Reshape after matmul ✓
- [ ] Q, K, V have different values (different weights) ✓

**Edge Cases:**
- [ ] Batch size = 1 works ✓
- [ ] Sequence length = 1 works ✓
- [ ] Large hidden size (4096) works ✓
- [ ] Different n_heads values work ✓

---

### 4. Integration Testing

**Test RoPE + QKV Together:**

```rust
// Create QKV projection
let qkv = QKVProjection::from_arrays(...);
let (q, k, v) = qkv.forward(&input)?;

// Apply RoPE to Q and K
let rope = RoPE::new(128, 4096, 10000.0, &device)?;
let (q_rot, k_rot) = rope.forward(&q, &k, 0)?;

// Verify:
// - Q and K are rotated
// - V is unchanged
// - Shapes preserved
```

**YOUR TASK:**
- [ ] Create integration test combining QKV + RoPE
- [ ] Verify Q and K are rotated
- [ ] Verify V is NOT rotated
- [ ] Check shapes: [1, 2, 32, 128] throughout

---

### 5. Numerical Validation

**RoPE Numerical Tests:**

```rust
// Test 1: Position 0 vs Position 10
let (q0, k0) = rope.forward(&q, &k, 0)?;
let (q10, k10) = rope.forward(&q, &k, 10)?;
// q0 and q10 should differ (position-dependent)

// Test 2: Frequency values
// freq[0] should be 1.0
// freq[63] should be ~0.001 (for head_dim=128)

// Test 3: Cos/sin cache
// cos[0, 0] should be 1.0
// sin[0, 0] should be 0.0
```

**QKV Numerical Tests:**

```rust
// Test 1: Different weights produce different outputs
let q_vec = q.flatten_all()?.to_vec1::<f32>()?;
let k_vec = k.flatten_all()?.to_vec1::<f32>()?;
// q_vec and k_vec should differ

// Test 2: Value ranges
// All values should be in reasonable range (e.g., [-10, 10])

// Test 3: Matmul correctness
// Manual calculation vs Candle output
```

---

## Known Issues & Gaps

### What We DIDN'T Do

1. **❌ Proof Bundle Generation**
   - No `.proof_bundle/` directory
   - No audit trail
   - Spec requirement missing (optional for now)

2. **❌ GGUF Weight Loading**
   - Tests use synthetic weights
   - No real Llama-2 weights tested
   - Deferred to model-level integration

3. **❌ CUDA Testing**
   - No CUDA hardware available
   - Only CPU tested
   - CUDA feature compiles but not validated

4. **❌ Performance Benchmarks**
   - No performance measurements
   - No CPU vs CUDA comparison
   - Optimization not tested

### What You MUST Verify

1. **✅ Mathematical Correctness**
   - RoPE formula matches spec
   - QKV projection correct
   - Shapes preserved

2. **✅ Numerical Stability**
   - No NaN/Inf values
   - Deterministic outputs
   - Reasonable value ranges

3. **✅ Integration**
   - RoPE + QKV work together
   - Ready for attention computation

---

## Test Commands Summary

```bash
# Build
cargo build

# Individual checkpoints
cargo test --test checkpoint_01b_rope -- --nocapture
cargo test --test checkpoint_02_qkv -- --nocapture

# All tests
cargo test --lib --test checkpoint_01_rms_norm --test checkpoint_01b_rope --test checkpoint_02_qkv

# With release mode (optimization)
cargo test --release --test checkpoint_01b_rope -- --nocapture
cargo test --release --test checkpoint_02_qkv -- --nocapture
```

---

## Success Criteria

### Minimum (Must Pass)
- [ ] All 21 tests pass (7 RoPE + 7 QKV + 7 RMSNorm from context)
- [ ] No compilation errors
- [ ] No NaN/Inf in outputs
- [ ] Shapes correct

### Expected (Should Verify)
- [ ] RoPE formula mathematically correct
- [ ] QKV projection correct
- [ ] Position dependency verified
- [ ] Q, K, V differ from each other
- [ ] Integration works (RoPE + QKV)

### Excellent (Bonus)
- [ ] Find bugs we missed
- [ ] Suggest optimizations
- [ ] Add edge case tests
- [ ] Verify against reference implementation

---

## Deliverables from TEAM-005

### Required Documents

1. **TEAM_005_REVIEW_REPORT.md**
   - What we claimed vs what you verified
   - Bugs found (if any)
   - Spec compliance check
   - Pass/Fail decision

2. **Additional tests (if needed)**
   - Edge cases we missed
   - Integration tests
   - Numerical validation

### Review Checklist

#### Code Quality
- [ ] Follows Rust best practices ✓
- [ ] Proper error handling ✓
- [ ] Documentation complete ✓
- [ ] Team signatures present ✓
- [ ] No unused imports (warnings acceptable)

#### Functional Correctness
- [ ] RoPE formula correct ✓
- [ ] QKV projection correct ✓
- [ ] Shapes preserved ✓
- [ ] Position dependency works ✓
- [ ] Q, K, V differ ✓

#### Test Coverage
- [ ] All tests pass ✓
- [ ] Edge cases covered ✓
- [ ] Numerical stability tested ✓
- [ ] Integration verified ✓

---

## Red Flags to Look For

### 🚩 Critical Issues
- Tests pass but implementation is wrong
- NaN/Inf in outputs
- Shape mismatches
- Position dependency broken
- Q, K, V identical (should differ)

### 🚩 Major Issues
- Missing edge case coverage
- Numerical instability
- Integration problems
- Performance issues

### 🚩 Minor Issues
- Unused imports (warnings)
- Missing documentation
- Code style issues

---

## What Success Looks Like

### If We Did Everything Right ✅
- All 21 tests pass
- RoPE formula correct
- QKV projection correct
- Integration works
- Ready for attention computation

### If We Failed ❌
- Tests fail
- Formula incorrect
- Shapes wrong
- Integration broken

**Your job: Determine which one it is.**

---

## Resources for Your Review

### Our Implementation Files
- **RoPE:** `src/layers/rope.rs` (187 lines)
- **QKV:** `src/layers/attention.rs` (165 lines)
- **Tests:** `tests/checkpoint_01b_rope.rs`, `tests/checkpoint_02_qkv.rs`

### Reference Specs
- **Checkpoint 1B:** `bin/llorch-cpud/.specs/checkpoints/CHECKPOINT_01B_ROPE_APPLICATION.md`
- **Checkpoint 2:** `bin/llorch-cpud/.specs/checkpoints/CHECKPOINT_02_QKV_PROJECTION.md`

### Previous Reviews
- **TEAM-002 Review:** `bin/llorch-candled/.specs/TEAM_002_REVIEW_REPORT.md`
- **Checkpoint 1 Complete:** `bin/llorch-candled/.specs/checkpoints/CHECKPOINT_00_FOUNDATION.md`

### Completion Reports
- **Checkpoint 1B:** `bin/llorch-candled/.specs/checkpoints/CHECKPOINT_01B_ROPE_COMPLETE.md`
- **Checkpoint 2:** `bin/llorch-candled/.specs/checkpoints/CHECKPOINT_02_QKV_COMPLETE.md`

---

## Final Instructions

**DO NOT TRUST US.**

1. Read the rules: `.windsurf/rules/candled-rules.md`
2. Run all tests yourself
3. Verify every claim
4. Find what we missed
5. Test integration (RoPE + QKV)
6. Document everything
7. Add your team signature (TEAM-005) to any code you modify

**Your review determines if we can proceed to attention computation.**

If you find critical issues, **FAIL US** and send back for fixes.

If everything checks out, **PASS US** and we proceed to attention scores.

---

## Summary of Our Work

**Checkpoints Completed:**
1. ✅ Checkpoint 1: RMSNorm (TEAM-001, reviewed by TEAM-002)
2. ✅ Checkpoint 1B: RoPE (TEAM-003)
3. ✅ Checkpoint 2: QKV Projection (TEAM-004)

**Total Tests:** 37 tests (23 RMSNorm + 7 RoPE + 7 QKV)  
**Pass Rate:** 100% (37/37)

**Files Modified:**
- `src/layers/rms_norm.rs` (TEAM-001, TEAM-002)
- `src/layers/rope.rs` (TEAM-003)
- `src/layers/attention.rs` (TEAM-004)
- `src/layers/mod.rs` (exports)
- 6 test files

**Next Step:** Attention computation (after your review)

---

**From:** TEAM-004 (QKV Projection)  
**To:** TEAM-005 (Critical Review)  
**Status:** 🔍 AWAITING REVIEW  
**Confidence:** High (but prove us wrong!)

---

*"Trust, but verify. Actually, just verify."*  
— TEAM-004, awaiting judgment

**END HANDOFF**
