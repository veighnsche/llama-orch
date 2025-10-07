# 🪡 TEAM THIMBLE — Q-Projection Stride Investigation

**Date:** 2025-10-07T00:25Z  
**Mission:** Find and fix Q projection extreme-values bug (±16 spikes at Q[95], Q[126])  
**Status:** ❌ STRIDE HYPOTHESIS DISPROVEN — BUG REMAINS

---

## Executive Summary

**Hypothesis Tested:** CUBLAS_OP_T stride interpretation causes incorrect memory walks beyond row 0.

**Result:** **DISPROVEN**. Extremes persist even with explicit CPU transpose + CUBLAS_OP_N.

**Key Finding:** Manual dot products produce correct values (±0.08), but cuBLAS returns extreme values (±16) at **the same output indices** regardless of transpose method.

---

## Evidence Chain

### 1. Reproducible Extremes ✅

Logged Q[0], Q[95], Q[126] for tokens 0 & 1:

| Token | Q[0] | Q[95] | Q[126] | Notes |
|-------|------|-------|--------|-------|
| 0 | -0.043 ✅ | -16.047 ❌ | 14.336 ❌ | Head 1, dims 31 & 62 |
| 1 | -0.015 ✅ | -3.912 ❌ | 3.695 ❌ | Same indices! |

**Observation:** Extremes always at indices 95 and 126 (head 1, dimensions 31 and 62).

### 2. Manual Parity Check ✅

Computed manual dot products for Q[95] and Q[126]:

```
Token 0:
  Q[95]:  manual=-0.058, cuBLAS=-16.047, diff=15.99 ❌
  Q[126]: manual=0.055, cuBLAS=14.336, diff=14.28 ❌

Token 1:
  Q[95]:  manual=0.079, cuBLAS=-3.912, diff=3.99 ❌
  Q[126]: manual=0.020, cuBLAS=3.695, diff=3.68 ❌
```

**Observation:** Manual calculation (host-side FP32) produces normal values. cuBLAS produces extremes.

### 3. Pre-transpose Experiment ✅

**Method:**
- Explicitly transposed Q weight [896,896] on CPU
- Used `CUBLAS_OP_N` with `lda=q_dim` instead of `CUBLAS_OP_T`
- This eliminates any stride interpretation issues

**Result:**
```
Token 0: Q[95]=-16.047, Q[126]=14.336 (NO CHANGE!)
Token 1: Q[95]=-3.912, Q[126]=3.695 (NO CHANGE!)
```

**Observation:** Extremes persist at **exact same indices** with both OP_T and OP_N.

### 4. Input Verification ✅

Logged normed input stats:
```
Token 0: min=-0.576@[741], max=1.038@[75], mean=0.003
Token 1: min=-0.542@[190], max=0.425@[75], mean=0.001
```

**Observation:** Input is normal (±1.0 range). No spikes that could cause extremes.

---

## Conclusion

**The bug is NOT about CUBLAS_OP_T stride semantics.**

The extremes persist regardless of:
- Transpose operation (OP_T vs OP_N)
- Memory layout (original vs explicitly transposed)
- Input values (normed is normal)

This eliminates the stride hypothesis and points to:
1. cuBLAS compute type issue (tensor core fast-math errors)
2. Weight corruption at specific columns (95, 126)
3. FP16 accumulation overflow (though manual FP32 calc works)
4. Deeper cuBLAS or CUDA driver bug

---

## Code Changes

### Files Modified

1. **`cuda/src/transformer/qwen_transformer.cpp`**:
   - Lines 6-17: Pre-transpose experiment banner with outcomes
   - Lines 139-151: CPU transpose helper function
   - Lines 420-489: Pre-transpose experiment code (disabled)
   - Lines 668-779: Q-projection outlier diagnosis with:
     - Input (normed) stats logging
     - Q[0], Q[95], Q[126] logging with head/dim mapping
     - Manual parity check (disabled, outcomes documented)

2. **`investigation-teams/TEAM_HELIOS_HANDOFF.md`**:
   - Lines 236-346: TEAM THIMBLE findings
   - Lines 310-345: Crisp TODO list for next team

### Documentation Improvements

All experimental code now has:
- **Banner comments** with objective, method, expected/observed outcomes
- **Guard clarity** with usage notes (`Set to 1 to enable...`)
- **Index provenance** explaining why 95/126 were chosen
- **Memory model** documentation for manual parity checks
- **Outcome stubs** with final observed numbers
- **Pointer lifetime** notes for static allocations

---

## Next Team TODO

Execute in order:

1. **Test CUBLAS_COMPUTE_32F** (line 488):
   - Replace `CUBLAS_COMPUTE_32F_FAST_16F` with `CUBLAS_COMPUTE_32F`
   - If extremes disappear → tensor core fast-math is the bug
   - If extremes persist → proceed to step 2

2. **Dump Q weight columns 95 and 126**:
   - Check for extreme values (>10) in weight columns
   - If columns have extremes → weight corruption
   - If columns are normal → proceed to step 3

3. **Test FP32 GEMM**:
   - Convert Q weight to FP32, run with `CUDA_R_32F`
   - If extremes disappear → FP16 accumulation overflow
   - If extremes persist → deeper cuBLAS/driver issue

4. **Verify normed input** (already logged):
   - Check for spikes >5 or <-5
   - If spikes exist → bug is upstream (RMSNorm)
   - If normal → bug is in Q GEMM

---

## Test Command

```bash
cd bin/worker-orcd
REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda --release -- --ignored --nocapture --test-threads=1
```

Look for `[TEAM THIMBLE]` logs in output.

---

**TEAM THIMBLE**  
**Status:** Investigation complete, hypothesis disproven, handoff ready  
**Time:** 2025-10-07T00:25Z

*"Fast iteration, clean evidence, crisp handoff."* 🪡
