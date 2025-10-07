# 👒 TEAM TOP HAT — Q-Projection Anomaly Investigation

**Date:** 2025-10-07T00:34Z  
**Mission:** Eliminate ±16 spikes at Q[95] & Q[126] by testing compute type, weight corruption, and input spikes  
**Status:** ❌ ALL HYPOTHESES ELIMINATED — BUG REMAINS UNEXPLAINED

---

## Executive Summary

**Hypotheses Tested:**
- **H1. Compute type/tensor-core fast-math:** ELIMINATED ❌
- **H2. Weight column corruption:** ELIMINATED ❌  
- **H3. Input spikes in normed:** ELIMINATED ❌

**Critical Finding:** cuBLAS GEMM produces extremes (±16) at Q[95] and Q[126] **even with:**
- Full FP32 compute (`CUBLAS_COMPUTE_32F`)
- Normal weight columns (min≈-0.22, max≈0.18)
- Normal inputs (min≈-0.58, max≈1.04)
- Manual FP32 calculation gives correct values (±0.08)

**The bug is deeper than expected.** All standard explanations have been eliminated.

---

## Evidence Chain

### Step 1: Compute Type Test ✅

**Test:** Switch between `CUBLAS_COMPUTE_32F_FAST_16F` (tensor-core fast-math) and `CUBLAS_COMPUTE_32F` (full precision).

**Results:**

| Config | Q[0] | Q[95] | Q[126] | Verdict |
|--------|------|-------|--------|---------|
| FAST_16F (baseline) | -0.043 ✅ | -16.047 ❌ | 14.336 ❌ | Extremes present |
| 32F (full precision) | -0.043 ✅ | -16.047 ❌ | 14.336 ❌ | **IDENTICAL!** |

**Observation:** Extremes persist with full FP32 compute. **H1 ELIMINATED.**

### Step 2: Weight Column Verification ✅

**Test:** Dump Q weight columns 95 and 126 statistics to check for corruption.

**Results:**
```
Column 95: min=-0.217407, max=0.173706, mean=-0.000443
  First 16 values: 0.000155 -0.012703 -0.015511 0.003748 -0.022446 ...
  
Column 126: min=-0.193970, max=0.179932, mean=-0.000864
  First 16 values: 0.006535 0.004253 0.016647 0.003979 0.075378 ...
```

**Observation:** Both columns have normal values (|max| < 0.22). No extremes, NaNs, or inf. **H2 ELIMINATED.**

### Step 3: Input Hot-Spot Check ✅

**Test:** Verify normed input doesn't have spikes that couple into Q[95]/Q[126].

**Results:**
```
Token 0: min=-0.575684@741, max=1.038086@75, mean=0.002826
Token 1: min=-0.541504@190, max=0.424805@75, mean=0.001082
```

**Observation:** Input is normal (roughly ±1 range, no spikes >2). **H3 ELIMINATED.**

### Additional Finding: Pre-Bias Verification ✅

**Test:** Check Q output **before** bias is added to confirm GEMM itself produces extremes.

**Results:**
```
Q before bias: Q[0]=-0.043060 Q[95]=-16.046875 Q[126]=14.335938
```

**Observation:** Extremes are present in raw cuBLAS output, not introduced by bias addition.

---

## Contradiction Analysis

**The Core Mystery:**

1. **Manual calculation (host FP32):** Q[95]≈±0.08, Q[126]≈±0.08 ✅
2. **cuBLAS output (FAST_16F):** Q[95]≈-16, Q[126]≈+14 ❌
3. **cuBLAS output (32F):** Q[95]≈-16, Q[126]≈+14 ❌

**What we've ruled out:**
- ✅ Stride/transpose issues (TEAM THIMBLE pre-transpose experiment)
- ✅ Tensor-core fast-math errors (32F mode shows same extremes)
- ✅ Weight corruption (columns 95/126 are normal)
- ✅ Input spikes (normed is normal)
- ✅ Bias corruption (biases are all zeros)

**What remains:**
- ❓ cuBLAS internal bug or misuse
- ❓ Memory alignment issue
- ❓ CUDA driver/hardware bug
- ❓ Incorrect cuBLAS parameters we haven't identified
- ❓ Something unusual about this specific weight matrix that triggers edge cases

---

## Code Changes

### Files Modified

**`cuda/src/transformer/qwen_transformer.cpp`:**

1. **Lines 420-422:** Early declaration of `top_hat_token_count` and `do_top_hat_log`
2. **Lines 424-487:** TEAM TOP HAT investigation banner and diagnostic code:
   - Step 2: Weight column verification (lines 438-464)
   - Step 3: Input hot-spot check (lines 466-484)
3. **Lines 546-554:** Step 1: Compute type A/B test macro
4. **Lines 566-577:** Pre-bias Q output logging
5. **Lines 880-903:** Post-projection Q logging with bias checks

All changes are guarded by macros:
- `TOP_HAT_Q_GEMM_COMPUTE_32F` (default: 0)
- `TOP_HAT_DUMP_Q_COLS` (default: 1)
- `TOP_HAT_NORMED_HOTSPOTS` (default: 1)

### Logs Generated

```
[TEAM TOP HAT] Q weight col 95 stats: min=... max=... mean=...
[TEAM TOP HAT] Q weight col 126 stats: min=... max=... mean=...
[TEAM TOP HAT] normed stats: min=...@idx max=...@idx mean=...
[TEAM TOP HAT] TOP_HAT_Q_GEMM_COMPUTE_32F=0/1
[TEAM TOP HAT] Q before bias: Q[0]=... Q[95]=... Q[126]=...
[TEAM TOP HAT] START layer=0 pos=... head_dim=... q_dim=... compute32f=...
[TEAM TOP HAT] Q[0]=... Q[95]=... Q[126]=...
[TEAM TOP HAT] Q bias[0]=... bias[95]=... bias[126]=...
```

---

## Next Team Recommendations

Since all standard hypotheses are eliminated, the next team should investigate:

### Option A: Deep cuBLAS Audit

1. **Verify all cuBLAS parameters:**
   - Double-check `lda`, `ldb`, `ldc` values
   - Verify `CUBLAS_OP_T` interpretation with cuBLAS documentation
   - Test with `CUBLAS_OP_N` and manually transposed weight (TEAM THIMBLE did this, but re-verify)

2. **Test alternative GEMM implementations:**
   - Write a custom FP16 GEMM kernel for columns 95/126 only
   - Use `cublasSgemm` (FP32 weights/inputs) instead of `cublasGemmEx`
   - Try different cuBLAS algorithms (CUBLAS_GEMM_ALGO_*)

3. **Memory inspection:**
   - Dump raw memory at `layer.attn_q_weight + 95` and `+ 126` in hex
   - Check for NaN/inf bits (0x7C00/0xFC00 in FP16)
   - Verify memory alignment of weight buffer

### Option B: Workaround Path

1. **Skip problematic indices:**
   - Zero out Q[95] and Q[126] after GEMM
   - Measure impact on haiku generation quality
   - If output improves → confirms Q spikes are breaking downstream

2. **Test with different model:**
   - Run same code with a different Qwen2.5 checkpoint
   - If different checkpoint works → weight file corruption
   - If all checkpoints fail → code bug

### Option C: Attention Output Investigation (TEAM BATTLESHIP)

Since Q-projection is persistently broken, investigate if:
- Attention output projection has similar issues
- RoPE is amplifying the spikes
- The spikes are actually being used correctly but causing downstream chaos

---

## Test Command

```bash
cd bin/worker-orcd
REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda --release -- --ignored --nocapture --test-threads=1 | grep "TEAM TOP HAT"
```

---

## Verdict

**All three hypotheses (H1, H2, H3) are ELIMINATED.**

The Q-projection extremes at indices 95 and 126 remain unexplained. The bug is:
- ❌ NOT compute type (32F shows same issue)
- ❌ NOT weight corruption (weights are normal)
- ❌ NOT input spikes (normed is normal)
- ❌ NOT bias (biases are zeros)
- ❌ NOT stride/transpose (TEAM THIMBLE tested this)

**Recommendation:** Move to deep cuBLAS audit or implement workaround while investigating root cause in parallel.

---

**TEAM TOP HAT**  
**Status:** Investigation complete, all hypotheses eliminated, bug remains  
**Time:** 2025-10-07T00:34Z

*"When you eliminate the impossible, whatever remains, however improbable, must be the truth."* 👒
