# 🚢 TEAM BATTLESHIP — Downstream Wiring Investigation

**Date:** 2025-10-07T00:40Z  
**Mission:** Prove whether garbled logits come from downstream wiring (attention out projection, buffer aliasing, residual adds) rather than Q-projection itself  
**Status:** 🔧 INSTRUMENTATION READY — Ready for systematic testing

---

## Executive Summary

**Objective:** Top Hat & Thimble ruled out Q-projection issues (transpose/lda, compute type, weight/input corruption, bias). Extremes exist pre-bias, but we haven't excluded:
- Wrong destination buffers / aliasing (attn_out → ffn_out reuse)
- Overwrites between steps (e.g., RoPE or attention writing into scratch used later)
- Wrong strides on attention output projection (not Q)
- Residual mixups (residual_ / output) or misplaced memcpy

**Approach:** Work in append-only, foreground, one variable per change. Tag all logs `[TEAM BATTLESHIP]`.

---

## Investigation Plan

### 1. Buffer Integrity Tripwires (Step 1 - OPTIONAL)
**Guard:** `BATTLESHIP_CANARIES=1`

**What it does:**
- Writes canary values (0x7e00) to last 8 elements of each buffer
- Checks canaries after every major operation
- Logs any corruption with exact stage/buffer name

**Why disabled by default:** High overhead, only needed if suspect buffer overwrites.

**How to test:**
```bash
cd bin/worker-orcd
BATTLESHIP_CANARIES=1 REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda --release -- --ignored --nocapture --test-threads=1 | grep "BATTLESHIP"
```

**Expected output:**
```
[TEAM BATTLESHIP] CANARY OK q_proj_ stage=after_Q_GEMM
[TEAM BATTLESHIP] CANARY OK k_proj_ stage=after_K_GEMM
...
```

**Win condition:** Any canary flip pinpoints the overwriter.

---

### 2. Attention Output Projection Audit (Step 2 - START HERE)
**Guard:** `BATTLESHIP_ATTN_PROJ_AUDIT=1`

**What it does:**
- Logs attn_output_[0, 95, 126] before and after attention output GEMM
- Verifies output projection doesn't introduce/amplify spikes

**How to test:**
```bash
cd bin/worker-orcd
BATTLESHIP_ATTN_PROJ_AUDIT=1 REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda --release -- --ignored --nocapture --test-threads=1 | grep "BATTLESHIP"
```

**Expected output (per token):**
```
[TEAM BATTLESHIP] START layer=0 pos=0
[TEAM BATTLESHIP] Q_pre_bias q[0]=-0.0431 q[95]=-16.0469 q[126]=14.3359
[TEAM BATTLESHIP] ATTN_PROJ pre: attn_out[0]=... [95]=... [126]=...
[TEAM BATTLESHIP] ATTN_PROJ post: attn_out[0]=... [95]=... [126]=...
```

**Win condition:** If spikes are introduced/removed at ATTN_PROJ, that's the culprit.

---

### 3. Attention Output Projection Compute Type (Step 2a - OPTIONAL)
**Guard:** `BATTLESHIP_ATTN_PROJ_COMPUTE_32F=1` (use with `BATTLESHIP_ATTN_PROJ_AUDIT=1`)

**What it does:**
- Switches attention output projection from `CUBLAS_COMPUTE_32F_FAST_16F` to `CUBLAS_COMPUTE_32F`
- Tests if tensor-core fast-math introduces errors in this specific GEMM

**How to test:**
```bash
cd bin/worker-orcd
BATTLESHIP_ATTN_PROJ_AUDIT=1 BATTLESHIP_ATTN_PROJ_COMPUTE_32F=1 \
  REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda --release -- --ignored --nocapture --test-threads=1 | grep "BATTLESHIP"
```

**Win condition:** If output improves with COMPUTE_32F, tensor cores are suspect.

---

### 4. Buffer Pointer Trace (Step 3 - DIAGNOSTIC)
**Guard:** `BATTLESHIP_PTR_TRACE=1`

**What it does:**
- Logs device pointers for `attn_out_half`, `ffn_out_half`, `attn_output_`
- Verifies no aliasing between attention output and FFN scratch buffers

**How to test:**
```bash
cd bin/worker-orcd
BATTLESHIP_PTR_TRACE=1 REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda --release -- --ignored --nocapture --test-threads=1 | grep "BATTLESHIP"
```

**Expected output:**
```
[TEAM BATTLESHIP] PTR attn_out_half=0x7f8a40000000 ffn_out_half=0x7f8a40001000 attn_output_=0x7f8a40000000
```

**Win condition:** All three pointers should differ. If any match, buffers are aliased (BUG!).

---

### 5. No-Op Residual Toggle #1 (Step 4 - ISOLATE RESIDUAL PATH)
**Guard:** `BATTLESHIP_BYPASS_RESIDUAL1=1`

**What it does:**
- Skips `cuda_residual_add(input, attn_output_, residual_)`
- Instead: `residual_ = attn_output_` (no addition)
- Isolates whether residual connection corrupts values

**How to test:**
```bash
cd bin/worker-orcd
BATTLESHIP_BYPASS_RESIDUAL1=1 REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda --release -- --ignored --nocapture --test-threads=1
```

**Expected output:**
```
[TEAM BATTLESHIP] BYPASS_RESIDUAL1 active: copying attn_output_ to residual_
```

**Win condition:** If spikes disappear, residual add is corrupt (wrong src/dst/stride).

---

### 6. No-Op Residual Toggle #2 (Step 5 - ISOLATE FFN RESIDUAL)
**Guard:** `BATTLESHIP_BYPASS_RESIDUAL2=1`

**What it does:**
- Skips `cuda_residual_add(residual_, ffn_output_, output)`
- Instead: `output = ffn_output_` (no addition)
- Isolates whether second residual connection corrupts values

**How to test:**
```bash
cd bin/worker-orcd
BATTLESHIP_BYPASS_RESIDUAL2=1 REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda --release -- --ignored --nocapture --test-threads=1
```

**Expected output:**
```
[TEAM BATTLESHIP] BYPASS_RESIDUAL2 active: copying ffn_output_ to output
```

**Win condition:** If spikes disappear, second residual add is corrupt.

---

### 7. Minimal Workaround (Step 6 - CONTAINMENT)
**Guard:** `BATTLESHIP_MASK_Q_SPIKES=1`

**What it does:**
- After Q GEMM (pre-bias), clamps Q[95] and Q[126] to [-0.5, 0.5]
- Containment strategy while root cause is being investigated

**How to test:**
```bash
cd bin/worker-orcd
BATTLESHIP_MASK_Q_SPIKES=1 REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda --release -- --ignored --nocapture --test-threads=1
```

**Expected output:**
```
[TEAM BATTLESHIP] TEMP MASK applied to Q[95]=-16.0469->-0.5000 Q[126]=14.3359->0.5000
```

**Win condition:** If haiku becomes human-readable, spikes are breaking quality → ship workaround with TODO.

---

## Instrumentation Points

The code logs at these checkpoints for layer 0, tokens 0 & 1:

| Stage | Log | Indices Checked |
|-------|-----|-----------------|
| Q GEMM (pre-bias) | `Q_pre_bias` | 0, 95, 126 |
| Attention Output Proj (pre) | `ATTN_PROJ pre` | 0, 95, 126 |
| Attention Output Proj (post) | `ATTN_PROJ post` | 0, 95, 126 |
| Residual #1 | (only toggle) | — |
| Residual #2 | (only toggle) | — |

---

## Exit Criteria

### Success A: Root Cause Identified
- A canary or bypass test singles out one stage (e.g., attention output projection or a residual add) as the spike source
- Commit the minimal fix (buffer target/ldc/alias correction) with logs demonstrating before/after
- Document in `TEAM_BATTLESHIP_FINDINGS.md`

### Success B: Workaround Deployed
- With `BATTLESHIP_MASK_Q_SPIKES=1`, haiku becomes human-readable and quality gate improves
- Ship as temporary guard with TODO pointing to the failing stage you narrowed
- Document in `TEAM_BATTLESHIP_WORKAROUND.md`

---

## Recommended Testing Sequence

1. **Start:** Run with `BATTLESHIP_ATTN_PROJ_AUDIT=1` to see if attention output projection introduces spikes
2. **If spikes present:** Test `BATTLESHIP_ATTN_PROJ_COMPUTE_32F=1` to rule out tensor-core errors
3. **If no change:** Test `BATTLESHIP_PTR_TRACE=1` to verify no buffer aliasing
4. **If still stuck:** Test `BATTLESHIP_BYPASS_RESIDUAL1=1` and `BATTLESHIP_BYPASS_RESIDUAL2=1` separately
5. **If quality breaks with bypasses:** One of the residual adds is using wrong buffers
6. **If nothing works:** Deploy `BATTLESHIP_MASK_Q_SPIKES=1` as workaround while investigating

---

## Code Changed

### Files Modified
- `cuda/src/transformer/qwen_transformer.cpp`:
  - Lines 44-89: TEAM BATTLESHIP banner and macro definitions
  - Lines 496-502: Token counter and START log
  - Lines 659-694: Q projection logging and MASK_Q_SPIKES workaround
  - Lines 1162-1197: Attention output projection audit
  - Lines 1216-1224: Residual #1 bypass toggle
  - Lines 1295-1303: Residual #2 bypass toggle
  - Lines 1328-1331: Token counter increment

### Files Created
- `investigation-teams/TEAM_BATTLESHIP_HANDOFF.md`: This document

---

## Notes

- **Append-only:** Do not remove Thimble/Top Hat code. All BATTLESHIP code is guarded by macros (default disabled).
- **Foreground only:** Run tests with `--nocapture --test-threads=1` to see logs.
- **One toggle at a time:** Test each macro independently to isolate variables.
- **Commit between toggles:** If a toggle reveals something useful, commit before trying next.

---

## Previous Teams' Findings

### What Top Hat & Thimble Eliminated
- ✅ Stride/transpose issues (THIMBLE pre-transpose experiment)
- ✅ Tensor-core fast-math for Q projection (TOP HAT H1)
- ✅ Weight corruption at columns 95/126 (TOP HAT H2)
- ✅ Input spikes in normed (TOP HAT H3)
- ✅ Bias corruption (all biases are zeros)
- ✅ Manual FP32 calculation works correctly

### The Core Mystery
- Manual FP32: Q[95]≈±0.08, Q[126]≈±0.08 ✅
- cuBLAS (FAST_16F): Q[95]≈-16, Q[126]≈+14 ❌
- cuBLAS (32F): Q[95]≈-16, Q[126]≈+14 ❌

**Hypothesis:** The spikes may be introduced downstream (attention output projection, residual adds) or the Q spikes may be overwriting other buffers.

---

**TEAM BATTLESHIP**  
**Status:** Instrumentation deployed, ready for systematic testing  
**Time:** 2025-10-07T00:40Z

*"Good hunting, Battleship. 🚢"*
