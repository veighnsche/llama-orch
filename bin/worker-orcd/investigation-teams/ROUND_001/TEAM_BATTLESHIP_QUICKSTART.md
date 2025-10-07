# 🚢 TEAM BATTLESHIP — Quick Start Guide

**Mission:** Find where Q[95]/Q[126] spikes propagate or get amplified in the forward pass.

---

## TL;DR — Run This First

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd

# Test 1: Check if attention output projection introduces spikes
BATTLESHIP_ATTN_PROJ_AUDIT=1 REQUIRE_REAL_LLAMA=1 \
  cargo test --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda --release -- --ignored --nocapture --test-threads=1 \
  2>&1 | grep -A 2 "TEAM BATTLESHIP"
```

---

## What to Look For

### Token 0 Output
```
[TEAM BATTLESHIP] START layer=0 pos=0
[TEAM BATTLESHIP] Q_pre_bias q[0]=-0.0431 q[95]=-16.0469 q[126]=14.3359
[TEAM BATTLESHIP] ATTN_PROJ pre: attn_out[0]=X.XXXX [95]=Y.YYYY [126]=Z.ZZZZ
[TEAM BATTLESHIP] ATTN_PROJ post: attn_out[0]=A.AAAA [95]=B.BBBB [126]=C.CCCC
```

### Key Questions

**Q1:** Are `[95]` and `[126]` normal (±1) at `ATTN_PROJ pre`?
- **YES** → Q spikes don't propagate to attention output (isolated to Q buffer)
- **NO (still ±16)** → Q spikes leaked through attention mechanism

**Q2:** Do `[95]` and `[126]` change dramatically at `ATTN_PROJ post`?
- **YES (new spikes introduced)** → Attention output projection is the culprit!
- **NO (values stay normal)** → Projection is fine, look elsewhere

---

## Decision Tree

```
START HERE
    │
    ├─> Run BATTLESHIP_ATTN_PROJ_AUDIT=1
    │
    ├─> Are ATTN_PROJ pre values normal (±1)?
    │   ├─> YES: Q spikes isolated, attention filters them out
    │   │   └─> Check ATTN_PROJ post for new spikes
    │   │       ├─> Spikes at post? → FOUND BUG (attn output projection)
    │   │       └─> No spikes? → Bug is in Q GEMM itself, revisit Top Hat findings
    │   │
    │   └─> NO: Q spikes propagate through attention
    │       └─> Test BATTLESHIP_BYPASS_RESIDUAL1=1
    │           ├─> Spikes disappear? → FOUND BUG (first residual add)
    │           └─> Spikes persist? → Bug is deeper, test residual #2
```

---

## Next Steps Based on Results

### Scenario A: Spikes Appear at ATTN_PROJ post
**Action:** Investigate attention output projection GEMM parameters
- Verify `ldc` (should be `hidden_dim`)
- Check if writing to correct buffer (not aliasing ffn_output_)
- Test with `BATTLESHIP_ATTN_PROJ_COMPUTE_32F=1`

**Fix location:** `qwen_transformer.cpp` line ~1184

---

### Scenario B: Spikes Filtered by Attention
**Action:** Q spikes are isolated, verify they don't break downstream
- Test with `BATTLESHIP_MASK_Q_SPIKES=1` to clamp them
- Check if quality improves

**Hypothesis:** Q spikes exist but are harmless (attention softmax washes them out)

---

### Scenario C: Spikes Propagate Through Attention
**Action:** Attention mechanism amplifies Q spikes
- Check if GQA attention uses wrong stride
- Test residual bypasses to isolate corruption point

**Next test:** `BATTLESHIP_BYPASS_RESIDUAL1=1`

---

## How to Read the Logs

### Normal Values
- `q[0]`: Should be ±0.1 (reference point)
- `attn_out[0]`: Should be ±0.5 after attention
- All values should be in range [-5, 5] for FP16 activations

### Red Flags
- Any value >10 or <-10 (FP16 overflow territory)
- Indices [95] or [126] having drastically different magnitudes than [0]
- Values jumping orders of magnitude between stages

---

## Compilation Guide

The code uses `#ifndef` guards, so macros default to 0 (disabled).

To enable a macro:
1. **Option A (compile-time):** Edit `qwen_transformer.cpp` line 70:
   ```cpp
   #define BATTLESHIP_ATTN_PROJ_AUDIT 1
   ```
   
2. **Option B (environment — NOT SUPPORTED YET):** 
   Would need to pass via `RUSTFLAGS` → currently not wired up

**Recommended:** Use Option A for now, recompile after each macro change.

---

## Performance Impact

| Macro | Overhead | Safe for Production? |
|-------|----------|----------------------|
| `BATTLESHIP_ATTN_PROJ_AUDIT` | Low (2 cudaMemcpy per token) | ✅ Yes (debug only) |
| `BATTLESHIP_PTR_TRACE` | Negligible (1 fprintf per token) | ✅ Yes |
| `BATTLESHIP_MASK_Q_SPIKES` | Low (2 cudaMemcpy + 2 clamps per token) | ⚠️ Workaround only |
| `BATTLESHIP_BYPASS_RESIDUAL1/2` | None (replaces residual_add) | ❌ No (breaks model) |
| `BATTLESHIP_CANARIES` | HIGH (16 cudaMemcpy per layer!) | ❌ No (debug only) |

---

## Expected Timeline

- **Test 1 (ATTN_PROJ_AUDIT):** 5 minutes (compile + run + analyze)
- **Test 2a (PTR_TRACE):** 3 minutes
- **Test 2b (ATTN_PROJ_COMPUTE_32F):** 5 minutes
- **Test 3 (BYPASS_RESIDUAL1):** 5 minutes
- **Test 4 (BYPASS_RESIDUAL2):** 5 minutes
- **Test 5 (MASK_Q_SPIKES):** 10 minutes (check quality)

**Total:** ~30-40 minutes for full systematic investigation.

---

## Success Metrics

### Investigation Complete When:
1. ✅ You identify the exact stage where spikes are introduced/amplified
2. ✅ You verify the stage by toggling a bypass or compute mode
3. ✅ You document findings in `TEAM_BATTLESHIP_FINDINGS.md`
4. ✅ You either fix the bug or deploy `MASK_Q_SPIKES` workaround

### Quality Gate
Run the haiku test without grep:
```bash
REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda --release -- --ignored --nocapture --test-threads=1
```

Look for test output to contain:
- ✅ **WIN:** Haiku with "forty-six" or similar coherent English
- ❌ **FAIL:** Mojibake, repetitive tokens, or wrong language

---

**TEAM BATTLESHIP**  
**Quick Start Created:** 2025-10-07T00:40Z

*"Start with ATTN_PROJ_AUDIT. That's your best lead. 🎯"*
