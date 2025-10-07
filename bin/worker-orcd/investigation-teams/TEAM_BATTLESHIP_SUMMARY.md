# 🚢 TEAM BATTLESHIP — Implementation Summary

**Date:** 2025-10-07T00:40Z  
**Status:** ✅ INSTRUMENTATION COMPLETE — Ready for testing

---

## What Was Implemented

### 7 Macro-Guarded Investigation Tools

All disabled by default (`#define MACRO 0`), append-only design:

1. **`BATTLESHIP_CANARIES`** — Buffer integrity tripwires
   - Writes 0x7e00 to buffer tails, checks after each op
   - High overhead, use only if suspect overwrites

2. **`BATTLESHIP_ATTN_PROJ_AUDIT`** — Attention output projection logging
   - Logs indices [0, 95, 126] before/after attention output GEMM
   - **START HERE** — Most likely culprit

3. **`BATTLESHIP_ATTN_PROJ_COMPUTE_32F`** — Compute type toggle for attn proj
   - Switches from FAST_16F to 32F for attention output projection
   - Tests if tensor-core fast-math introduces errors

4. **`BATTLESHIP_PTR_TRACE`** — Buffer pointer logging
   - Dumps device pointers to verify no aliasing
   - Low overhead, safe for all runs

5. **`BATTLESHIP_BYPASS_RESIDUAL1`** — Skip first residual add
   - Tests if residual #1 corrupts values
   - Breaks model, debug only

6. **`BATTLESHIP_BYPASS_RESIDUAL2`** — Skip second residual add
   - Tests if residual #2 corrupts values
   - Breaks model, debug only

7. **`BATTLESHIP_MASK_Q_SPIKES`** — Workaround clamp
   - Clamps Q[95]/Q[126] to [-0.5, 0.5] after Q GEMM
   - Containment strategy if root cause elusive

---

## Code Changes

### Modified Files
- **`cuda/src/transformer/qwen_transformer.cpp`**
  - Lines 44-89: Banner + macro definitions
  - Lines 496-502: Token counter initialization
  - Lines 659-694: Q projection logging + MASK toggle
  - Lines 1162-1197: Attention projection audit
  - Lines 1216-1224: Residual #1 bypass
  - Lines 1295-1303: Residual #2 bypass
  - Lines 1328-1331: Token counter increment

### Created Files
- **`investigation-teams/TEAM_BATTLESHIP_HANDOFF.md`** — Full investigation guide
- **`investigation-teams/TEAM_BATTLESHIP_QUICKSTART.md`** — Quick start guide
- **`investigation-teams/TEAM_BATTLESHIP_SUMMARY.md`** — This file

---

## How to Use

### Step 1: Enable a Macro
Edit `qwen_transformer.cpp` line 73 (or whichever macro):
```cpp
#ifndef BATTLESHIP_ATTN_PROJ_AUDIT
#define BATTLESHIP_ATTN_PROJ_AUDIT 1  // <-- Change 0 to 1
#endif
```

### Step 2: Recompile and Run
```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd

REQUIRE_REAL_LLAMA=1 cargo test --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  --features cuda --release -- --ignored --nocapture --test-threads=1 \
  2>&1 | grep "TEAM BATTLESHIP"
```

### Step 3: Analyze Output
Look for patterns in the logs (see QUICKSTART.md for decision tree).

---

## Investigation Sequence (Recommended)

```
1. BATTLESHIP_ATTN_PROJ_AUDIT=1        (5 min)  ← START HERE
   ↓
2. BATTLESHIP_PTR_TRACE=1              (3 min)  ← Verify no aliasing
   ↓
3. BATTLESHIP_ATTN_PROJ_COMPUTE_32F=1  (5 min)  ← If attn proj suspect
   ↓
4. BATTLESHIP_BYPASS_RESIDUAL1=1       (5 min)  ← If residual suspect
   ↓
5. BATTLESHIP_BYPASS_RESIDUAL2=1       (5 min)  ← Test second residual
   ↓
6. BATTLESHIP_MASK_Q_SPIKES=1          (10 min) ← Workaround if needed
```

**Total time:** ~30-40 minutes for full investigation.

---

## What to Expect

### Successful Root Cause Identification
You'll see logs like:
```
[TEAM BATTLESHIP] ATTN_PROJ pre: attn_out[95]=0.23 [126]=-0.18  ← Normal!
[TEAM BATTLESHIP] ATTN_PROJ post: attn_out[95]=14.56 [126]=-12.34  ← SPIKE!
```
**Conclusion:** Attention output projection introduces spikes → Fix GEMM params.

### Successful Workaround
With `BATTLESHIP_MASK_Q_SPIKES=1`, test output shows:
```
[TEAM BATTLESHIP] TEMP MASK applied to Q[95]=-16.05->-0.50 Q[126]=14.34->0.50
```
And haiku test passes with coherent English → Ship workaround.

### No Smoking Gun
All tests show normal values, spikes persist → Bug is in Q GEMM itself (revisit cuBLAS params).

---

## Safety Notes

1. **All macros default to 0 (disabled)** — No risk of accidentally enabling debug code
2. **Append-only design** — Previous teams' code (Thimble, Top Hat) untouched
3. **Foreground logging only** — No performance impact in production (all fprintf guarded)
4. **Token-limited** — Logs only layer 0, tokens 0-1 (avoids log spam)

---

## Next Team Handoff

If you don't find the bug:
1. Document what you tested in `TEAM_BATTLESHIP_FINDINGS.md`
2. Note which macro combinations were tried
3. Include log snippets showing key values
4. Recommend next investigation direction (e.g., custom GEMM kernel, memory alignment audit)

---

## References

- **Investigation Guide:** `TEAM_BATTLESHIP_HANDOFF.md`
- **Quick Start:** `TEAM_BATTLESHIP_QUICKSTART.md`
- **Previous Teams:**
  - `TEAM_TOP_HAT_HANDOFF.md` — Eliminated H1/H2/H3
  - `TEAM_THIMBLE_SUMMARY.md` — Disproved stride hypothesis
  - `TEAM_HELIOS_HANDOFF.md` — Sampling architecture verified

---

**TEAM BATTLESHIP**  
**Implementation:** Complete ✅  
**Documentation:** Complete ✅  
**Status:** Ready for testing 🚢

*"The instrumentation is in place. Now go find that bug!"*
