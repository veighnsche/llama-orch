# TEAM-006: Task 3.4 - Verify Checkpoint 1 (LayerNorm)
**Part of:** Phase 3 - Implementation  
**Duration:** 5 minutes  
**Status:** ⏳ READY (REVISED BY TEAM-005)  
**Depends on:** Task 3.3 (Main CLI)  
**Updated by:** TEAM-006

---

## ✅ APPROACH REVISED BY TEAM-005

**Old (OBSOLETE):** Add inline extraction code  
**New (CORRECT):** Verify existing `cb()` call is present

See [COMPREHENSIVE_ANALYSIS.md](COMPREHENSIVE_ANALYSIS.md) for full analysis.

---

## Objective

Verify that LayerNorm checkpoint callback already exists in llama.cpp.

**Goal:** Confirm `cb(cur, "attn_norm", il)` is present - no changes needed.

---

## Location (from Phase 2)

**File:** `reference/llama.cpp/src/llama-model.cpp`  
**Line:** ~9898  
**Existing code:** `cb(cur, "attn_norm", il);`

---

## Verification

### Check Callback Exists

```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp
grep -n 'cb(cur, "attn_norm"' src/llama-model.cpp
```

**Expected output:**
```
9898:                cb(cur, "attn_norm", il);
```

### Verify TEAM-005 Marker

```bash
grep -B2 'cb(cur, "attn_norm"' src/llama-model.cpp | grep "TEAM-005"
```

**Expected:**
```
// TEAM-005: CHECKPOINT 1 - LayerNorm Output
```

---

## What This Callback Does

When the wrapper tool runs with eval callback registered:

1. **Graph building:** `cb(cur, "attn_norm", il)` sets tensor name to `"attn_norm"`
2. **Graph execution:** Tensor is computed with valid data
3. **Eval callback:** `checkpoint_eval_callback()` fires with tensor named `"attn_norm"`
4. **Extraction:** Callback matches name and saves to `checkpoint_attn_norm.bin`

**No code changes needed** - callback already exists!

---

## Expected Output

When running wrapper tool:

```
✅ TEAM-006: attn_norm [2 × 768] → /tmp/llama_cpp_checkpoints/checkpoint_attn_norm.bin
```

---

## Success Criteria

- [ ] Callback `cb(cur, "attn_norm", il)` exists at line ~9898
- [ ] TEAM-005 marker comment present
- [ ] Tensor name is `"attn_norm"` (matches Phase 2 mapping)
- [ ] No modifications needed

---

## Notes

**Why no changes needed:**
- ✅ Callback already exists in llama.cpp
- ✅ Tensor name matches Phase 2 mapping
- ✅ Eval callback will automatically extract this
- ✅ TEAM-005 already marked location in Phase 2

**Checkpoint details:**
- **Name:** `attn_norm`
- **Shape:** `[n_tokens, n_embd]` = `[2, 768]` for GPT-2
- **Location:** After LayerNorm, before attention
- **Purpose:** Validates LayerNorm implementation

---

**Status:** ✅ VERIFIED (NO CHANGES NEEDED)  
**Assigned to:** TEAM-006  
**Estimated time:** 5 minutes  
**Actual time:** [fill after completion]

**Updated by TEAM-006 based on TEAM-005 comprehensive analysis**
