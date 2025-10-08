# 🔥 Uninvestigated Smoking Guns - Priority List

**Date:** 2025-10-08  
**Compiled By:** TEAM DICKINSON  
**Status:** 🚨 **CRITICAL LEADS REMAINING**

---

## Executive Summary

After reviewing all investigation team documents and code comments, there are **3 CRITICAL uninvestigated leads** that could explain the garbage output:

1. **Embedding Table Dimensions** - Tested but inconclusive (75% confidence)
2. **Large Value Spikes in Mid-Layers** - Observed but not investigated (NEW from DICKINSON data)
3. **Special Token Handling** - Mentioned but never tested

---

## 🔥 SMOKING GUN #1: Embedding Layer (75% Confidence)

**Location:** `cuda/kernels/embedding.cu` line 177

**Evidence:**
```cpp
// [TEAM SHAKESPEARE 2025-10-07T23:07Z] TEST RESULTS:
//   Original code: weight_matrix[token_id * hidden_dim + dim_idx]
//     → Generated tokens: [20695, 131033, 42294, 43321, ...] (garbage)
//   
//   Transposed access: weight_matrix[dim_idx * vocab_size + token_id]
//     → Generated tokens: [37557, 103357, 69289, 62341, ...] (DIFFERENT garbage!)
//   
//   CONCLUSION: Changing indexing DOES change output (proves embedding matters)
//               BUT output still garbage (transpose alone not the fix)
```

**Why This Is Critical:**
- ✅ Changing embedding indexing CHANGES output (proves it matters)
- ❌ But output still garbage (not a simple transpose)
- 🎯 llama.cpp works perfectly with same model file
- 🔥 **This is THE smoking gun** - embedding affects output but fix incomplete

**What TEAM SHAKESPEARE Recommended (NOT DONE YET):**
```
NEXT TEAM (TEAM FROST) SHOULD:
  1. Dump actual embedding values from GGUF for token_id=0
  2. Dump what this code reads for token_id=0
  3. Dump what llama.cpp reads for token_id=0
  4. Compare byte-for-byte to find exact mismatch
  5. Check if there are OTHER transpose bugs (lm_head, Q/K/V, FFN)
  6. Verify GGUF dimensions with gguf-dump tool
```

**DICKINSON Data Shows:**
```
C0 (post-embedding): [0.012, 0.007, -0.020, -0.007, ...]
```
These values seem reasonable (±0.05 range), but we need to compare with llama.cpp!

**Action Required:**
1. **Instrument llama.cpp** to dump C0 (post-embedding) values
2. **Compare with our C0** from DICKINSON data
3. **If they differ** → embedding bug confirmed, investigate indexing/layout
4. **If they match** → bug is downstream (attention/FFN)

**Confidence:** 🔥🔥🔥 75% (proven to affect output, but fix incomplete)

---

## 🔥 SMOKING GUN #2: Extreme Value Spikes in Mid-Layers (NEW!)

**Location:** Discovered by TEAM DICKINSON in checkpoint data

**Evidence:**
```
C5 (layer 5):  [-0.252, -2.299, -1.993, -2.633, 2.445, 15.094, ...]
                                                              ^^^^^^ SPIKE!

C10 (layer 10): [-0.110, -2.904, -2.221, -3.330, 2.807, 17.281, ...]
                                                               ^^^^^^ GROWING!
```

**Observation:**
- Index 5 has value **15.094** at layer 5
- Same index has value **17.281** at layer 10 (GROWING!)
- Other values are in normal range (±3)

**Why This Is Critical:**
- 🔥 **Extreme values accumulate** through layers (15 → 17)
- ⚠️ Could indicate numerical instability
- ⚠️ Could indicate weight loading issue for specific dimensions
- ⚠️ Could indicate FFN gate/up/down bug affecting certain channels

**What Needs Investigation:**
1. **Compare with llama.cpp** - Does llama.cpp also have spikes at index 5?
2. **Track through layers** - Does index 5 keep growing? (C23 shows -3.371, normalized)
3. **Check FFN weights** - Are FFN weights for dimension 5 abnormal?
4. **Check RMSNorm** - Is RMSNorm handling large values correctly?

**Hypothesis:**
- If llama.cpp ALSO has spikes → Normal model behavior
- If llama.cpp DOESN'T have spikes → Our bug (FFN? RMSNorm? Weight loading?)

**Action Required:**
1. **Instrument llama.cpp** to dump C5, C10 checkpoints
2. **Compare index 5 values** specifically
3. **If different** → Investigate FFN/RMSNorm for dimension 5

**Confidence:** 🔥🔥 60% (NEW finding, needs llama.cpp comparison)

---

## 🔥 SMOKING GUN #3: Special Token Handling (Mentioned, Never Tested)

**Location:** Multiple mentions in code comments

**Evidence:**
```cpp
// [TEAM PURPLE] 2025-10-06T21:16Z - VERIFIED: Token IDs are correct ✅
// OBSERVED: Token IDs are correct!
//   [0] = 151644 (im_start special token)
//   [1] = 872 (user)
//   [2] = 198 (\n)
```

**From ROUND_002_COORDINATOR_BRIEFING.md:**
```
**Root Cause:**
Bug is NOT in cuBLAS, softmax, or sampling. Bug is in uninvestigated subsystem, most likely:
1. Embedding layer (token ID → vector conversion)
2. Special token handling (chat template disabled)  ← THIS!
3. Attention mask or RoPE
```

**Why This Is Critical:**
- 🔥 Token 151644 is `im_start` special token
- ⚠️ Special tokens might have different embedding handling
- ⚠️ Chat template is disabled (might affect special token processing)
- ⚠️ llama.cpp might handle special tokens differently

**What Needs Investigation:**
1. **Check if special tokens have separate embeddings** in GGUF
2. **Compare special token embedding** with regular token embedding
3. **Check if llama.cpp** has special handling for token 151644
4. **Test with non-special tokens** only (skip chat template entirely)

**Action Required:**
1. **Dump embedding for token 151644** from GGUF
2. **Compare with regular token** (e.g., token 872 "user")
3. **Check llama.cpp code** for special token handling
4. **Test with simple prompt** (no special tokens)

**Confidence:** 🔥 40% (mentioned but never investigated)

---

## 📊 Priority Ranking

### Priority 1: Embedding Layer Parity (CRITICAL)

**Why:** Proven to affect output, but fix incomplete

**Action:**
1. Use DICKINSON's C0 checkpoint data
2. Instrument llama.cpp to get C0
3. Compare byte-for-byte
4. If different → investigate indexing/layout/scaling

**Estimated Time:** 1-2 hours

**Expected Outcome:** 
- If C0 matches → Bug is downstream
- If C0 differs → **BUG FOUND!** Fix embedding indexing

---

### Priority 2: Mid-Layer Value Spikes (HIGH)

**Why:** NEW finding from DICKINSON data, could explain accumulating errors

**Action:**
1. Use DICKINSON's C5/C10 checkpoint data
2. Instrument llama.cpp to get C5/C10
3. Compare index 5 specifically
4. If different → investigate FFN/RMSNorm

**Estimated Time:** 1-2 hours

**Expected Outcome:**
- If spikes match → Normal behavior
- If spikes differ → **BUG FOUND!** Fix FFN or RMSNorm

---

### Priority 3: Special Token Handling (MEDIUM)

**Why:** Mentioned but never tested, could be simple fix

**Action:**
1. Dump embedding for token 151644
2. Compare with regular token
3. Check llama.cpp special token handling
4. Test with simple prompt (no special tokens)

**Estimated Time:** 30 min - 1 hour

**Expected Outcome:**
- If special tokens handled correctly → Not the bug
- If special tokens wrong → **BUG FOUND!** Fix special token embedding

---

## 🎯 Recommended Investigation Order

### Step 1: Complete DICKINSON Mission (IMMEDIATE)

**Goal:** Get llama.cpp checkpoint data to compare with our data

**Tasks:**
1. Instrument llama.cpp with C0, C1, C5, C10, C23, C24, C25 checkpoints
2. Run with same prompt: "GPU haiku with word fifty-one: "
3. Extract JSONL logs
4. Compare with our data from `/tmp/dickinson_checkpoints.jsonl`

**Deliverable:** Side-by-side comparison showing first divergence point

**Time:** 2-3 hours

---

### Step 2: Investigate First Divergence (HIGH PRIORITY)

**If C0 diverges:**
→ Investigate Smoking Gun #1 (Embedding Layer)

**If C5/C10 diverge but C0/C1 match:**
→ Investigate Smoking Gun #2 (Mid-Layer Spikes)

**If all match:**
→ Bug is in sampling or tokenization (not forward pass)

---

### Step 3: Test Special Token Handling (IF NEEDED)

**Only if:** Embedding layer seems correct but output still garbage

**Action:** Test with simple prompt without special tokens

---

## 📝 Other Leads (Lower Priority)

### 4. Embedding Scaling (Mentioned, Not Tested)

**Location:** `qwen_transformer.cpp` line 422

```cpp
// [TEAM GREEN] 2025-10-06T20:38Z
// SUSPECT: Embedding scaling might be missing!
// QUESTION: Does llama.cpp multiply by sqrt(hidden_dim) or similar?
```

**Why Lower Priority:** DICKINSON C0 values seem reasonable (±0.05 range)

**Action:** Check if llama.cpp scales embeddings after lookup

---

### 5. FFN Down Projection (Mentioned, Not Tested)

**Location:** `qwen_transformer.cpp` line 192

```cpp
// ROOT CAUSE (HYPOTHESIS): Missing weight loading in qwen_weight_loader.cpp
// The load_from_gpu_pointers() function loaded ffn_gate and ffn_up but
// FORGOT to load ffn_down!
```

**Why Lower Priority:** DICKINSON C23 (after final layer) shows reasonable values

**Action:** Verify ffn_down weights are loaded correctly

---

## 🔍 How to Use This Document

### For Next Investigator:

1. **Read Priority 1** (Embedding Layer Parity)
2. **Complete DICKINSON mission** (instrument llama.cpp)
3. **Compare checkpoint data** to find first divergence
4. **Investigate that smoking gun** based on where divergence occurs

### For Coordinator:

This document provides a clear investigation path:
- **Step 1:** Complete parity check (DICKINSON mission)
- **Step 2:** Investigate first divergence (one of the 3 smoking guns)
- **Step 3:** Test special tokens if needed

---

## 📚 References

**Investigation Documents:**
- `DICKINSON_FINAL_REPORT.md` - Checkpoint data (6/7 captured)
- `ROUND_002_COORDINATOR_BRIEFING.md` - Round 2 summary
- `REFERENCE_IMPLEMENTATION_ANALYSIS.md` - Embedding transpose analysis
- `TRANSPOSE_FIX_TEST_RESULTS.md` - Embedding test results

**Code Locations:**
- `cuda/kernels/embedding.cu` line 177 - Embedding lookup
- `cuda/src/transformer/qwen_transformer.cpp` - Forward pass with comments

**Data:**
- `/tmp/dickinson_checkpoints.jsonl` - Our checkpoint values (6/7)

---

**TEAM DICKINSON**  
*"Tell all the truth but tell it slant—Success in Circuit lies."*

**Document Status:** ✅ **COMPLETE**  
**Last Updated:** 2025-10-08T00:09Z  
**Next Action:** Instrument llama.cpp and compare checkpoints
