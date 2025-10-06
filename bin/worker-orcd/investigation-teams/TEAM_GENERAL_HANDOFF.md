# Team General → Next Team Handoff

**Date**: 2025-10-06 18:26 UTC  
**Status**: ✅ **2 CRITICAL BUGS FIXED - 1 BUG REMAINS**

---

## Mission Accomplished

Fixed 2 critical bugs that were preventing the haiku test from running:

### ✅ Bug #1: Missing Theta Calculation (COMPILATION ERROR)
**Location**: `cuda/kernels/rope.cu` line 155-156  
**Problem**: Code used `theta` variable without calculating it  
**Fix**: Added missing calculation
```cpp
float inv_freq = 1.0f / powf(freq_base, (float)dim / (float)head_dim);
float theta = (float)pos * inv_freq;
```
**Impact**: Code wouldn't compile at all

### ✅ Bug #3: Infinite Loop in Softmax Reduction (HANG/CRASH)
**Location**: `cuda/kernels/gqa_attention.cu` line 365  
**Problem**: Team Supernova's "fix" created infinite loop
- Changed `s >>= 1` to `s = (s + 1) / 2`
- When `s = 1`: `s = (1+1)/2 = 1` (stays at 1 forever!)
**Fix**: Changed to `s /= 2` (integer division)
**Impact**: GPU ran at 100% for minutes then crashed

---

## Test Results

**Before my fixes**:
- ❌ Code wouldn't compile (missing theta)
- ❌ Test hung for minutes with GPU at 100%
- ❌ Worker crashed

**After my fixes**:
- ✅ Code compiles successfully
- ✅ Test completes in 6.7 seconds
- ✅ Generates 100 tokens
- ❌ Output is repetitive garbage (not a valid haiku)

---

## Bug Still Remaining

### ❌ Bug #4: Repetitive Token Generation

**Symptom**:
```
ĠseparatelyĠKwĠKwĠKwĠKwĠKwĠKwä¹Ŀå¤§awsawsawsaws...
```

**Pattern**:
- First token: "Ġseparately" (varies by prompt)
- Then gets stuck: "ĠKw" repeated 5-10 times
- Then switches to: "aws" repeated
- Then back to "ĠKw" repeated
- Eventually some other tokens but still repetitive

**Key Clue**: First token works, then breaks immediately!

---

## What's Been Verified Correct

✅ **Cache Infrastructure** (Team Water)
- cache_len parameter passing (0→1→2→3...)
- Cache write positions
- Cache read indexing
- Position tracking

✅ **RoPE** (Team Water)
- Theta values change with position
- Rotations are applied correctly

✅ **Softmax** (Team Alpha + Peer Review)
- Weights sum to 1.0 after normalization
- Reduction pattern now correct (no infinite loop)

✅ **Weight Loading** (Team Charlie Beta)
- ffn_down is loaded (was missing, now added)
- All weights present

✅ **Model File** (Team Charlie)
- llama.cpp generates perfect haiku with same model
- Weights are correct, bug is in OUR code

---

## Where the Bug Actually Is

Since first token works but second token fails, the bug is likely:

### Hypothesis 1: Attention Mechanism
- Q·K scores might be identical for all positions
- This would cause uniform attention weights
- Model wouldn't learn from context
- Would generate same token repeatedly

**How to test**:
- Uncomment debug printfs in `gqa_attention.cu`
- Check if attention scores vary or are all the same
- Compare with llama.cpp attention scores

### Hypothesis 2: KV Cache Corruption
- First token: cache is empty, works fine
- Second token: reads from cache, gets corrupted data
- Cache write might be writing to wrong location
- Or cache read might be reading from wrong location

**How to test**:
- Print K/V values being written to cache
- Print K/V values being read from cache
- Compare with llama.cpp cache values

### Hypothesis 3: Numerical Instability
- Values might explode/vanish after first token
- Check for NaN/Inf in hidden states
- Check for very large/small values

**How to test**:
- Add NaN/Inf checks after each operation
- Print min/max/mean of hidden states after each layer
- Look for values > 100 or < -100

---

## How to Continue Investigation

### Step 1: Enable Debug Output
Uncomment the debug printf statements in:
- `cuda/kernels/gqa_attention.cu` (lines 162-166, 206-218, 236-241)
- `cuda/kernels/rope.cu` (lines 177-179)

### Step 2: Run Test and Analyze
```bash
cd bin/worker-orcd
REQUIRE_REAL_LLAMA=1 cargo test --release --features cuda \
  --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  -- --ignored --nocapture --test-threads=1 2>&1 | tee debug.log
```

### Step 3: Look for Patterns
- Do attention weights vary or are they uniform?
- Do Q·K scores change between tokens?
- Are cache values reasonable or garbage?
- Do values explode after first token?

### Step 4: Compare with llama.cpp
Run same prompt in llama.cpp with verbose output:
```bash
./llama-cli -m qwen2.5-0.5b-instruct-fp16.gguf \
  -p "Write a haiku" --verbose
```
Compare intermediate values with our implementation.

---

## Files Modified

### Code Changes
1. `cuda/kernels/rope.cu` line 155-156: Added theta calculation
2. `cuda/kernels/gqa_attention.cu` line 365: Fixed infinite loop

### Documentation Added
1. `cuda/kernels/rope.cu` lines 144-154: Documented theta bug
2. `cuda/kernels/gqa_attention.cu` lines 7-36: Status for next team
3. `cuda/kernels/gqa_attention.cu` lines 352-364: Infinite loop bug
4. `investigation-teams/TEAM_GENERAL_FINDINGS.md`: Full investigation
5. `investigation-teams/TEAM_GENERAL_HANDOFF.md`: This document

---

## Quick Reference

**Test Command**:
```bash
cd bin/worker-orcd && REQUIRE_REAL_LLAMA=1 cargo test --release \
  --features cuda --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only -- --ignored --nocapture
```

**Key Files**:
- Attention: `cuda/kernels/gqa_attention.cu`
- RoPE: `cuda/kernels/rope.cu`
- Transformer: `cuda/src/transformer/qwen_transformer.cpp`
- Weight Loading: `cuda/src/model/qwen_weight_loader.cpp`

**Previous Investigations**:
- Team Water: `investigation-teams/TEAM_WATER_FINDINGS.md`
- Team Charlie Beta: `investigation-teams/TEAM_CHARLIE_BETA_ROOT_CAUSE.md`
- Team Alpha: `investigation-teams/TEAM_ALPHA_RESULTS.md`

---

## Success Criteria

The haiku test passes when:
1. ✅ Test completes without hanging (DONE!)
2. ✅ Generates 100 tokens (DONE!)
3. ❌ Output is a valid haiku (NOT YET)
4. ❌ Contains the minute word exactly once (NOT YET)
5. ❌ Is creative and coherent (NOT YET)

**Current Score**: 2/5 ✅✅❌❌❌

---

## Additional Finding (2025-10-06 18:31 UTC)

Re-enabled debug output for first 3 tokens. Key observations:

**Attention weights are CORRECT**:
- cache_len=0: weight=[1.0] (only current token)
- cache_len=1: weights=[0.51, 0.49] or [0.34, 0.66] (varies properly)
- Softmax sums to 1.0 correctly
- Weights are NOT uniform (attention is working)

**Token generation pattern**:
- Token 0: "Ġseparately" (ID=25156) ✅ Good
- Token 1: "(epoch" (ID=61290) ✅ Good
- Token 2: "ĠKw" (ID=64362) ❌ Starts repeating
- Token 3+: "ĠKw" repeated or switches to other repetitive tokens

**Hypothesis**: The bug is NOT in attention mechanism. Attention weights vary correctly.
The bug is likely in:
1. FFN computation producing biased logits
2. Or final layer norm/projection amplifying certain logits
3. Or numerical instability causing certain tokens to dominate

---

**Team General**  
**Signing off**: 2025-10-06 18:26 UTC  
**Status**: 2 bugs fixed, test runs but output quality poor  
**Next Team**: Focus on why first token works but second token fails! 🔦
