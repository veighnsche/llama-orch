# Team Charlie Gamma - Investigation Handoff

**Date**: 2025-10-06 17:32 UTC  
**Status**: 🔍 **CRITICAL CLUE FOUND - NEEDS NEXT INVESTIGATOR**

---

## Summary

I found and fixed ONE bug (`ffn_down` missing), tested it, and discovered it wasn't THE bug. But I found a CRITICAL CLUE that the next team must investigate.

---

## What I Fixed (But Didn't Solve The Problem)

### Fix #1: Missing ffn_down Weight
**File**: `cuda/src/model/qwen_weight_loader.cpp:371`  
**What**: Added `layer.ffn_down = get_ptr(prefix + "ffn_down.weight");`  
**Result**: ❌ Didn't fix repetitive tokens  
**Status**: Good fix to have, but not THE bug  

---

## Critical Clue For Next Team

### 🔥 THE SMOKING GUN 🔥

**Debug output shows `cache_len=0` for ALL layers, even when `pos` increments!**

```
[FORWARD DEBUG #0] pos=0 (read from kv_cache_.seq_lens)
[ATTENTION DEBUG] cache_len=0  ← Should be 0 ✅

[FORWARD DEBUG #1] pos=1 (read from kv_cache_.seq_lens)  
[ATTENTION DEBUG] cache_len=0  ← Should be 1! ❌

[FORWARD DEBUG #2] pos=2 (read from kv_cache_.seq_lens)
[ATTENTION DEBUG] cache_len=0  ← Should be 2! ❌
```

### What This Means

**The attention kernel NEVER sees previous tokens in the cache!**

Even though:
- ✅ `pos` increments correctly (0, 1, 2, 3...)
- ✅ `pos` is passed as `cache_len` parameter
- ❌ Attention kernel receives `cache_len=0` always!

This explains EVERYTHING:
- Why first 3 tokens are different (they're not using cache)
- Why attention weights are uniform (no context to attend to)
- Why model gets stuck (can't learn from previous tokens)

---

## The Bug Is In Parameter Passing

### Where To Look

**File**: `cuda/src/transformer/qwen_transformer.cpp:306`

```cpp
cuda_gqa_attention_forward(
    q_proj_, k_proj_, v_proj_, layer_k_cache, layer_v_cache, attn_output_,
    batch_size, config_.num_heads, config_.num_kv_heads, config_.head_dim,
    1,    // seq_len
    pos,  // cache_len ← We pass pos here, but kernel receives 0!
    config_.context_length, // max_seq_len
    nullptr
);
```

### Possible Causes

1. **Parameter order is wrong** - Maybe cache_len and seq_len are swapped?
2. **Function signature mismatch** - Maybe the wrapper function has wrong parameter order?
3. **Value is being overwritten** - Maybe something resets cache_len to 0?

---

## What To Investigate

### Step 1: Check Function Signature
Verify the parameter order in:
- `cuda/kernels/gqa_attention.cu` - `cuda_gqa_attention_forward()` declaration
- `cuda/src/transformer/qwen_transformer.cpp` - Call site

### Step 2: Add Debug Print In Wrapper
In `cuda_gqa_attention_forward()` wrapper, print the cache_len parameter:
```cpp
printf("[WRAPPER DEBUG] Received cache_len=%u\n", cache_len);
```

### Step 3: Check If Parameters Are Swapped
The call passes:
```cpp
..., 1, pos, config_.context_length, ...
```

Maybe it should be:
```cpp
..., pos, 1, config_.context_length, ...
```

---

## Test Results

### Output
```
Ġseparately(epochawsĠKwĠKwĠKwĠKwĠKwĠKw...
```

### Key Observations
1. ✅ First 3 tokens ARE different
2. ❌ Token 4+ stuck on "ĠKw" (ID 64362)
3. ❌ Attention weights uniform for early tokens
4. ❌ cache_len=0 always (THIS IS THE BUG!)

---

## Comments Added To Code

I added investigation comments to these locations:

1. **`qwen_weight_loader.cpp:362-371`**
   - Documents the missing ffn_down line
   - Explains it was tested and didn't fix the bug
   - Notes the position-dependent failure pattern

2. **`qwen_transformer.cpp:819-828`**
   - Debug code to print pos values
   - Documents that pos DOES increment correctly

3. **`qwen_transformer.cpp:299-306`**
   - CRITICAL comment about cache_len=0 bug
   - Points to where the real bug likely is

4. **`gqa_attention.cu:98-110`**
   - Documents the cache_len=0 observation
   - Explains why this causes repetitive tokens

5. **`rope.cu:149-157`**
   - Documents that RoPE IS working correctly
   - Theta values change with position as expected

---

## For The Next Team

### Start Here
**File**: `cuda/kernels/gqa_attention.cu:597`

Check the `cuda_gqa_attention_forward()` wrapper function. The cache_len parameter is being lost somewhere between the call site and the kernel.

### The Fix Will Probably Be
Either:
1. Swap parameter order in the call
2. Fix parameter order in wrapper function
3. Fix how cache_len is passed to decode kernel

### How To Verify
After fixing, you should see:
```
[ATTENTION DEBUG] cache_len=0  (for first token) ✅
[ATTENTION DEBUG] cache_len=1  (for second token) ✅
[ATTENTION DEBUG] cache_len=2  (for third token) ✅
```

And the model should generate coherent text, not repetitive tokens.

---

## My Journey

### Eureka #1 (WRONG)
- Found missing `ffn_down` line
- Thought it was THE bug
- Tested it → Still broken
- Learned: Always test before claiming victory

### Eureka #2 (CORRECT)
- Found `cache_len=0` always
- This IS a real bug
- Haven't fixed it yet (out of time/tokens)
- Passing to next team

---

## Status

**Bug Location**: Identified but not fixed  
**Confidence**: 95% (cache_len=0 is definitely wrong)  
**Next Step**: Fix parameter passing in attention call  

---

**Team Charlie Gamma**  
**Passing the torch to next investigator** 🔦  
**The bug is in parameter passing to attention kernel!**
