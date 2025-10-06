# Team VANGUARD → Next Team Handoff

**Date:** 2025-10-07T20:15Z  
**Status:** ❌ Bug NOT fixed - Critical dequantization bug found but output still garbage

---

## 🎯 Mission Accomplished

Executed systematic weight integrity verification as commanded:
- ✅ Dumped first 100 FP16 values from GPU memory for 8 critical tensors
- ✅ Found Q4_K dequantization formula bug (+ should be -)
- ❌ Fix applied but model still outputs garbage

---

## 🔍 Critical Bug Found (Partial Fix)

### Bug #1: Q4_K Dequantization Formula Sign Error

**Location:** `cuda/kernels/q4_k_dequant.cu` line 140

**WRONG (before):**
```cuda
float result = scale * static_cast<float>(q) + min_val;
```

**CORRECT (after):**
```cuda
float result = scale * static_cast<float>(q) - min_val;  // MINUS not PLUS
```

**Evidence:**
- Compared with llama.cpp: `ggml/src/ggml-cuda/convert.cu` line 224
- llama.cpp uses: `y[l + 0] = d1 * (q[l] & 0xF) - m1;` (MINUS)
- Our code used PLUS, causing all weights to be shifted incorrectly

**Impact:**
- This bug affects EVERY Q4_K quantized weight in the model
- Would cause systematic bias in all layer weights
- Explains why algorithms are correct but output is wrong

---

## ❌ Why Fix Didn't Work

After applying the sign fix and rebuilding:
- Model still outputs garbage: "ĠgeniÅŁĠ'**à¸ªà¸ķà¹ĮĠ'**(ICĠinsults..."
- No improvement in output quality
- Still generates random foreign/code tokens

**Hypothesis:** There are additional bugs in Q4_K dequantization beyond the sign error.

---

## 🔬 Additional Suspects

### Suspect #1: Scale/Min Decoding Logic

Our `decode_scales_and_mins()` function may not match llama.cpp's `get_scale_min_k4()`:

**llama.cpp approach:**
```cuda
static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
```

**Our approach:**
- Decodes all 8 scales and 8 mins upfront in `decode_scales_and_mins()`
- Uses complex bit-shifting logic that may not match llama.cpp exactly

**Action Required:**
1. Trace through llama.cpp's `get_scale_min_k4` for indices 0-7
2. Trace through our `decode_scales_and_mins` for same inputs
3. Compare outputs byte-for-byte
4. Fix any mismatches in bit-packing logic

### Suspect #2: Thread Organization Mismatch

**llama.cpp:**
- Uses 32 threads per block: `dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);`
- Thread organization: `il = tid/8`, `ir = tid%8`, `is = 2*il`
- Each thread processes 8 elements (4 low nibbles + 4 high nibbles)

**Our code:**
- Uses 256 threads per block: `dim3 block(Q4K_BLOCK_SIZE, 1, 1);` where Q4K_BLOCK_SIZE=256
- Thread organization: `s = thread_idx / Q4K_SUB_BLOCK_SIZE` where Q4K_SUB_BLOCK_SIZE=32
- Each thread processes 1 element

**Potential Issue:**
- Different thread organization may cause index calculation mismatches
- Scale/min indexing may be off by one or accessing wrong sub-block

**Action Required:**
1. Verify our thread-to-element mapping matches llama.cpp's intent
2. Consider rewriting kernel to match llama.cpp's 32-thread organization
3. Or prove our 256-thread approach is mathematically equivalent

### Suspect #3: dm Union Access

**llama.cpp block structure:**
```c
union {
    struct {
        ggml_half d;    // super-block scale
        ggml_half dmin; // super-block min-scale
    } GGML_COMMON_AGGR;
    ggml_half2 dm;
} GGML_COMMON_AGGR_U;
```

**llama.cpp access:**
```cuda
const float dall = __low2half(x[i].dm);   // Uses half2
const float dmin = __high2half(x[i].dm);
```

**Our code:**
```cuda
half d_half = __ushort_as_half(block->d);     // Separate fields
half dmin_half = __ushort_as_half(block->dmin);
```

**Potential Issue:**
- llama.cpp uses `half2` union for atomic access
- We access `d` and `dmin` as separate uint16_t fields
- Endianness or alignment could cause swapped values

**Action Required:**
1. Check if `d` and `dmin` are in correct order in memory
2. Verify `__low2half` gets `d` not `dmin`
3. Test swapping our `d` and `dmin` access

---

## 📊 Weight Dump Output

Successfully captured first 100 FP16 values for:
- ✅ `blk.0.attn_q.weight`
- ✅ `blk.0.attn_k.weight`
- ✅ `blk.0.attn_v.weight`
- ✅ `blk.0.attn_output.weight`
- ✅ `blk.0.ffn_gate.weight`
- ✅ `blk.0.ffn_up.weight`
- ✅ `blk.0.ffn_down.weight`
- ✅ `output.weight`

Output saved in test logs. Next team should compare with llama.cpp dumps.

---

## 🎯 Recommended Next Steps

### Priority 1: Fix Remaining Q4_K Dequantization Bugs (HIGHEST)

**Approach:**
1. **Verify scale/min decoding:**
   - Add debug prints to dump all 8 decoded scales and 8 mins
   - Implement llama.cpp's get_scale_min_k4 exactly
   - Compare outputs for first Q4_K block

2. **Test d/dmin order:**
   - Swap `d` and `dmin` in our code
   - Run test to see if output improves
   - If so, we had them backwards

3. **Match thread organization:**
   - Consider rewriting kernel to use 32 threads like llama.cpp
   - Or prove our 256-thread approach is equivalent

4. **Create minimal test:**
   - Load one Q4_K block from model file
   - Dequantize with our kernel → dump 256 floats
   - Dequantize with llama.cpp → dump 256 floats
   - Find first mismatch → that's where the bug is

### Priority 2: Compare with llama.cpp Reference Implementation

If Q4_K fixes don't work:
1. Create llama.cpp weight dumper (started in `investigation-teams/llama_cpp_weight_dumper.cpp`)
2. Dump same tensors from llama.cpp after dequantization
3. Byte-for-byte comparison with our dumps
4. Identify first divergence → that's the bug

---

## 📁 Files Modified

1. `cuda/kernels/q4_k_dequant.cu` (line 152):
   - Changed `+ min_val` to `- min_val`
   - Added TEAM VANGUARD comment documenting bug

2. `cuda/src/model/qwen_weight_loader.cpp` (lines 382-438):
   - Added weight dump instrumentation
   - Dumps first 100 FP16 values for 8 critical tensors
   - Includes float values, raw bytes, and statistics

3. `investigation-teams/llama_cpp_weight_dumper.cpp`:
   - Started C++ tool to dump llama.cpp weights
   - Needs completion to dequantize Q4_K properly

---

## 🧪 Test Results

**Command:**
```bash
REQUIRE_REAL_LLAMA=1 cargo test --release --features cuda \
  --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  -- --ignored --nocapture --test-threads=1
```

**Result:** ❌ FAIL
- Generated 100 tokens
- Output: "ĠgeniÅŁĠ'**à¸ªà¸ķà¹ĮĠ'**(ICĠinsultsĠTB..."
- Still garbage (foreign languages, code tokens)
- Quality check failed: "forty-eight" not found

---

## 💡 Key Insights

1. **The bug is definitively in Q4_K dequantization**
   - Found one bug (sign error), but more remain
   - All other components verified correct by previous teams
   - llama.cpp works with same model → our dequant is wrong

2. **Systematic approach is working**
   - Weight dumps give us concrete data to compare
   - Side-by-side comparison with llama.cpp will find remaining bugs
   - Don't need to guess - can prove bugs with evidence

3. **Complex bit-packing is error-prone**
   - Q4_K uses intricate 6-bit scale packing
   - Small mistakes in bit-shifting cause wrong weights
   - Must match llama.cpp's logic exactly

---

## 🚦 Definition of Done (Not Met)

- ❌ Haiku test passes with human-readable output
- ❌ Generated text contains minute word (e.g., "forty-eight")
- ✅ Weight dumps captured for comparison
- ✅ At least one Q4_K bug found and fixed (sign error)
- ❌ All Q4_K bugs fixed (more remain)

---

**Team VANGUARD**  
*"Found the dequant bug. More work needed."*

**Handoff Complete:** 2025-10-07T20:15Z
