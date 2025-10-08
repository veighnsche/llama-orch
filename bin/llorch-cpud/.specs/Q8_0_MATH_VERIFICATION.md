# Q8_0 Quantization Math Verification

**Created by:** TEAM-008  
**Date:** 2025-10-08  
**Reference:** llama.cpp `ggml-common.h`

---

## Verification Against llama.cpp

### llama.cpp Definition

**File:** `reference/llama.cpp/ggml/src/ggml-common.h:219-224`

```c
#define QK8_0 32
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0, "wrong q8_0 block size/padding");
```

**Where:**
- `ggml_half` = `uint16_t` (2 bytes) - FP16 format
- `QK8_0` = 32 (block size)
- `int8_t` = 1 byte

---

## Math Verification

### Block Structure
```
block_q8_0 {
    d: ggml_half (2 bytes)    // Scale factor (FP16)
    qs: int8_t[32] (32 bytes) // Quantized values
}
```

### Size Calculation

**Our implementation:**
```rust
Self::Q8_0 => 36,  // 32 * 1 + 4 (32 int8 + 1 float32 scale)
```

**llama.cpp actual:**
```c
sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0
                   == 2 bytes + 32 bytes
                   == 34 bytes
```

---

## ❌ ERROR FOUND

### Our Implementation (WRONG)
- **Block size:** 32 ✅ Correct
- **Bytes per block:** 36 ❌ **INCORRECT**
- **Assumption:** 1 float32 scale (4 bytes) + 32 int8 (32 bytes) = 36 bytes

### llama.cpp (CORRECT)
- **Block size:** 32 ✅
- **Bytes per block:** 34 ✅
- **Actual:** 1 FP16 scale (2 bytes) + 32 int8 (32 bytes) = 34 bytes

### Root Cause
We assumed the scale factor `d` is `float32` (4 bytes), but llama.cpp uses `ggml_half` which is **FP16** (2 bytes).

---

## Required Fix

### File: `src/model/gguf_parser.rs`

**Line 122 - INCORRECT:**
```rust
Self::Q8_0 => 36,  // 32 * 1 + 4 (32 int8 + 1 float32 scale)
```

**Should be:**
```rust
Self::Q8_0 => 34,  // 32 * 1 + 2 (32 int8 + 1 FP16 scale)
```

**Comment should explain:**
```rust
Self::Q8_0 => 34,  // 32 int8 values + 1 FP16 scale (2 bytes)
```

---

## Impact Analysis

### Where This Matters
1. **Weight Loading:** Calculating tensor sizes from GGUF file
2. **Memory Allocation:** Allocating correct buffer sizes
3. **File Offsets:** Computing correct byte offsets for tensors
4. **Dequantization:** Reading the correct number of bytes per block

### Severity
**HIGH** - This will cause:
- ❌ Incorrect tensor size calculations
- ❌ Wrong memory allocations (2 bytes too large per block)
- ❌ Misaligned tensor reads
- ❌ Potential segfaults or data corruption

### Example Impact
For a tensor with 4,194,304 elements (e.g., `[4096, 1024]`):
- Number of blocks: 4,194,304 / 32 = 131,072 blocks
- **Our calculation:** 131,072 × 36 = 4,718,592 bytes
- **Correct calculation:** 131,072 × 34 = 4,456,448 bytes
- **Error:** 262,144 bytes (256 KB) too large!

For the full Llama-2 7B model:
- Approximate quantized weights: ~7 GB
- Error accumulation: Could be several MB off

---

## Verification Commands

### Check llama.cpp Source
```bash
cd reference/llama.cpp
grep -A 5 "block_q8_0" ggml/src/ggml-common.h
```

**Output:**
```c
#define QK8_0 32
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0, "wrong q8_0 block size/padding");
```

### Verify ggml_half Size
```bash
grep "typedef.*ggml_half" reference/llama.cpp/ggml/src/ggml-common.h
```

**Output:**
```c
typedef uint16_t ggml_half;  // 2 bytes (FP16)
```

---

## Additional Notes

### Q8_1 for Comparison
```c
#define QK8_1 32
typedef struct {
    ggml_half d; // delta
    ggml_half s; // d * sum(qs[i])
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2*sizeof(ggml_half) + QK8_1, "wrong q8_1 block size/padding");
```

**Q8_1 size:** 2 × 2 + 32 = **36 bytes**

So our 36-byte calculation was actually correct for Q8_1, but wrong for Q8_0!

### Why FP16 for Scale?
- **Memory efficiency:** 2 bytes vs 4 bytes
- **Sufficient precision:** FP16 has ~3 decimal digits of precision, enough for scale factors
- **Hardware support:** Modern GPUs have native FP16 operations
- **GGUF standard:** Uses FP16 for quantization scales

---

## Fix Implementation

### Before (INCORRECT)
```rust
pub fn bytes_per_block(&self) -> usize {
    match self {
        Self::F32 => 4,
        Self::F16 => 2,
        Self::Q4_0 => 18,  // 16 * 0.5 + 2
        Self::Q4_1 => 20,  // 16 * 0.5 + 4
        Self::Q5_0 => 22,  // 16 * 0.625 + 2
        Self::Q5_1 => 24,  // 16 * 0.625 + 4
        Self::Q8_0 => 36,  // 32 * 1 + 4 (32 int8 + 1 float32 scale) ❌ WRONG
        Self::Q8_1 => 40,  // 32 * 1 + 8
        // ... rest
    }
}
```

### After (CORRECT)
```rust
pub fn bytes_per_block(&self) -> usize {
    match self {
        Self::F32 => 4,
        Self::F16 => 2,
        Self::Q4_0 => 18,  // 16 * 0.5 + 2
        Self::Q4_1 => 20,  // 16 * 0.5 + 4
        Self::Q5_0 => 22,  // 16 * 0.625 + 2
        Self::Q5_1 => 24,  // 16 * 0.625 + 4
        Self::Q8_0 => 34,  // 32 int8 + 1 FP16 scale (2 bytes) ✅ CORRECT
        Self::Q8_1 => 36,  // 32 int8 + 2 FP16 (d + s) (4 bytes) ✅ CORRECT
        // ... rest
    }
}
```

**Note:** Q8_1 also needs correction: 32 + 2×2 = 36, not 40!

---

## Test After Fix

### Verify Tensor Sizes
```bash
cargo run --example test_gguf_parser
```

**Expected output should show correct sizes:**
- Token embeddings: `[4096, 32000]` Q8_0
- Attention weights: `[4096, 4096]` Q8_0
- FFN weights: `[4096, 11008]` Q8_0

### Calculate Expected Size
For `token_embd.weight: [4096, 32000]` Q8_0:
- Elements: 4096 × 32000 = 131,072,000
- Blocks: 131,072,000 / 32 = 4,096,000
- Bytes: 4,096,000 × 34 = 139,264,000 bytes ≈ 132.8 MB

---

## Lessons Learned

### Always Verify Against Reference
- ✅ Check source code, not documentation
- ✅ Verify data structure sizes
- ✅ Don't assume standard sizes (float32 vs FP16)

### GGUF Uses FP16 Extensively
- Quantization scales are FP16
- Reduces memory footprint
- Sufficient precision for scales

### Test Early
- This error would have caused issues during weight loading
- Caught during math verification before implementation
- Saved debugging time later

---

## Sign-off

**Verified by:** TEAM-008  
**Date:** 2025-10-08  
**Status:** ❌ Error found, fix required

**Action Required:** Update `bytes_per_block()` for Q8_0 and Q8_1 before proceeding with weight loading.

---

*"Measure twice, cut once. Verify against source, not assumptions."*  
— TEAM-008, Foundation Implementation Division

**END VERIFICATION**
