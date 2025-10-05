# Research Needed: Q5_0, Q6_K, Q8_0 Dequantization

**Date**: 2025-10-05  
**Status**: 🔴 **BLOCKING HAIKU TEST** - Need these formats to use Qwen2.5-0.5B Q4_K_M model  
**Priority**: P0 - Blocks real inference with current model

---

## Problem Statement

We have successfully implemented:
- ✅ Q4_K dequantization (working!)
- ✅ Rust weight loading infrastructure
- ✅ GPU memory allocation and upload
- ✅ C++ integration

**BUT**: The Qwen2.5-0.5B Q4_K_M model uses **mixed quantization**:
- `token_embd.weight`: Q4_K ✅ (we support this)
- Most attention weights: **Q5_0** ❌ (not implemented)
- FFN down weights: **Q6_K** ❌ (not implemented)  
- Some attention V weights: **Q8_0** ❌ (not implemented)

**Result**: ~80% of weights are zeros → model can't generate text

---

## What We Need to Research

### 1. Q8_0 Format Specification

**Questions**:
- What is the exact binary layout of Q8_0 blocks?
- How many elements per block?
- How are scale factors stored?
- How are 8-bit values packed?
- What is the dequantization formula?

**Expected Complexity**: LOW (simplest format)

**Sources to Check**:
- GGML source code: `ggml-quants.c` and `ggml-quants.h`
- llama.cpp: `ggml-cuda/dequantize.cu`
- GGUF specification document

**Expected Output**:
```
Q8_0 Block Structure:
- Block size: 32 elements
- Scale storage: 1 fp16 (2 bytes)
- Quantized values: 32 × 8-bit (32 bytes)
- Total bytes per block: 34 bytes

Dequantization Formula:
value_fp16 = quantized_int8 * scale
```

---

### 2. Q5_0 Format Specification

**Questions**:
- What is the exact binary layout of Q5_0 blocks?
- How many elements per block?
- How are scale factors stored?
- How are 5-bit values packed into bytes?
- What is the dequantization formula?

**Expected Complexity**: MEDIUM (5-bit packing is tricky)

**Sources to Check**:
- GGML source code: `ggml-quants.c`
- llama.cpp: `ggml-cuda/dequantize.cu`
- Community documentation

**Expected Output**:
```
Q5_0 Block Structure:
- Block size: 32 elements
- Scale storage: 1 fp16 (2 bytes)
- High bits: 4 bytes (1 bit per element)
- Low bits: 16 bytes (4 bits per element)
- Total bytes per block: 22 bytes

Dequantization Formula:
q5 = (low_4bit | (high_1bit << 4)) - 16  // 5-bit signed
value_fp16 = q5 * scale
```

---

### 3. Q6_K Format Specification

**Questions**:
- What is the exact binary layout of Q6_K blocks?
- How many elements per block?
- How are scale factors stored?
- How are 6-bit values packed?
- What is the dequantization formula?
- Is it block-based like Q4_K or different?

**Expected Complexity**: HIGH (K-quant family, complex structure)

**Sources to Check**:
- GGML source code: `ggml-quants.c`
- llama.cpp: `ggml-cuda/dequantize.cu`
- GGUF specification

**Expected Output**:
```
Q6_K Block Structure:
- Block size: 256 elements (like Q4_K)
- Super-block structure: TBD
- Scale storage: TBD
- Quantized values: 6-bit per element
- Total bytes per block: ~210 bytes

Dequantization Formula:
TBD (research needed)
```

---

## Research Deliverables

### Document 1: Q8_0 Format Specification
**File**: `Q8_0_FORMAT_SPEC.md`

**Contents**:
- Binary layout diagram
- Block structure details
- Scale encoding
- 8-bit value interpretation
- Dequantization formula
- Example block with hex dump
- Pseudocode implementation

**Estimated Time**: 30 minutes

---

### Document 2: Q5_0 Format Specification
**File**: `Q5_0_FORMAT_SPEC.md`

**Contents**:
- Binary layout diagram
- Block structure details
- Scale encoding
- 5-bit packing scheme (high/low bits)
- Dequantization formula
- Example block with hex dump
- Pseudocode implementation

**Estimated Time**: 1 hour

---

### Document 3: Q6_K Format Specification
**File**: `Q6_K_FORMAT_SPEC.md`

**Contents**:
- Binary layout diagram
- Block/super-block structure
- Scale/min encoding
- 6-bit packing scheme
- Dequantization formula
- Comparison with Q4_K
- Example block with hex dump
- Pseudocode implementation

**Estimated Time**: 1.5 hours

---

### Document 4: Implementation Plan
**File**: `Q5_Q6_Q8_DEQUANT_IMPL_PLAN.md`

**Contents**:
- Implementation order (Q8_0 → Q5_0 → Q6_K)
- Rust code structure
- Testing strategy
- Integration with existing weight loader
- Performance estimates

**Estimated Time**: 30 minutes

---

## Implementation Estimates

### Q8_0 Dequantization
**Complexity**: LOW  
**Time**: 30 minutes  
**Lines of Code**: ~50 lines

**Rationale**: Simplest format, just scale * int8

---

### Q5_0 Dequantization
**Complexity**: MEDIUM  
**Time**: 1 hour  
**Lines of Code**: ~100 lines

**Rationale**: Need to unpack 5-bit values from high/low bit arrays

---

### Q6_K Dequantization
**Complexity**: HIGH  
**Time**: 2 hours  
**Lines of Code**: ~200 lines

**Rationale**: Similar to Q4_K but with 6-bit values and different packing

---

### Total Time Estimate
- **Research**: 3 hours
- **Implementation**: 3.5 hours
- **Testing**: 1 hour
- **Total**: **7.5 hours**

---

## Testing Strategy

### 1. Unit Tests per Format

For each format (Q8_0, Q5_0, Q6_K):

**Test 1: Block Size Verification**
```rust
#[test]
fn test_q8_0_block_size() {
    assert_eq!(std::mem::size_of::<Q8_0Block>(), 34);
}
```

**Test 2: Zero Block Dequantization**
```rust
#[test]
fn test_q8_0_dequantize_zero_block() {
    let block = vec![0u8; 34];
    let mut output = vec![f16::ZERO; 32];
    dequantize_q8_0_block(&block, &mut output);
    // All outputs should be zero
}
```

**Test 3: Known Value Test**
```rust
#[test]
fn test_q8_0_known_values() {
    // Create block with known scale and values
    // Verify dequantization matches expected output
}
```

---

### 2. Cross-Validation with gguf-py

**Strategy**:
1. Extract a sample block from the Qwen model
2. Dequantize using our Rust implementation
3. Dequantize using `gguf-py` Python library
4. Compare outputs (should match within epsilon)

**Example**:
```python
# Python validation script
from gguf import Q8_0, Q5_0, Q6_K
import numpy as np

# Load sample block
block_bytes = open('sample_q8_0_block.bin', 'rb').read(34)

# Dequantize with gguf-py
expected = Q8_0.dequantize_blocks(np.frombuffer(block_bytes, dtype=np.uint8))

# Compare with Rust output
rust_output = np.load('rust_q8_0_output.npy')
assert np.allclose(expected, rust_output, atol=1e-3)
```

---

### 3. Integration Test

**Test**: Load Qwen2.5-0.5B model with all formats
```rust
#[test]
fn test_load_qwen_model_all_formats() {
    let weights = load_weights_to_gpu("qwen2.5-0.5b-instruct-q4_k_m.gguf")?;
    
    // Verify all 291 tensors loaded
    assert_eq!(weights.len(), 291);
    
    // Verify no zeros (all formats supported)
    for (name, ptr) in &weights {
        // Check first few values are non-zero
    }
}
```

---

## Success Criteria

After research and implementation:
- [ ] Q8_0 format fully documented
- [ ] Q5_0 format fully documented
- [ ] Q6_K format fully documented
- [ ] All three formats implemented in Rust
- [ ] Unit tests pass for all formats
- [ ] Cross-validation with gguf-py passes
- [ ] Qwen2.5-0.5B loads with real weights (no zeros)
- [ ] Embeddings show reasonable values (-1.0 to 1.0)
- [ ] No NaN in logits
- [ ] **Haiku test generates real text** ✅

---

## References to Find

### Primary Sources
- [ ] GGML repository: `ggml-quants.c` (Q8_0, Q5_0, Q6_K implementations)
- [ ] llama.cpp: `dequantize.cu` (GPU implementations)
- [ ] GGUF specification document
- [ ] gguf-py: `quants.py` (Python reference implementations)

### Community Resources
- [ ] Reddit r/LocalLLaMA discussions on quantization
- [ ] llama.cpp GitHub issues about Q5_0/Q6_K/Q8_0
- [ ] GGML documentation
- [ ] Blog posts on GGUF quantization formats

### Code Examples
- [ ] llama.cpp test files for each format
- [ ] gguf-py example scripts
- [ ] Community implementations

---

## Implementation Order

### Phase 1: Q8_0 (Easiest)
1. Research format (30 mins)
2. Implement dequantization (30 mins)
3. Write tests (15 mins)
4. Validate with gguf-py (15 mins)

**Total**: 1.5 hours

---

### Phase 2: Q5_0 (Medium)
1. Research format (1 hour)
2. Implement bit unpacking (30 mins)
3. Implement dequantization (30 mins)
4. Write tests (15 mins)
5. Validate with gguf-py (15 mins)

**Total**: 2.5 hours

---

### Phase 3: Q6_K (Hardest)
1. Research format (1.5 hours)
2. Study Q4_K implementation for patterns (30 mins)
3. Implement scale/min decoding (30 mins)
4. Implement 6-bit unpacking (30 mins)
5. Implement dequantization (30 mins)
6. Write tests (30 mins)
7. Validate with gguf-py (30 mins)

**Total**: 4.5 hours

---

### Phase 4: Integration
1. Update `load_tensor()` to use new formats (15 mins)
2. Run full model load test (15 mins)
3. Debug any issues (30 mins)
4. Run haiku test (5 mins)

**Total**: 1 hour

---

## Alternative: Quick Workaround

While researching, we could:
1. Download FP16 model (5 minutes)
2. Verify haiku test passes with FP16
3. Then implement quantization formats properly

This de-risks the work and proves the infrastructure is correct.

---

## Notes

- **NO_LLAMA_CPP rule**: We can study their code but must write our own
- **Q4_K already works**: Use it as reference for K-quant family patterns
- **Rust implementation**: All dequantization in pure Rust (no C++)
- **This is the ONLY blocker**: Everything else works!

---

## Current Status

**What's Blocking**:
```
⚠️  [Rust] Unsupported quantization Q5_0 for tensor blk.0.attn_q.weight, using zeros
⚠️  [Rust] Unsupported quantization Q6_K for tensor blk.0.ffn_down.weight, using zeros
⚠️  [Rust] Unsupported quantization Q8_0 for tensor blk.0.attn_v.weight, using zeros
```

**Impact**: ~230 out of 291 tensors are zeros → model can't work

**Solution**: Implement these 3 formats → **haiku test passes** ✅

---

**Next Action**: Assign researcher to spend 3 hours studying GGML/llama.cpp Q8_0, Q5_0, and Q6_K implementations and document findings.

---

**Built by Foundation-Alpha 🏗️**
**Researched by Cascade 🔬**
