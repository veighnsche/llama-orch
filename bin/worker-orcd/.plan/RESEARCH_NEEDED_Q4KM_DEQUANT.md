# Research Needed: Q4_K_M Dequantization for Qwen2.5-0.5B

**Date**: 2025-10-05  
**Status**: 🔴 **CRITICAL BLOCKER** - Cannot run real inference without this  
**Priority**: P0 - Blocks haiku test completion

---

## Problem Statement

We have successfully implemented:
- ✅ CUDA transformer kernels (RMSNorm, RoPE, Attention, FFN)
- ✅ Weight loading infrastructure
- ✅ Tokenizer integration
- ✅ Inference pipeline
- ✅ SSE streaming

**BUT**: The Qwen2.5-0.5B model is in Q4_K_M quantized format, and we're treating quantized bytes as FP16, causing:
- NaN values in embeddings
- NaN propagation through all layers
- Always sampling token ID 0

---

## What We Need to Research

### 1. Q4_K_M Format Specification

**Questions**:
- What is the exact binary layout of Q4_K_M blocks?
- How many elements per block?
- How are scale factors stored?
- How are quantized values packed?
- What is the dequantization formula?

**Sources to Check**:
- GGML source code: `ggml-quants.c` and `ggml-quants.h`
- llama.cpp: `ggml-cuda/dequantize.cu`
- GGUF specification document
- Community documentation (Reddit, GitHub issues)

**Expected Output**:
```
Block Structure:
- Block size: 256 elements
- Super-block structure: 8 sub-blocks of 32 elements
- Scale storage: 6-bit per sub-block
- Min storage: 4-bit per sub-block
- Quantized values: 4-bit per element
- Total bytes per block: X bytes

Dequantization Formula:
value_fp16 = (quantized_4bit * scale) + min
```

### 2. Dequantization Algorithm

**Questions**:
- How to parse a Q4_K_M block?
- How to extract scale factors?
- How to unpack 4-bit values from bytes?
- How to apply dequantization formula?
- Can we do this on CPU or need GPU kernel?

**Sources to Check**:
- llama.cpp: `convert-hf-to-gguf.py` (quantization side)
- llama.cpp: `ggml-cuda/dequantize.cu` (GPU dequantization)
- ggml: `ggml-quants.c` (CPU dequantization)

**Expected Output**:
```cpp
// Pseudocode
struct Q4_K_Block {
    uint8_t scales[...];  // Scale factors
    uint8_t mins[...];    // Min values
    uint8_t qs[...];      // Quantized values (packed 4-bit)
};

void dequantize_q4_k_m_block(
    const Q4_K_Block* block,
    half* output,  // 256 FP16 values
    int block_idx
) {
    // 1. Extract scales and mins
    // 2. Unpack 4-bit quantized values
    // 3. Apply formula: output[i] = (qs[i] * scale) + min
}
```

### 3. Implementation Strategy

**Questions**:
- Should we dequantize during weight loading (one-time cost)?
- Or dequantize on-the-fly during inference (save VRAM)?
- What's the VRAM tradeoff?
- What's the performance impact?

**Analysis Needed**:
```
Option A: Dequantize on Load
- VRAM: Q4_K_M (1.2GB) → FP16 (2.4GB)
- Speed: One-time cost, fast inference
- Complexity: Simple

Option B: On-the-fly Dequantization
- VRAM: Keep Q4_K_M (1.2GB)
- Speed: Dequant overhead on every forward pass
- Complexity: Need custom CUDA kernels
```

**Recommendation**: Start with Option A (dequantize on load) for simplicity.

### 4. Code References

**Questions**:
- Where in llama.cpp is Q4_K_M dequantization implemented?
- Can we adapt the algorithm (not copy code)?
- Are there any patents or licensing issues?

**Files to Study**:
```
llama.cpp:
- ggml/src/ggml-quants.c (CPU dequant)
- ggml/src/ggml-cuda/dequantize.cu (GPU dequant)
- ggml/include/ggml-quants.h (format definitions)

GGML:
- Same files in standalone GGML repo
```

**Note**: We can study the algorithm but must implement our own version (NO_LLAMA_CPP rule).

### 5. Testing Strategy

**Questions**:
- How to verify dequantization is correct?
- What test cases to use?
- How to compare with reference implementation?

**Test Plan**:
1. Load a small Q4_K_M tensor (e.g., 256 elements)
2. Dequantize using our implementation
3. Compare with llama.cpp output (via Python script)
4. Verify values match within epsilon
5. Test edge cases (zeros, max values, negatives)

---

## Research Deliverables

### Document 1: Q4_K_M Format Specification
**File**: `Q4_K_M_FORMAT_SPEC.md`

**Contents**:
- Binary layout diagram
- Block structure details
- Scale/min encoding
- Quantized value packing
- Dequantization formula
- Example block with hex dump

### Document 2: Dequantization Implementation Plan
**File**: `Q4_K_M_DEQUANT_IMPL_PLAN.md`

**Contents**:
- Algorithm pseudocode
- C++ implementation outline
- CUDA kernel design (if needed)
- Memory layout considerations
- Performance estimates

### Document 3: Reference Code Analysis
**File**: `Q4_K_M_REFERENCE_ANALYSIS.md`

**Contents**:
- llama.cpp implementation review
- GGML implementation review
- Key insights and learnings
- Differences from our approach
- Licensing notes

---

## Time Estimate

- **Research**: 2-3 hours
  - Read GGML source code: 1 hour
  - Study llama.cpp dequant: 1 hour
  - Document findings: 1 hour

- **Implementation**: 2-3 hours
  - Write dequantization function: 1 hour
  - Integrate into weight loader: 1 hour
  - Test and debug: 1 hour

- **Total**: 4-6 hours

---

## Alternative: Quick Test with FP16

While researching, we can:
1. Download FP16 model (5 minutes)
2. Test that pipeline works end-to-end
3. Prove infrastructure is correct
4. Then implement Q4_K_M dequant

This de-risks the work and proves the 99% of code that's already done.

---

## Success Criteria

After research and implementation:
- [ ] Q4_K_M format fully documented
- [ ] Dequantization algorithm implemented
- [ ] Weights load as valid FP16 values
- [ ] Embeddings show reasonable values (-1.0 to 1.0)
- [ ] No NaN in logits
- [ ] Haiku test generates real text (not "!!!!")
- [ ] Model produces coherent output

---

## References to Find

### Primary Sources
- [ ] GGML repository: `ggml-quants.c`
- [ ] llama.cpp: `dequantize.cu`
- [ ] GGUF specification document
- [ ] GGML quantization paper/docs

### Community Resources
- [ ] Reddit r/LocalLLaMA discussions on Q4_K_M
- [ ] llama.cpp GitHub issues about quantization
- [ ] GGML documentation
- [ ] Blog posts on GGUF quantization

### Academic/Technical
- [ ] Quantization papers (if any specific to GGML)
- [ ] NVIDIA quantization best practices
- [ ] FP16/INT4 conversion techniques

---

## Notes

- **NO_LLAMA_CPP rule**: We can study their code but must write our own
- **MXFP4 exists**: We already have MXFP4 dequant for GPT models
- **This is the ONLY blocker**: Everything else works!

---

**Next Action**: Assign researcher to spend 2-3 hours studying GGML/llama.cpp Q4_K_M implementation and document findings.

---

**Built by Foundation-Alpha 🏗️**
**Researched by Cascade 🔬**
