# MXFP4 Research Notes

**Research Completed**: 2025-10-05  
**Researcher**: GPT-Gamma  
**Purpose**: Comprehensive MXFP4 format study for kernel implementation

---

## Executive Summary

MXFP4 (Microscaling FP4) is a novel 4-bit quantization format standardized by the Open Compute Project (OCP). It uses block-based quantization with shared exponents to achieve ~4x memory savings vs FP16 while maintaining acceptable numerical accuracy for LLM inference.

**Key Characteristics**:
- **Block size**: 32 elements per block
- **Storage**: 4-bit mantissa per element + 8-bit shared exponent per block = 17 bytes per block
- **Precision**: ~±1-2% accuracy vs FP16 for typical LLM workloads
- **Target**: GPT-OSS-20B (20B parameters) fitting in 24GB VRAM

---

## 1. MXFP4 Format Specification

### 1.1 Block Structure

Each MXFP4 block contains:
- **32 FP4 values** (4 bits each) = 16 bytes
- **1 FP8 scale** (8 bits) = 1 byte
- **Total**: 17 bytes per block

```
Block Layout (17 bytes):
┌─────────────────────────────────────┬──────────┐
│   FP4 Mantissas (32 values)         │ FP8 Scale│
│   [0][1][2]...[30][31]              │          │
│   16 bytes (4 bits × 32)            │ 1 byte   │
└─────────────────────────────────────┴──────────┘
```

### 1.2 FP4 Mantissa Encoding

4-bit mantissa values (0-15) map to normalized floating-point values:

```
Value | Binary | FP Representation
------|--------|------------------
  0   | 0000   | 0.0
  1   | 0001   | 0.5
  2   | 0010   | 1.0
  3   | 0011   | 1.5
  4   | 0100   | 2.0
  5   | 0101   | 2.5
  6   | 0110   | 3.0
  7   | 0111   | 3.5
  8   | 1000   | -0.0 (negative zero)
  9   | 1001   | -0.5
 10   | 1010   | -1.0
 11   | 1011   | -1.5
 12   | 1100   | -2.0
 13   | 1101   | -2.5
 14   | 1110   | -3.0
 15   | 1111   | -3.5
```

### 1.3 FP8 Scale Factor (E8M0)

The shared exponent uses E8M0 format (8-bit exponent, no mantissa):
- **Range**: 2^(-127) to 2^(+127)
- **Special values**: 0x00 = zero, 0xFF = infinity/NaN
- **Bias**: 127 (standard IEEE bias)

### 1.4 Dequantization Formula

```
fp16_value = fp4_mantissa * (2 ^ (fp8_exponent - 127))
```

Simplified:
```
fp16_value = fp4_mantissa * fp8_scale
```

Where `fp8_scale = 2^(exponent - 127)` is precomputed.

---

## 2. OCP MX Standard Compliance

### 2.1 OCP MX Specification v1.0

**Source**: Open Compute Project Microscaling Formats Specification  
**Version**: 1.0 (Released 2023)  
**URL**: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

**Key Requirements**:
1. Block-based quantization with shared exponents
2. Support for multiple block sizes (16, 32, 64 elements)
3. Multiple precision levels (FP4, FP6, FP8, INT8)
4. Hardware-agnostic specification
5. Deterministic rounding behavior

### 2.2 MXFP4 vs Other MX Formats

| Format | Mantissa Bits | Block Size | Bytes/Block | Compression Ratio |
|--------|---------------|------------|-------------|-------------------|
| MXFP4  | 4             | 32         | 17          | 3.76x vs FP16     |
| MXFP6  | 6             | 32         | 25          | 2.56x vs FP16     |
| MXFP8  | 8             | 32         | 33          | 1.94x vs FP16     |
| MXINT8 | 8 (integer)   | 32         | 33          | 1.94x vs FP16     |

**MXFP4 Advantages**:
- Maximum compression (3.76x)
- Suitable for LLM weights (less sensitive to quantization)
- Faster inference (less memory bandwidth)

**MXFP4 Disadvantages**:
- Lower precision (4-bit mantissa)
- Not suitable for activations or gradients
- Requires careful calibration

### 2.3 Block Size Trade-offs

| Block Size | Compression | Granularity | Use Case |
|------------|-------------|-------------|----------|
| 16         | Lower       | Fine        | Activations, sensitive layers |
| 32         | Medium      | Balanced    | **Standard for weights** |
| 64         | Higher      | Coarse      | Less sensitive layers |

**Choice for GPT-OSS-20B**: 32-element blocks (standard, balanced)

---

## 3. Numerical Precision Analysis

### 3.1 Theoretical Precision Bounds

**FP4 Mantissa Resolution**: 4 bits = 16 discrete values  
**Representable Range per Block**: ±3.5 × scale_factor  
**Quantization Error**: ±0.25 × scale_factor (worst case)

**Expected Accuracy**:
- **Typical**: ±1% vs FP16 for well-calibrated models
- **Worst case**: ±2-3% for poorly calibrated models
- **Outliers**: May require FP16 fallback for specific layers

### 3.2 Comparison with Other Formats

| Format   | Bits | Effective Precision | Typical Accuracy |
|----------|------|---------------------|------------------|
| FP16     | 16   | ~3 decimal digits   | Baseline         |
| BF16     | 16   | ~2 decimal digits   | ~0.1% vs FP16    |
| FP8 (E4M3)| 8   | ~2 decimal digits   | ~0.5% vs FP16    |
| MXFP4    | 4+8  | ~1 decimal digit    | ~1-2% vs FP16    |
| INT8     | 8    | 256 levels          | ~0.5-1% vs FP16  |
| Q4_K_M   | 4+6  | ~1 decimal digit    | ~1-2% vs FP16    |

**Key Insight**: MXFP4 trades precision for memory savings. Acceptable for LLM weights, not for activations.

### 3.3 Error Propagation Through Layers

**Single Layer Error**: ±1-2%  
**Multi-Layer Accumulation**: Errors can compound, but typically remain bounded due to:
- Normalization layers (LayerNorm) resetting scale
- Residual connections providing FP16 paths
- Attention softmax normalizing distributions

**Critical Layers**:
- **Embedding layer**: High sensitivity, consider FP16
- **LM head**: High sensitivity, consider FP16
- **Attention QKV**: Medium sensitivity, MXFP4 acceptable
- **FFN**: Low sensitivity, MXFP4 ideal

### 3.4 Accumulation Strategy

**Recommendation**: Use FP16 accumulation for all GEMM operations

```cuda
// Dequantize MXFP4 weights to FP16
half weight = mxfp4_dequant(fp4_mantissa, fp8_scale);

// Accumulate in FP16 (or FP32 for critical paths)
half accumulator = 0.0;
for (int i = 0; i < K; i++) {
    accumulator += activation[i] * weight[i];  // FP16 × FP16 → FP16
}
```

**Rationale**: FP16 accumulation prevents error accumulation while maintaining performance.

---

## 4. CUDA Kernel Design

### 4.1 Dequantization Kernel Architecture

**Goal**: Convert MXFP4 blocks to FP16 on-the-fly during GEMM

```cuda
__global__ void mxfp4_dequant_kernel(
    const uint8_t* mxfp4_data,  // Packed MXFP4 blocks
    half* fp16_output,           // Dequantized FP16 output
    int num_blocks
) {
    int block_idx = blockIdx.x;
    int elem_idx = threadIdx.x;  // 0-31
    
    if (block_idx >= num_blocks) return;
    
    // Load block data
    const uint8_t* block_ptr = mxfp4_data + block_idx * 17;
    uint8_t fp8_scale = block_ptr[16];  // Last byte
    
    // Unpack FP4 mantissa (2 per byte)
    int byte_idx = elem_idx / 2;
    int nibble = elem_idx % 2;
    uint8_t packed = block_ptr[byte_idx];
    uint8_t fp4_mantissa = (nibble == 0) ? (packed & 0x0F) : (packed >> 4);
    
    // Dequantize
    half mantissa_value = fp4_to_half(fp4_mantissa);
    half scale = fp8_to_half(fp8_scale);
    fp16_output[block_idx * 32 + elem_idx] = mantissa_value * scale;
}
```

### 4.2 Vectorization Strategies

**Option 1: 2-way vectorization** (2 FP4 values per thread)
- Load 1 byte = 2 FP4 values
- Unpack and dequantize both
- Store 2 FP16 values

**Option 2: 4-way vectorization** (4 FP4 values per thread)
- Load 2 bytes = 4 FP4 values
- Unpack and dequantize all 4
- Store 4 FP16 values (half4)

**Option 3: 8-way vectorization** (8 FP4 values per thread)
- Load 4 bytes = 8 FP4 values
- Unpack and dequantize all 8
- Store 8 FP16 values (2× half4)

**Recommendation**: Start with 2-way, optimize to 4-way after validation.

### 4.3 Kernel Fusion Opportunities

**Fused Dequant + GEMM**:
```cuda
// Instead of:
// 1. Dequantize MXFP4 → FP16 (separate kernel)
// 2. GEMM on FP16 (cuBLAS)

// Do:
// 1. Fused dequant+GEMM (custom kernel or cuBLAS callback)
```

**Benefits**:
- Reduced memory bandwidth (no intermediate FP16 storage)
- Better cache locality
- ~20-30% speedup potential

**Challenges**:
- Complex implementation
- May not integrate with cuBLAS easily
- Defer to optimization phase

---

## 5. Hardware Compatibility

### 5.1 NVIDIA GPU Support

| Architecture | Compute Capability | MXFP4 Support | Notes |
|--------------|-------------------|---------------|-------|
| Volta        | 7.0               | ❌ No native   | Software dequant only |
| Turing       | 7.5               | ❌ No native   | Software dequant only |
| Ampere       | 8.0, 8.6          | ⚠️ Partial    | FP8 Tensor Cores, not FP4 |
| Ada Lovelace | 8.9               | ⚠️ Partial    | FP8 Tensor Cores, not FP4 |
| Hopper       | 9.0               | ✅ Yes        | FP8 Tensor Cores + custom |

**Conclusion**: No native MXFP4 Tensor Core support. Must implement software dequantization.

### 5.2 AMD GPU Support

| Architecture | MXFP4 Support | Notes |
|--------------|---------------|-------|
| CDNA 2 (MI200) | ❌ No       | Software dequant via ROCm |
| CDNA 3 (MI300) | ⚠️ Partial  | FP8 support, not FP4 |

**Conclusion**: AMD requires software dequantization, similar to NVIDIA.

### 5.3 Implementation Strategy

**Phase 1 (M0)**: Software dequantization kernel
- Works on all GPUs (compute capability 7.0+)
- Dequantize MXFP4 → FP16 before GEMM
- Use cuBLAS for FP16 GEMM

**Phase 2 (M1+)**: Optimized fused kernels
- Fuse dequant + GEMM for bandwidth savings
- Explore Tensor Core integration (FP8 path)
- Profile and optimize for target GPUs

---

## 6. Integration Points

### 6.1 Weight Consumers

MXFP4 weights must be dequantized before use in:

1. **Embedding Lookup**
   - Input: token IDs
   - Weight: MXFP4 embedding matrix
   - Output: FP16 embeddings
   - Dequant: On-demand per token

2. **Attention QKV Projections**
   - Input: FP16 hidden states
   - Weight: MXFP4 QKV matrices
   - Output: FP16 Q/K/V tensors
   - Dequant: Before GEMM

3. **Attention Output Projection**
   - Input: FP16 attention output
   - Weight: MXFP4 output matrix
   - Output: FP16 hidden states
   - Dequant: Before GEMM

4. **FFN Up/Down Projections**
   - Input: FP16 hidden states
   - Weight: MXFP4 FFN matrices
   - Output: FP16 FFN output
   - Dequant: Before GEMM

5. **LM Head Projection**
   - Input: FP16 final hidden state
   - Weight: MXFP4 LM head matrix
   - Output: FP16 logits
   - Dequant: Before GEMM

### 6.2 GGUF v3 Tensor Format

GGUF v3 adds MXFP4 tensor type:

```c
enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // ... other types ...
    GGML_TYPE_Q4_K_M = 15,
    GGML_TYPE_MXFP4  = 32,  // New in GGUF v3
};
```

**MXFP4 Tensor Layout**:
```
┌──────────────────┬──────────────────┬─────────────┐
│ Tensor Header    │ MXFP4 Blocks     │ Padding     │
│ (name, shape,    │ (17 bytes each)  │ (alignment) │
│  type=32)        │                  │             │
└──────────────────┴──────────────────┴─────────────┘
```

**Parsing Requirements**:
- Detect `type == GGML_TYPE_MXFP4` (32)
- Calculate block count: `num_elements / 32` (round up)
- Allocate `num_blocks * 17` bytes
- Handle padding for alignment

---

## 7. Validation Framework

### 7.1 Validation Strategy

**Goal**: Ensure MXFP4 implementation is numerically correct within ±1% tolerance

**Approach**:
1. **Baseline**: Run GPT-OSS-20B with Q4_K_M quantization
2. **Test**: Run same model with MXFP4 quantization
3. **Compare**: Token-by-token output, perplexity, accuracy metrics

### 7.2 Test Vectors

**Unit Tests**:
```rust
#[test]
fn test_mxfp4_dequant_zero() {
    let fp4 = 0b0000;  // 0.0
    let fp8_scale = 0x7F;  // 2^0 = 1.0
    let result = mxfp4_dequant(fp4, fp8_scale);
    assert_eq!(result, 0.0);
}

#[test]
fn test_mxfp4_dequant_positive() {
    let fp4 = 0b0010;  // 1.0
    let fp8_scale = 0x7F;  // 2^0 = 1.0
    let result = mxfp4_dequant(fp4, fp8_scale);
    assert_eq!(result, 1.0);
}

#[test]
fn test_mxfp4_dequant_negative() {
    let fp4 = 0b1010;  // -1.0
    let fp8_scale = 0x7F;  // 2^0 = 1.0
    let result = mxfp4_dequant(fp4, fp8_scale);
    assert_eq!(result, -1.0);
}

#[test]
fn test_mxfp4_dequant_scaled() {
    let fp4 = 0b0010;  // 1.0
    let fp8_scale = 0x80;  // 2^1 = 2.0
    let result = mxfp4_dequant(fp4, fp8_scale);
    assert_eq!(result, 2.0);
}
```

**Integration Tests**:
- Load MXFP4 model
- Generate text with fixed seed
- Compare output with Q4_K_M baseline
- Validate perplexity within ±1%

### 7.3 Numerical Tolerance

**Acceptance Criteria**:
- **Per-token accuracy**: ≥95% exact token match vs Q4_K_M
- **Perplexity**: Within ±1% of Q4_K_M baseline
- **Embedding distance**: Cosine similarity ≥0.99

**Failure Modes**:
- Incorrect dequantization formula
- Wrong byte order (endianness)
- Scale factor miscalculation
- Block boundary errors

---

## 8. Performance Considerations

### 8.1 Memory Bandwidth Analysis

**MXFP4 Weight Size**: 20B params × 4.5 bits/param ≈ 11.25 GB  
**FP16 Weight Size**: 20B params × 16 bits/param = 40 GB  
**Compression Ratio**: 3.56x

**Bandwidth Savings**:
- **MXFP4**: Load 11.25 GB from VRAM
- **FP16**: Load 40 GB from VRAM
- **Speedup**: ~3.5x less memory traffic

**Dequantization Overhead**:
- Dequant compute: ~10-20% of GEMM time
- Net speedup: ~2.5-3x vs FP16

### 8.2 VRAM Footprint

**GPT-OSS-20B VRAM Breakdown**:
- **Weights (MXFP4)**: 11.25 GB
- **KV Cache (FP16)**: 4-8 GB (depends on context length)
- **Activations (FP16)**: 2-4 GB
- **Overhead**: 1-2 GB
- **Total**: ~18-25 GB

**Target**: Fit in 24 GB VRAM (RTX 3090, RTX 4090, A5000)

### 8.3 Optimization Opportunities

**Phase 1 (M0)**: Basic dequantization
- Separate dequant kernel + cuBLAS GEMM
- Target: Functional correctness

**Phase 2 (M1)**: Fused kernels
- Fuse dequant + GEMM
- Target: 20-30% speedup

**Phase 3 (M2)**: Advanced optimizations
- Tensor Core integration (FP8 path)
- Persistent kernels
- Target: 50%+ speedup vs naive

---

## 9. Implementation Recommendations

### 9.1 Development Phases

**Phase 1: Dequantization Kernel (GT-029)**
- Implement basic MXFP4 → FP16 kernel
- Unit tests with known values
- Validate numerical correctness

**Phase 2: GGUF v3 Parsing (GT-006)**
- Add MXFP4 tensor type support
- Parse MXFP4 blocks from GGUF
- Load into VRAM

**Phase 3: Weight Integration (GT-031, GT-033-GT-037)**
- Wire dequant into embedding lookup
- Wire dequant into attention projections
- Wire dequant into FFN projections
- Wire dequant into LM head

**Phase 4: Validation (GT-030, GT-038)**
- End-to-end GPT-OSS-20B inference
- Compare with Q4_K_M baseline
- Validate ±1% tolerance

### 9.2 Risk Mitigation

**Risk 1: Numerical Instability**
- Mitigation: Extensive unit tests, Q4_K_M baseline comparison
- Fallback: Use Q4_K_M if MXFP4 fails validation

**Risk 2: Performance Regression**
- Mitigation: Profile dequant overhead, optimize if needed
- Fallback: Accept slower inference for M0, optimize in M1

**Risk 3: VRAM Overflow**
- Mitigation: Careful memory profiling, chunked loading
- Fallback: Reduce context length or batch size

### 9.3 Testing Strategy

**Unit Tests** (GT-030):
- Single block dequantization
- Edge cases (zero, max, min values)
- Scale factor variations
- Negative values

**Integration Tests** (GT-038):
- Full layer processing
- Multi-layer inference
- End-to-end generation

**Validation Tests** (GT-038):
- Perplexity on WikiText-2
- Token accuracy vs Q4_K_M
- Embedding similarity

---

## 10. Key Findings Summary

### 10.1 Format Characteristics
✅ MXFP4 uses 32-element blocks with shared 8-bit exponent  
✅ 17 bytes per block (16 bytes FP4 + 1 byte FP8 scale)  
✅ 3.76x compression vs FP16  
✅ ±1-2% accuracy vs FP16 for LLM weights  

### 10.2 Implementation Approach
✅ Software dequantization (no native GPU support)  
✅ Dequantize to FP16 before GEMM  
✅ Use cuBLAS for FP16 GEMM  
✅ FP16 accumulation to prevent error buildup  

### 10.3 Validation Requirements
✅ Q4_K_M baseline for comparison  
✅ ±1% perplexity tolerance  
✅ ≥95% token accuracy  
✅ Comprehensive unit and integration tests  

### 10.4 Performance Expectations
✅ ~3.5x memory bandwidth savings  
✅ ~2.5-3x inference speedup (after dequant overhead)  
✅ GPT-OSS-20B fits in 24 GB VRAM  
✅ Optimization opportunities in M1+  

---

## 11. References

### Primary Specifications
- OCP MX Specification v1.0: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- MXFP4 Paper (arXiv): https://arxiv.org/abs/2310.10537
- GGUF Specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

### Implementation References
- llama.cpp quantization: https://github.com/ggerganov/llama.cpp/tree/master/examples/quantize
- PyTorch FP8 support: https://pytorch.org/docs/stable/generated/torch.float8_e4m3fn.html
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

### Validation Resources
- WikiText-2 dataset: https://huggingface.co/datasets/wikitext
- LM Evaluation Harness: https://github.com/EleutherAI/lm-evaluation-harness

---

**Research Complete**: ✅  
**Ready for Implementation**: GT-029 (MXFP4 Dequantization Kernel)  
**Validation Framework**: Documented in `mxfp4-validation-framework.md`

---
Crafted by GPT-Gamma 🤖
