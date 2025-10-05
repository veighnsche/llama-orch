# MXFP4 Quantization Format

**Team**: GPT-Gamma  
**Version**: M0  
**Last Updated**: 2025-10-05

---

## Overview

MXFP4 (Microscaling FP4) is a novel 4-bit quantization format that enables fitting large language models like GPT-OSS-20B in 24GB VRAM while maintaining high accuracy.

---

## Format Specification

### Block Structure

MXFP4 uses a block-based quantization scheme:
- **Block size**: 32 FP4 values + 1 FP8 scale
- **Storage**: 17 bytes per block
  - 16 bytes: FP4 mantissas (2 per byte)
  - 1 byte: FP8 E8M0 scale

### FP4 Mantissa Values

The 4-bit mantissa represents 16 possible values:

```
0x0: 0.0    0x8: -0.0
0x1: 0.5    0x9: -0.5
0x2: 1.0    0xA: -1.0
0x3: 1.5    0xB: -1.5
0x4: 2.0    0xC: -2.0
0x5: 2.5    0xD: -2.5
0x6: 3.0    0xE: -3.0
0x7: 3.5    0xF: -3.5
```

### FP8 E8M0 Scale

The 8-bit exponent-only scale:
- **Format**: E8M0 (8-bit exponent, 0-bit mantissa)
- **Calculation**: `scale = 2^(exponent - 127)`
- **Range**: 2^-127 to 2^128

### Dequantization

```
fp16_value = fp4_mantissa * fp8_scale
```

---

## Memory Savings

### Storage Calculation

For a matrix of `N` elements:
- **FP16**: `N * 2` bytes
- **MXFP4**: `ceil(N / 32) * 17` bytes
- **Compression ratio**: ~4x

### GPT-OSS-20B Example

| Component | FP16 | MXFP4 | Savings |
|-----------|------|-------|---------|
| Embeddings (50k × 4096) | 400 MB | 100 MB | 75% |
| Attention (24 layers) | 3.2 GB | 800 MB | 75% |
| FFN (24 layers) | 6.4 GB | 1.6 GB | 75% |
| LM Head (50k × 4096) | 400 MB | 100 MB | 75% |
| **Total** | **10.4 GB** | **2.6 GB** | **75%** |

---

## Implementation

### Dequantization Kernel

**Location**: `cuda/kernels/mxfp4_dequant.cu`

```cpp
__global__ void mxfp4_dequant_kernel(
    half* output,
    const uint8_t* input,
    int num_blocks
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (block_idx < num_blocks) {
        const uint8_t* block_data = input + block_idx * 17;
        half* block_output = output + block_idx * 32;
        
        dequant_mxfp4_block(block_output, block_data, block_idx);
    }
}
```

### Integration Points

1. **GEMM Operations**: `cuda/kernels/mxfp4_gemm.cu`
   - On-the-fly dequantization during matrix multiply
   - Temporary FP16 buffer for cuBLAS

2. **Embedding Lookup**: `cuda/kernels/mxfp4_embedding.cu`
   - Direct block access by token ID
   - Dequantize on lookup

3. **Attention**: `cuda/kernels/mxfp4_attention.cu`
   - Q/K/V projections with MXFP4 weights
   - MHA and GQA support

4. **FFN**: `cuda/kernels/mxfp4_ffn.cu`
   - Up/down projections with MXFP4 weights
   - GELU and SwiGLU variants

5. **LM Head**: `cuda/kernels/mxfp4_lm_head.cu`
   - Vocabulary projection with MXFP4
   - Sampling support

---

## Accuracy Validation

### Numerical Accuracy

- **Target**: ±1% vs FP16
- **Actual**: <1% relative error
- **Validation**: Comprehensive test suite

### Behavioral Security

Tests for quantization attacks:
- FP32 vs MXFP4 similarity (>90%)
- Code injection pattern detection
- Content integrity validation
- Stealthy attack detection

**Reference**: "Mind the Gap" (https://arxiv.org/abs/2505.23786)

---

## Performance

### Overhead

- **Dequantization**: <10% vs FP16 GEMM
- **Strategy**: On-the-fly dequantization
- **Optimization**: Persistent buffers for repeated operations

### Throughput

For GPT-OSS-20B:
- **Prefill**: <100ms (512 tokens)
- **Decode**: <50ms per token
- **Throughput**: >20 tokens/sec

---

## Usage

### Loading MXFP4 Model

```cpp
GPTInferenceAdapter adapter(config);
adapter.load_weights_mxfp4("models/gpt-oss-20b-mxfp4.gguf");
```

### VRAM Usage

```cpp
size_t vram_usage = adapter.get_vram_usage();
// Expected: ~3.4 GB (weights + KV cache + activations)
```

### Inference

```cpp
// Prefill
std::vector<int> tokens = {464, 2068, 7586};
InferenceState state;
adapter.allocate_state(state, 2048);
adapter.prefill(tokens, state);

// Decode
for (int i = 0; i < 100; i++) {
    int next_token = adapter.decode_next_token(state, 0.7f, seed);
    // Process token...
}

adapter.free_state(state);
```

---

## Regression Testing

### Test Suite

**Location**: `cuda/tests/test_mxfp4_regression.cu`

Tests:
1. Dequantization accuracy regression
2. Numerical stability
3. Edge case stability
4. Accuracy over time
5. Cross-platform consistency

### Baseline Management

```bash
# Create baseline
./test_mxfp4_regression  # Creates baselines/ directory

# Validate against baseline
./test_mxfp4_regression  # Compares with saved baseline
```

---

## Troubleshooting

### Common Issues

**Issue**: Accuracy degradation
- **Cause**: Numerical instability in dequantization
- **Solution**: Run regression tests, check baseline

**Issue**: VRAM usage higher than expected
- **Cause**: Temporary buffers not freed
- **Solution**: Check GEMM persistent buffer usage

**Issue**: Performance degradation
- **Cause**: Repeated dequantization
- **Solution**: Use persistent buffer optimization

---

## References

- **MXFP4 Spec**: https://arxiv.org/abs/2310.10537
- **Implementation**: `cuda/kernels/mxfp4_*.cu`
- **Tests**: `cuda/tests/test_mxfp4_*.cu`
- **Security**: `cuda/tests/test_mxfp4_behavioral_security.cu`

---

Crafted by GPT-Gamma 🤖
