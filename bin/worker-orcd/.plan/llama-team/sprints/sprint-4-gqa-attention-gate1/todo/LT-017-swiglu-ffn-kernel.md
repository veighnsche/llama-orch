# LT-017: SwiGLU FFN Kernel

**Team**: Llama-Beta  
**Sprint**: Sprint 4 - GQA Attention + Integration  
**Size**: M (2 days)  
**Days**: 48-49  
**Spec Ref**: M0-W-1214

---

## Story Description

Implement SwiGLU (Swish-Gated Linear Unit) feed-forward network CUDA kernel for Llama models. Apply gated activation function with SiLU (Swish) to enable more expressive FFN layers.

---

## Acceptance Criteria

- [ ] Implement SwiGLU FFN CUDA kernel
- [ ] Compute gate projection: gate = W_gate @ x
- [ ] Compute up projection: up = W_up @ x
- [ ] Apply SiLU activation: silu(gate) = gate * sigmoid(gate)
- [ ] Element-wise multiply: hidden = silu(gate) * up
- [ ] Compute down projection: output = W_down @ hidden
- [ ] Support variable FFN dimensions (e.g., 4864 for Qwen)
- [ ] Optimize for memory bandwidth (fused operations)
- [ ] Unit tests validate SwiGLU computation
- [ ] Unit tests validate SiLU activation
- [ ] Benchmark kernel performance (TFLOPS)
- [ ] Error handling for invalid dimensions
- [ ] Log kernel launch parameters at DEBUG level

---

## Dependencies

### Upstream (Blocks This Story)
- FT-016: cuBLAS GEMM Wrapper (needs matrix multiplication)
- FT-013: Device Memory RAII (needs VRAM allocation)

### Downstream (This Story Blocks)
- LT-024: Qwen Forward Pass (needs FFN)
- LT-031: Phi-3 Forward Pass (needs FFN)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/kernels/swiglu.cu` - SwiGLU CUDA kernel
- `bin/worker-orcd/cuda/kernels/swiglu.h` - SwiGLU interface
- `bin/worker-orcd/src/kernels/swiglu.rs` - Rust FFI wrapper

### Key Interfaces
```cpp
// SwiGLU FFN configuration
struct SwiGLUConfig {
    int batch_size;        // Batch size
    int seq_len;           // Sequence length
    int hidden_dim;        // Hidden dimension (e.g., 896)
    int ffn_dim;           // FFN intermediate dimension (e.g., 4864)
};

// SwiGLU FFN forward pass
void swiglu_ffn_forward(
    half* output,          // [batch, seq_len, hidden_dim]
    const half* input,     // [batch, seq_len, hidden_dim]
    const half* w_gate,    // [ffn_dim, hidden_dim]
    const half* w_up,      // [ffn_dim, hidden_dim]
    const half* w_down,    // [hidden_dim, ffn_dim]
    const SwiGLUConfig& config,
    cudaStream_t stream = nullptr
);
```

```rust
#[repr(C)]
pub struct SwiGLUConfig {
    pub batch_size: i32,
    pub seq_len: i32,
    pub hidden_dim: i32,
    pub ffn_dim: i32,
}

extern "C" {
    pub fn swiglu_ffn_forward(
        output: *mut f16,
        input: *const f16,
        w_gate: *const f16,
        w_up: *const f16,
        w_down: *const f16,
        config: *const SwiGLUConfig,
        stream: cudaStream_t,
    );
}
```

### Implementation Notes

**SwiGLU Algorithm**:
```
1. gate = W_gate @ x          (linear projection)
2. up = W_up @ x              (linear projection)
3. silu_gate = gate * sigmoid(gate)  (SiLU activation)
4. hidden = silu_gate * up    (element-wise multiply)
5. output = W_down @ hidden   (linear projection)
```

**SiLU (Swish) Activation**:
```
silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

**CUDA Implementation**:
```cuda
// SiLU activation kernel (fused with element-wise multiply)
__global__ void swiglu_activation_kernel(
    half* hidden,
    const half* gate,
    const half* up,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        
        // SiLU: g * sigmoid(g)
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        float silu_g = g * sigmoid_g;
        
        // Element-wise multiply
        hidden[idx] = __float2half(silu_g * u);
    }
}

// Wrapper function
void swiglu_ffn_forward(
    half* output,
    const half* input,
    const half* w_gate,
    const half* w_up,
    const half* w_down,
    const SwiGLUConfig& config,
    cudaStream_t stream
) {
    int tokens = config.batch_size * config.seq_len;
    
    // Allocate temporary buffers
    half* gate = allocate_temp(tokens * config.ffn_dim);
    half* up = allocate_temp(tokens * config.ffn_dim);
    half* hidden = allocate_temp(tokens * config.ffn_dim);
    
    // 1. Gate projection: gate = W_gate @ input
    cublas_gemm(gate, w_gate, input, tokens, config.ffn_dim, config.hidden_dim, stream);
    
    // 2. Up projection: up = W_up @ input
    cublas_gemm(up, w_up, input, tokens, config.ffn_dim, config.hidden_dim, stream);
    
    // 3. SwiGLU activation: hidden = silu(gate) * up
    int threads = 256;
    int blocks = (tokens * config.ffn_dim + threads - 1) / threads;
    swiglu_activation_kernel<<<blocks, threads, 0, stream>>>(hidden, gate, up, tokens * config.ffn_dim);
    
    // 4. Down projection: output = W_down @ hidden
    cublas_gemm(output, w_down, hidden, tokens, config.hidden_dim, config.ffn_dim, stream);
    
    // Free temporary buffers
    free_temp(gate);
    free_temp(up);
    free_temp(hidden);
}
```

**Optimization**:
- Fuse SiLU and element-wise multiply in single kernel
- Use cuBLAS for GEMM operations (optimized)
- Minimize temporary buffer allocations
- Use CUDA streams for async execution

---

## Testing Strategy

### Unit Tests
- Test SiLU activation (verify sigmoid computation)
- Test SwiGLU with simple matrices (2x2, 4x4)
- Test SwiGLU with Qwen dimensions (896 → 4864 → 896)
- Test element-wise multiply (silu_gate * up)
- Test numerical correctness (compare with CPU)
- Test with different batch sizes and sequence lengths

### Numerical Validation
- Compare against reference SwiGLU (PyTorch)
- Tolerance: ±0.05 (accumulation errors in GEMM)
- Test with random inputs (normal distribution)

### Performance Tests
- Benchmark TFLOPS (compare with theoretical peak)
- Measure memory bandwidth utilization
- Compare fused vs unfused activation

### Manual Verification
1. Apply SwiGLU to random input tensor
2. Verify output shape [batch, seq_len, hidden_dim]
3. Check logs show kernel launch parameters

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (6+ tests)
- [ ] Numerical validation passing (±0.05 tolerance)
- [ ] Performance benchmarks recorded
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.5 (Inference Kernels)
- SwiGLU Paper: https://arxiv.org/abs/2002.05202
- Llama FFN: https://github.com/facebookresearch/llama/blob/main/llama/model.py
- Related Stories: LT-024, LT-031

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋
