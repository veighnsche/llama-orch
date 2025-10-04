# LT-014: Residual Connection Kernel

**Team**: Llama-Beta  
**Sprint**: Sprint 3 - UTF-8 Safety + Llama Kernels  
**Size**: S (1 day)  
**Days**: 41  
**Spec Ref**: M0-W-1214

---

## Story Description

Implement residual connection CUDA kernel for Llama transformer blocks. Add input tensor to output tensor element-wise to enable gradient flow and training stability in deep networks.

---

## Acceptance Criteria

- [ ] Implement residual connection CUDA kernel (element-wise add)
- [ ] Support in-place operation (output += input)
- [ ] Support out-of-place operation (output = x + residual)
- [ ] Handle variable tensor shapes (batch, seq_len, hidden_dim)
- [ ] Optimize for memory bandwidth (coalesced access)
- [ ] Unit tests validate residual addition
- [ ] Unit tests validate in-place vs out-of-place
- [ ] Benchmark kernel performance (GB/s)
- [ ] Error handling for shape mismatches
- [ ] Log kernel launch parameters at DEBUG level

---

## Dependencies

### Upstream (Blocks This Story)
- FT-010: CUDA Context Init (needs CUDA runtime)
- FT-013: Device Memory RAII (needs VRAM allocation)

### Downstream (This Story Blocks)
- LT-024: Qwen Forward Pass (needs residual connections)
- LT-031: Phi-3 Forward Pass (needs residual connections)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/kernels/residual.cu` - Residual CUDA kernel
- `bin/worker-orcd/cuda/kernels/residual.h` - Residual interface
- `bin/worker-orcd/src/kernels/residual.rs` - Rust FFI wrapper

### Key Interfaces
```cpp
// Residual connection configuration
struct ResidualConfig {
    int batch_size;        // Batch size
    int seq_len;           // Sequence length
    int hidden_dim;        // Hidden dimension
    bool in_place;         // In-place operation (output += input)
};

// Apply residual connection
void residual_forward(
    half* output,          // [batch, seq_len, hidden_dim]
    const half* input,     // [batch, seq_len, hidden_dim]
    const half* residual,  // [batch, seq_len, hidden_dim]
    const ResidualConfig& config,
    cudaStream_t stream = nullptr
);
```

```rust
#[repr(C)]
pub struct ResidualConfig {
    pub batch_size: i32,
    pub seq_len: i32,
    pub hidden_dim: i32,
    pub in_place: bool,
}

extern "C" {
    pub fn residual_forward(
        output: *mut f16,
        input: *const f16,
        residual: *const f16,
        config: *const ResidualConfig,
        stream: cudaStream_t,
    );
}
```

### Implementation Notes

**Residual Algorithm**:
```
output[i] = input[i] + residual[i]  (for all i)
```

**CUDA Kernel**:
```cuda
__global__ void residual_kernel(
    half* output,
    const half* input,
    const half* residual,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        output[idx] = __hadd(input[idx], residual[idx]);
    }
}

// Wrapper function
void residual_forward(
    half* output,
    const half* input,
    const half* residual,
    const ResidualConfig& config,
    cudaStream_t stream
) {
    int total_elements = config.batch_size * config.seq_len * config.hidden_dim;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    if (config.in_place) {
        // In-place: output += residual (input is ignored)
        residual_kernel<<<blocks, threads, 0, stream>>>(output, output, residual, total_elements);
    } else {
        // Out-of-place: output = input + residual
        residual_kernel<<<blocks, threads, 0, stream>>>(output, input, residual, total_elements);
    }
}
```

**Optimization**:
- Use vectorized loads/stores (half2 or float4)
- Coalesce memory access (warp-aligned)
- Minimize kernel launch overhead (fuse with other ops if possible)

**Vectorized Version**:
```cuda
__global__ void residual_kernel_vectorized(
    half2* output,
    const half2* input,
    const half2* residual,
    int total_elements_half2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements_half2) {
        output[idx] = __hadd2(input[idx], residual[idx]);
    }
}
```

---

## Testing Strategy

### Unit Tests
- Test residual addition (simple vectors)
- Test in-place operation (output += residual)
- Test out-of-place operation (output = input + residual)
- Test with different shapes (batch=1, batch=4, seq_len=1, seq_len=128)
- Test numerical correctness (compare with CPU)
- Test shape mismatch error handling

### Performance Tests
- Benchmark kernel throughput (GB/s)
- Compare vectorized vs non-vectorized
- Measure memory bandwidth utilization

### Manual Verification
1. Apply residual to random tensors
2. Verify output = input + residual (element-wise)
3. Check logs show kernel launch parameters

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (6+ tests)
- [ ] Performance benchmarks recorded
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.5 (Inference Kernels)
- Residual Networks: https://arxiv.org/abs/1512.03385
- Llama Architecture: https://github.com/facebookresearch/llama
- Related Stories: LT-024, LT-031

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋
