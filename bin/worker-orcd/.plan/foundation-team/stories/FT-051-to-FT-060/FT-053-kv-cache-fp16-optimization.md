# FT-053: KV Cache FP16 Optimization

**Team**: Foundation-Alpha  
**Sprint**: Sprint 6 - FP16 Optimization (Post-M0)  
**Size**: S (1 day)  
**Priority**: Medium (Post-M0)  
**Spec Ref**: M0-W-1214, M0-W-1430

---

## ⚠️ Prerequisites

**Requires M0 completion:**
- Working FP32 KV cache in `cuda/kernels/kv_cache.cu`
- Functional cache read/write operations
- Integration with attention kernels

---

## Story Description

Optimize KV cache storage to use FP16 instead of FP32, reducing VRAM usage by 50%. This is critical for long-context generation where KV cache dominates memory consumption.

---

## Acceptance Criteria

- [ ] KV cache stored in FP16 format
- [ ] Automatic conversion FP32→FP16 on cache write
- [ ] Automatic conversion FP16→FP32 on cache read (if needed)
- [ ] Cache size calculation updated for FP16
- [ ] VRAM usage reduced by ~50% for cache
- [ ] Unit tests for cache operations
- [ ] Numerical accuracy validation
- [ ] Integration with attention kernels (FT-052)

---

## Dependencies

**Upstream**: FT-052 (FP16 attention kernels, Day 4)  
**Downstream**: FT-054 (Memory bandwidth profiling)

---

## Technical Details

### Current State

`cuda/kernels/kv_cache.cu` stores cache in FP32. Each token requires:
- K cache: `num_kv_heads * head_dim * 4 bytes`
- V cache: `num_kv_heads * head_dim * 4 bytes`

For Qwen2.5-0.5B (2 KV heads, 64 head_dim):
- Per token: 2 * 64 * 4 * 2 = 1024 bytes
- 512 tokens: 512 KB
- 2048 tokens: 2 MB

With FP16: **50% reduction** → 256 KB / 1 MB

### Implementation Plan

**Phase 1: FP16 Cache Storage** (Day 1)

```cpp
// cuda/kernels/kv_cache.cu

/**
 * Write K, V to cache (FP16 storage)
 * 
 * @param kv_cache_k Cache for keys [batch, max_seq_len, num_kv_heads, head_dim] (FP16)
 * @param kv_cache_v Cache for values [batch, max_seq_len, num_kv_heads, head_dim] (FP16)
 * @param k Current keys [batch, seq_len, num_kv_heads, head_dim] (FP16)
 * @param v Current values [batch, seq_len, num_kv_heads, head_dim] (FP16)
 * @param batch_size Batch size
 * @param seq_len Sequence length to write
 * @param cache_offset Offset in cache (where to start writing)
 * @param num_kv_heads Number of KV heads
 * @param head_dim Head dimension
 */
__global__ void kv_cache_write_fp16(
    half* kv_cache_k,
    half* kv_cache_v,
    const half* k,
    const half* v,
    int batch_size,
    int seq_len,
    int cache_offset,
    int max_seq_len,
    int num_kv_heads,
    int head_dim
) {
    int batch = blockIdx.z;
    int pos = blockIdx.y;
    int head = blockIdx.x;
    int dim = threadIdx.x;
    
    if (batch >= batch_size || pos >= seq_len || 
        head >= num_kv_heads || dim >= head_dim) {
        return;
    }
    
    // Source index in k/v
    int src_idx = batch * seq_len * num_kv_heads * head_dim +
                  pos * num_kv_heads * head_dim +
                  head * head_dim + dim;
    
    // Destination index in cache
    int cache_pos = cache_offset + pos;
    if (cache_pos >= max_seq_len) {
        return;  // Cache overflow
    }
    
    int dst_idx = batch * max_seq_len * num_kv_heads * head_dim +
                  cache_pos * num_kv_heads * head_dim +
                  head * head_dim + dim;
    
    // Direct copy (already FP16)
    kv_cache_k[dst_idx] = k[src_idx];
    kv_cache_v[dst_idx] = v[src_idx];
}

/**
 * Read from KV cache (FP16)
 * 
 * @param k_out Output keys [batch, cache_len, num_kv_heads, head_dim] (FP16)
 * @param v_out Output values [batch, cache_len, num_kv_heads, head_dim] (FP16)
 * @param kv_cache_k Cache for keys (FP16)
 * @param kv_cache_v Cache for values (FP16)
 * @param batch_size Batch size
 * @param cache_len Length to read
 * @param num_kv_heads Number of KV heads
 * @param head_dim Head dimension
 */
__global__ void kv_cache_read_fp16(
    half* k_out,
    half* v_out,
    const half* kv_cache_k,
    const half* kv_cache_v,
    int batch_size,
    int cache_len,
    int max_seq_len,
    int num_kv_heads,
    int head_dim
) {
    int batch = blockIdx.z;
    int pos = blockIdx.y;
    int head = blockIdx.x;
    int dim = threadIdx.x;
    
    if (batch >= batch_size || pos >= cache_len || 
        head >= num_kv_heads || dim >= head_dim) {
        return;
    }
    
    // Source index in cache
    int src_idx = batch * max_seq_len * num_kv_heads * head_dim +
                  pos * num_kv_heads * head_dim +
                  head * head_dim + dim;
    
    // Destination index in output
    int dst_idx = batch * cache_len * num_kv_heads * head_dim +
                  pos * num_kv_heads * head_dim +
                  head * head_dim + dim;
    
    // Direct copy (already FP16)
    k_out[dst_idx] = kv_cache_k[src_idx];
    v_out[dst_idx] = kv_cache_v[src_idx];
}

extern "C" {

/**
 * Initialize KV cache (FP16)
 * 
 * Allocates VRAM for KV cache with FP16 storage.
 * 
 * @param batch_size Batch size
 * @param max_seq_len Maximum sequence length
 * @param num_kv_heads Number of KV heads
 * @param head_dim Head dimension
 * @param cache_k_out Output pointer for K cache
 * @param cache_v_out Output pointer for V cache
 * @return 0 on success, error code on failure
 */
int cuda_kv_cache_init_fp16(
    int batch_size,
    int max_seq_len,
    int num_kv_heads,
    int head_dim,
    half** cache_k_out,
    half** cache_v_out
) {
    // Validate dimensions
    if (batch_size <= 0 || max_seq_len <= 0 || 
        num_kv_heads <= 0 || head_dim <= 0) {
        fprintf(stderr, "KV Cache: Invalid dimensions\n");
        return -1;
    }
    
    // Calculate cache size (FP16)
    size_t cache_size = (size_t)batch_size * max_seq_len * 
                        num_kv_heads * head_dim * sizeof(half);
    
    // Check for overflow
    if (cache_size > (size_t)INT_MAX * sizeof(half)) {
        fprintf(stderr, "KV Cache: Size overflow\n");
        return -1;
    }
    
    // Allocate K cache
    cudaError_t err = cudaMalloc(cache_k_out, cache_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "KV Cache: Failed to allocate K cache: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    
    // Allocate V cache
    err = cudaMalloc(cache_v_out, cache_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "KV Cache: Failed to allocate V cache: %s\n",
                cudaGetErrorString(err));
        cudaFree(*cache_k_out);
        return -1;
    }
    
    // Zero-initialize caches
    cudaMemset(*cache_k_out, 0, cache_size);
    cudaMemset(*cache_v_out, 0, cache_size);
    
    return 0;
}

/**
 * Append to KV cache (FP16)
 */
int cuda_kv_cache_append_fp16(
    half* kv_cache_k,
    half* kv_cache_v,
    const half* k,
    const half* v,
    int batch_size,
    int seq_len,
    int cache_offset,
    int max_seq_len,
    int num_kv_heads,
    int head_dim
) {
    // Validate
    if (cache_offset + seq_len > max_seq_len) {
        fprintf(stderr, "KV Cache: Overflow (offset=%d, seq_len=%d, max=%d)\n",
                cache_offset, seq_len, max_seq_len);
        return -1;
    }
    
    // Launch write kernel
    dim3 grid(num_kv_heads, seq_len, batch_size);
    dim3 block(head_dim);
    
    kv_cache_write_fp16<<<grid, block>>>(
        kv_cache_k, kv_cache_v, k, v,
        batch_size, seq_len, cache_offset, max_seq_len,
        num_kv_heads, head_dim
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "KV Cache write failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

/**
 * Free KV cache
 */
void cuda_kv_cache_free(half* cache_k, half* cache_v) {
    if (cache_k != nullptr) {
        cudaFree(cache_k);
    }
    if (cache_v != nullptr) {
        cudaFree(cache_v);
    }
}

} // extern "C"
```

**Phase 2: VRAM Usage Calculation** (Day 1)

Update VRAM calculation in model configs:

```rust
// src/models/qwen.rs

impl QwenModel {
    pub fn calculate_kv_cache_vram(
        config: &QwenConfig,
        max_seq_len: usize,
        batch_size: usize,
        use_fp16: bool,
    ) -> usize {
        let element_size = if use_fp16 { 2 } else { 4 };
        
        // K cache + V cache
        let cache_size = 2 * batch_size * max_seq_len * 
                         config.num_kv_heads * config.head_dim * element_size;
        
        cache_size
    }
}
```

**Phase 3: Integration with Attention** (Day 1)

Update attention kernels to use FP16 cache directly:

```cpp
// In cuda_gqa_attention_decode_fp16:
// Cache is already FP16, no conversion needed
// Just read from cache and use in GEMM
```

---

## Files to Create/Modify

**Create**:
- `cuda/tests/test_kv_cache_fp16.cu` - FP16 cache unit tests

**Modify**:
- `cuda/kernels/kv_cache.cu` - Add FP16 cache operations
- `cuda/include/kv_cache.h` - Add FP16 function declarations
- `src/models/qwen.rs` - Update VRAM calculation
- `src/models/phi3.rs` - Update VRAM calculation

---

## Testing Strategy

### Unit Tests (5 tests)

1. **test_kv_cache_init_fp16** - Allocation and initialization
2. **test_kv_cache_write_fp16** - Write to cache
3. **test_kv_cache_read_fp16** - Read from cache
4. **test_kv_cache_append_fp16** - Incremental append
5. **test_kv_cache_overflow** - Validate overflow handling

### Integration Tests (2 tests)

1. **test_attention_with_fp16_cache** - Full attention decode
2. **test_long_generation_fp16_cache** - 512+ token generation

---

## VRAM Savings

### Qwen2.5-0.5B (2 KV heads, 64 head_dim)

| Sequence Length | FP32 Cache | FP16 Cache | Savings |
|-----------------|------------|------------|---------|
| 512 tokens | 512 KB | 256 KB | 256 KB |
| 2048 tokens | 2 MB | 1 MB | 1 MB |
| 8192 tokens | 8 MB | 4 MB | 4 MB |

### Phi-3 Mini (32 KV heads, 96 head_dim)

| Sequence Length | FP32 Cache | FP16 Cache | Savings |
|-----------------|------------|------------|---------|
| 512 tokens | 12 MB | 6 MB | 6 MB |
| 2048 tokens | 48 MB | 24 MB | 24 MB |
| 8192 tokens | 192 MB | 96 MB | 96 MB |

**Impact**: For long-context generation (8K+ tokens), FP16 cache saves 100+ MB VRAM, enabling larger batch sizes or longer contexts.

---

## Numerical Accuracy

KV cache stores attention keys/values, which are already normalized. FP16 precision is sufficient:
- No accumulation (unlike GEMM)
- Values bounded by softmax normalization
- Tolerance: < 1e-2 difference in final output

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Unit tests passing (5 tests)
- [ ] Integration tests passing (2 tests)
- [ ] VRAM savings validated
- [ ] Numerical accuracy validated
- [ ] Documentation updated
- [ ] Code reviewed
- [ ] Story marked complete

---

**Status**: 📋 Ready  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-05

---

## References

- KV cache optimization: https://arxiv.org/abs/2211.05102
- Memory-efficient transformers: https://arxiv.org/abs/2112.05682

---
Built by Foundation-Alpha 🏗️
