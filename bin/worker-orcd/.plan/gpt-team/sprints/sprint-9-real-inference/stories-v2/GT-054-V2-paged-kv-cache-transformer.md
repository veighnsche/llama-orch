# GT-054-V2: Paged KV Cache + Transformer Execution

**Story ID**: GT-054-V2  
**Title**: Implement Paged KV Cache and Wire Transformer Layers  
**Size**: M (Medium)  
**Estimate**: 6-8 hours  
**Priority**: P0 (Critical Path)  
**Dependencies**: GT-052 (Weight Loading), GT-053 (Tokenizer)  
**Blocks**: GT-056 (Wire Inference)

---

## User Story

**As a** inference engine  
**I want** a paged KV cache with block management  
**So that** I can efficiently manage memory and support batching in the future

---

## Context

Traditional KV cache allocates contiguous memory per sequence, leading to:
- ❌ Memory fragmentation
- ❌ Wasted memory for variable-length sequences
- ❌ Difficult to batch multiple requests

vLLM's PagedAttention solves this with:
- ✅ Block pool with ref counting
- ✅ Dynamic allocation/deallocation
- ✅ Efficient memory reuse
- ✅ Batch-ready infrastructure

This story implements vLLM's proven pattern.

---

## Current State

**No KV cache implementation exists yet.**

We have:
- ✅ CUDA kernels (RoPE, RMSNorm, GQA attention, SwiGLU)
- ✅ Model weights loaded to GPU
- ❌ No KV cache management
- ❌ No transformer layer wiring

---

## Acceptance Criteria

### Paged KV Cache
- [ ] `PagedKVCache` class with block pool
- [ ] `KVBlock` struct: `{k_data, v_data, ref_count, block_id}`
- [ ] `allocate_blocks(num_tokens)` returns block IDs
- [ ] `free_blocks(block_ids)` decrements ref count
- [ ] `get_block_tables()` flattens block IDs for kernel
- [ ] Block size configurable (default: 16 tokens)
- [ ] Automatic pool sizing based on available VRAM

### Block Manager
- [ ] `BlockManager` class for lifecycle management
- [ ] Track free blocks in a queue
- [ ] Track allocated blocks per sequence
- [ ] Handle out-of-memory gracefully
- [ ] Support block sharing (for prefix caching - future)

### Updated Attention Kernel
- [ ] `paged_gqa_attention_kernel()` uses block tables
- [ ] Takes `block_tables` and `context_lens` as input
- [ ] Computes physical block addresses
- [ ] Iterates over blocks instead of contiguous memory
- [ ] Maintains same output as contiguous version

### Transformer Layer Wiring
- [ ] `GPTTransformerLayer` class
- [ ] `forward()` method: input → output
- [ ] Wire: RMSNorm → GQA → Residual → RMSNorm → SwiGLU → Residual
- [ ] Handle QKV bias (Qwen2) vs no bias (Llama)
- [ ] Use paged KV cache for attention
- [ ] Support both prefill and decode phases

### Testing
- [ ] Unit test: Block allocation/deallocation
- [ ] Unit test: Block table construction
- [ ] Unit test: Paged attention matches contiguous
- [ ] Integration test: Full transformer layer forward pass
- [ ] Verify VRAM usage with `nvidia-smi`

---

## Technical Design

### 1. Paged KV Cache

**File**: `cuda/src/kv_cache/paged_kv_cache.h`

```cpp
#ifndef WORKER_KV_CACHE_PAGED_KV_CACHE_H
#define WORKER_KV_CACHE_PAGED_KV_CACHE_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <queue>

namespace worker {
namespace kv_cache {

struct KVBlock {
    half* k_data;      // [BLOCK_SIZE, num_kv_heads, head_dim]
    half* v_data;      // [BLOCK_SIZE, num_kv_heads, head_dim]
    int ref_count;
    int block_id;
};

class PagedKVCache {
public:
    PagedKVCache(
        int num_layers,
        int num_kv_heads,
        int head_dim,
        int block_size = 16
    );
    
    ~PagedKVCache();
    
    // Allocate blocks for a sequence
    std::vector<int> allocate_blocks(int num_tokens);
    
    // Free blocks when sequence is done
    void free_blocks(const std::vector<int>& block_ids);
    
    // Get block pointers for attention kernel
    void get_block_tables(
        const std::vector<std::vector<int>>& block_ids_per_seq,
        int** block_tables_out
    );
    
    // Get K/V cache pointers for a layer
    half* get_k_cache(int layer_idx);
    half* get_v_cache(int layer_idx);
    
    // Stats
    int get_num_free_blocks() const { return free_blocks_.size(); }
    int get_num_total_blocks() const { return blocks_.size(); }
    size_t get_total_vram_bytes() const { return total_vram_bytes_; }
    
private:
    int num_layers_;
    int num_kv_heads_;
    int head_dim_;
    int block_size_;
    
    std::vector<KVBlock> blocks_;
    std::queue<int> free_blocks_;
    size_t total_vram_bytes_;
    
    void allocate_block_pool();
    void free_block_pool();
};

} // namespace kv_cache
} // namespace worker

#endif
```

**File**: `cuda/src/kv_cache/paged_kv_cache.cpp`

```cpp
#include "paged_kv_cache.h"
#include "../cuda_error.h"
#include <cuda_runtime.h>

namespace worker {
namespace kv_cache {

PagedKVCache::PagedKVCache(
    int num_layers,
    int num_kv_heads,
    int head_dim,
    int block_size
) : num_layers_(num_layers),
    num_kv_heads_(num_kv_heads),
    head_dim_(head_dim),
    block_size_(block_size),
    total_vram_bytes_(0) {
    
    allocate_block_pool();
}

PagedKVCache::~PagedKVCache() {
    free_block_pool();
}

void PagedKVCache::allocate_block_pool() {
    // Calculate block size in bytes
    size_t block_size_bytes = block_size_ * num_kv_heads_ * head_dim_ * sizeof(half);
    
    // Get available VRAM
    size_t free_vram, total_vram;
    cudaMemGetInfo(&free_vram, &total_vram);
    
    // Use 50% of free VRAM for KV cache
    size_t available_for_kv = free_vram / 2;
    
    // Calculate max blocks
    size_t bytes_per_block_all_layers = block_size_bytes * 2 * num_layers_;  // K + V
    int max_blocks = available_for_kv / bytes_per_block_all_layers;
    
    // Minimum 100 blocks
    if (max_blocks < 100) {
        throw CudaError::model_load_failed(
            "Insufficient VRAM for KV cache (need at least " +
            std::to_string(bytes_per_block_all_layers * 100 / 1024 / 1024) + " MB)"
        );
    }
    
    fprintf(stderr, "Allocating KV cache: %d blocks, %.2f MB total\n",
            max_blocks,
            (max_blocks * bytes_per_block_all_layers) / 1024.0 / 1024.0);
    
    blocks_.resize(max_blocks);
    
    for (int i = 0; i < max_blocks; ++i) {
        // Allocate K and V for all layers
        cudaMalloc(&blocks_[i].k_data, block_size_bytes * num_layers_);
        cudaMalloc(&blocks_[i].v_data, block_size_bytes * num_layers_);
        
        blocks_[i].ref_count = 0;
        blocks_[i].block_id = i;
        free_blocks_.push(i);
        
        total_vram_bytes_ += bytes_per_block_all_layers;
    }
}

void PagedKVCache::free_block_pool() {
    for (auto& block : blocks_) {
        if (block.k_data) cudaFree(block.k_data);
        if (block.v_data) cudaFree(block.v_data);
    }
    blocks_.clear();
}

std::vector<int> PagedKVCache::allocate_blocks(int num_tokens) {
    int num_blocks = (num_tokens + block_size_ - 1) / block_size_;
    std::vector<int> block_ids;
    block_ids.reserve(num_blocks);
    
    for (int i = 0; i < num_blocks; ++i) {
        if (free_blocks_.empty()) {
            // Out of blocks - free what we allocated and throw
            free_blocks(block_ids);
            throw CudaError::inference_failed(
                "Out of KV cache blocks (requested " + std::to_string(num_blocks) +
                ", have " + std::to_string(get_num_free_blocks()) + " free)"
            );
        }
        
        int block_id = free_blocks_.front();
        free_blocks_.pop();
        
        blocks_[block_id].ref_count++;
        block_ids.push_back(block_id);
    }
    
    return block_ids;
}

void PagedKVCache::free_blocks(const std::vector<int>& block_ids) {
    for (int block_id : block_ids) {
        if (block_id < 0 || block_id >= (int)blocks_.size()) {
            continue;  // Invalid block ID
        }
        
        blocks_[block_id].ref_count--;
        
        if (blocks_[block_id].ref_count == 0) {
            free_blocks_.push(block_id);
        }
    }
}

void PagedKVCache::get_block_tables(
    const std::vector<std::vector<int>>& block_ids_per_seq,
    int** block_tables_out
) {
    for (size_t seq = 0; seq < block_ids_per_seq.size(); ++seq) {
        for (size_t i = 0; i < block_ids_per_seq[seq].size(); ++i) {
            block_tables_out[seq][i] = block_ids_per_seq[seq][i];
        }
    }
}

half* PagedKVCache::get_k_cache(int layer_idx) {
    if (blocks_.empty()) return nullptr;
    
    // Return base pointer for layer
    size_t block_size_bytes = block_size_ * num_kv_heads_ * head_dim_ * sizeof(half);
    return blocks_[0].k_data + (layer_idx * block_size_bytes / sizeof(half));
}

half* PagedKVCache::get_v_cache(int layer_idx) {
    if (blocks_.empty()) return nullptr;
    
    // Return base pointer for layer
    size_t block_size_bytes = block_size_ * num_kv_heads_ * head_dim_ * sizeof(half);
    return blocks_[0].v_data + (layer_idx * block_size_bytes / sizeof(half));
}

} // namespace kv_cache
} // namespace worker
```

### 2. Updated Paged GQA Attention Kernel

**File**: `cuda/kernels/paged_gqa_attention.cu`

```cuda
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

__global__ void paged_gqa_attention_kernel(
    const half* __restrict__ q,           // [num_seqs, num_heads, head_dim]
    const half* __restrict__ k_cache,     // Block pool (all blocks)
    const half* __restrict__ v_cache,     // Block pool (all blocks)
    half* __restrict__ output,            // [num_seqs, num_heads, head_dim]
    const int* __restrict__ block_tables, // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens, // [num_seqs]
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    float scale
) {
    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.x;
    
    if (seq_idx >= num_seqs || head_idx >= num_heads) return;
    
    int context_len = context_lens[seq_idx];
    int num_blocks = (context_len + block_size - 1) / block_size;
    
    // GQA: map query head to KV head
    int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    
    // Load query
    extern __shared__ half shared_mem[];
    half* q_shared = shared_mem;
    
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_shared[i] = q[seq_idx * num_heads * head_dim + head_idx * head_dim + i];
    }
    __syncthreads();
    
    // Compute attention scores
    float max_logit = -INFINITY;
    float sum_exp = 0.0f;
    
    // First pass: compute max and sum for softmax
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        int physical_block_id = block_tables[seq_idx * max_blocks_per_seq + block_idx];
        
        int tokens_in_block = min(block_size, context_len - block_idx * block_size);
        
        for (int token_idx = 0; token_idx < tokens_in_block; ++token_idx) {
            // Compute Q·K
            float qk = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                int k_offset = physical_block_id * block_size * num_kv_heads * head_dim +
                               token_idx * num_kv_heads * head_dim +
                               kv_head_idx * head_dim + i;
                qk += __half2float(q_shared[i]) * __half2float(k_cache[k_offset]);
            }
            qk *= scale;
            
            max_logit = fmaxf(max_logit, qk);
        }
    }
    
    // Compute sum of exp(qk - max)
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        int physical_block_id = block_tables[seq_idx * max_blocks_per_seq + block_idx];
        
        int tokens_in_block = min(block_size, context_len - block_idx * block_size);
        
        for (int token_idx = 0; token_idx < tokens_in_block; ++token_idx) {
            float qk = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                int k_offset = physical_block_id * block_size * num_kv_heads * head_dim +
                               token_idx * num_kv_heads * head_dim +
                               kv_head_idx * head_dim + i;
                qk += __half2float(q_shared[i]) * __half2float(k_cache[k_offset]);
            }
            qk *= scale;
            
            sum_exp += expf(qk - max_logit);
        }
    }
    
    // Second pass: compute weighted sum of values
    float out_vec[128];  // Assume head_dim <= 128
    for (int i = 0; i < head_dim; ++i) {
        out_vec[i] = 0.0f;
    }
    
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        int physical_block_id = block_tables[seq_idx * max_blocks_per_seq + block_idx];
        
        int tokens_in_block = min(block_size, context_len - block_idx * block_size);
        
        for (int token_idx = 0; token_idx < tokens_in_block; ++token_idx) {
            // Recompute attention weight
            float qk = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                int k_offset = physical_block_id * block_size * num_kv_heads * head_dim +
                               token_idx * num_kv_heads * head_dim +
                               kv_head_idx * head_dim + i;
                qk += __half2float(q_shared[i]) * __half2float(k_cache[k_offset]);
            }
            qk *= scale;
            
            float attn_weight = expf(qk - max_logit) / sum_exp;
            
            // Accumulate V * attn_weight
            for (int i = 0; i < head_dim; ++i) {
                int v_offset = physical_block_id * block_size * num_kv_heads * head_dim +
                               token_idx * num_kv_heads * head_dim +
                               kv_head_idx * head_dim + i;
                out_vec[i] += attn_weight * __half2float(v_cache[v_offset]);
            }
        }
    }
    
    // Write output
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        output[seq_idx * num_heads * head_dim + head_idx * head_dim + i] = __float2half(out_vec[i]);
    }
}

// Host wrapper
void launch_paged_gqa_attention(
    const half* q,
    const half* k_cache,
    const half* v_cache,
    half* output,
    const int* block_tables,
    const int* context_lens,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    cudaStream_t stream
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    dim3 grid(num_heads, num_seqs);
    dim3 block(256);
    
    size_t shared_mem_size = head_dim * sizeof(half);
    
    paged_gqa_attention_kernel<<<grid, block, shared_mem_size, stream>>>(
        q, k_cache, v_cache, output,
        block_tables, context_lens,
        num_seqs, num_heads, num_kv_heads, head_dim,
        block_size, max_blocks_per_seq, scale
    );
}
```

### 3. Transformer Layer

**File**: `cuda/src/model/gpt_transformer_layer.h`

```cpp
#ifndef WORKER_MODEL_GPT_TRANSFORMER_LAYER_H
#define WORKER_MODEL_GPT_TRANSFORMER_LAYER_H

#include "gpt_weights.h"
#include "../kv_cache/paged_kv_cache.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace worker {
namespace model {

class GPTTransformerLayer {
public:
    GPTTransformerLayer(
        const GPTLayerWeights* weights,
        const GPTConfig& config,
        kv_cache::PagedKVCache* kv_cache,
        int layer_idx,
        cudaStream_t stream
    );
    
    // Forward pass
    void forward(
        const half* input,           // [batch_size, seq_len, hidden_dim]
        half* output,                // [batch_size, seq_len, hidden_dim]
        const int* block_tables,     // [batch_size, max_blocks]
        const int* context_lens,     // [batch_size]
        int batch_size,
        int seq_len,
        bool is_prefill
    );
    
private:
    const GPTLayerWeights* weights_;
    const GPTConfig& config_;
    kv_cache::PagedKVCache* kv_cache_;
    int layer_idx_;
    cudaStream_t stream_;
    
    // Intermediate buffers
    half* attn_norm_out_;
    half* q_;
    half* k_;
    half* v_;
    half* attn_out_;
    half* ffn_norm_out_;
    half* ffn_out_;
    
    void allocate_buffers(int max_batch_size, int max_seq_len);
    void free_buffers();
};

} // namespace model
} // namespace worker

#endif
```

---

## Implementation Steps

1. **Implement PagedKVCache** (3 hours)
   - Block pool allocation
   - Block allocation/deallocation
   - Block table construction
   - Write unit tests

2. **Update Paged GQA Attention Kernel** (2 hours)
   - Add block table support
   - Physical address computation
   - Test against contiguous version

3. **Implement GPTTransformerLayer** (2 hours)
   - Wire all components
   - Handle QKV bias
   - Prefill vs decode logic

4. **Integration Testing** (1 hour)
   - Full layer forward pass
   - Verify output correctness
   - Check VRAM usage

---

## Definition of Done

- [x] PagedKVCache implemented and tested
- [x] Paged GQA attention kernel working
- [x] GPTTransformerLayer wired correctly
- [x] Unit tests pass
- [x] Integration test passes
- [x] VRAM usage verified
- [x] Code reviewed and approved

---

## Time Estimate

**Optimistic**: 6 hours  
**Realistic**: 6-8 hours  
**Pessimistic**: 10 hours (if kernel debugging is complex)

---

**Created by**: Project Management Team 📋  
**Assigned to**: GPT-Gamma 🤖  
**Status**: TODO  
**Priority**: P0 (Critical Path)

---
Test opportunities identified by Testing Team 🔍
