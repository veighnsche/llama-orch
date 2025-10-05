# Best of Both Worlds: llama.cpp + vLLM Architecture

**Date**: 2025-10-05  
**Purpose**: Combine the best patterns from llama.cpp and vLLM for our implementation  
**For**: GT-052 onwards (Weight Loading, KV Cache, Inference)

---

## Executive Summary

**llama.cpp** excels at: Data-driven model loading, architecture registries, GGUF parsing  
**vLLM** excels at: Memory management, PagedAttention KV cache, batch inference, weight iterators

**Our strategy**: Use llama.cpp's patterns for loading, vLLM's patterns for runtime.

---

## Comparison Matrix

| Feature | llama.cpp | vLLM | **Our Choice** |
|---------|-----------|------|----------------|
| **Architecture Detection** | Enum + registry | HuggingFace config | **llama.cpp** (GGUF-native) |
| **Metadata Keys** | `LLM_KV` registry | HF config parser | **llama.cpp** (data-driven) |
| **Tensor Names** | `LLM_TENSOR_NAMES` per-arch | HF weight mapping | **llama.cpp** (pattern-based) |
| **Weight Loading** | Direct mmap → GPU | Iterator pattern | **Hybrid** (mmap + iterators) |
| **KV Cache** | Contiguous blocks | PagedAttention blocks | **vLLM** (paged blocks) |
| **Batch Inference** | Single request | Multi-request batching | **vLLM** (scheduler) |
| **Quantization** | Per-tensor GGML types | Unified quant API | **llama.cpp** (GGUF-native) |
| **Memory Management** | Static allocation | Dynamic block pool | **vLLM** (block manager) |

---

## Part 1: Model Loading (GT-051, GT-052, GT-053)

### Use llama.cpp Patterns

**Why**: llama.cpp is designed for GGUF, has proven architecture registry, handles 20+ models

#### 1.1 Architecture Registry (from llama.cpp)

```cpp
// cuda/src/model/architecture_registry.h

enum class Architecture {
    QWEN2,
    LLAMA,
    GPT2,
    PHI3,
    UNKNOWN
};

struct ArchitectureConfig {
    std::string name;              // "qwen2", "llama", "gpt2"
    std::string metadata_prefix;   // "qwen2", "llama", "gpt2"
    
    // Feature defaults
    bool has_qkv_bias;             // Qwen2: true, Llama: false
    bool qkv_fused;                // GPT2: true, Qwen2/Llama: false
    bool has_rope;                 // Llama/Qwen2: true, GPT2: false
    NormType norm_type;            // RMSNorm vs LayerNorm
    ActivationType activation;     // SwiGLU vs GELU
    
    // Tensor name patterns
    std::vector<TensorMapping> tensor_map;
};

// Registry
static const std::map<std::string, ArchitectureConfig> ARCH_REGISTRY = {
    {"qwen2", {
        .name = "qwen2",
        .metadata_prefix = "qwen2",
        .has_qkv_bias = true,
        .qkv_fused = false,
        .has_rope = true,
        .norm_type = NormType::RMSNorm,
        .activation = ActivationType::SwiGLU,
        .tensor_map = {
            {"token_embd.weight", "embeddings", true, false},
            {"blk.{L}.attn_q.weight", "layers[{L}].attn_q", true, true},
            {"blk.{L}.attn_q.bias", "layers[{L}].attn_q_bias", true, true},
            // ...
        }
    }},
    // ... more architectures
};
```

**Pattern**: Enum + registry → no hardcoded if/else chains

#### 1.2 Metadata Key Mapper (from llama.cpp)

```cpp
// cuda/src/model/metadata_key_mapper.h

class MetadataKeyMapper {
public:
    MetadataKeyMapper(Architecture arch, const std::vector<GGUFMetadata>& metadata)
        : arch_(arch), metadata_(metadata) {
        config_ = &ARCH_REGISTRY.at(get_arch_name(arch));
    }
    
    // Get required key
    uint32_t get_uint32(const std::string& key_suffix) {
        std::string full_key = config_->metadata_prefix + "." + key_suffix;
        return gguf::get_required_uint32(metadata_, full_key);
    }
    
    // Get optional key with default
    uint32_t get_uint32(const std::string& key_suffix, uint32_t default_value) {
        std::string full_key = config_->metadata_prefix + "." + key_suffix;
        return gguf::get_optional_uint32(metadata_, full_key, default_value);
    }
    
    // Special: vocab_size from tokenizer
    uint32_t get_vocab_size() {
        return gguf::get_array_length(metadata_, "tokenizer.ggml.tokens");
    }
};
```

**Pattern**: Prefix + suffix → dynamic key construction

#### 1.3 Tensor Mapper (from llama.cpp)

```cpp
// cuda/src/model/tensor_mapper.h

struct TensorMapping {
    std::string gguf_pattern;      // "blk.{L}.attn_q.weight"
    std::string internal_name;     // "layers[{L}].attn_q"
    bool required;                 // true = error if missing
    bool per_layer;                // true = expand for each layer
};

class TensorMapper {
public:
    TensorMapper(Architecture arch, int num_layers)
        : arch_(arch), num_layers_(num_layers) {
        config_ = &ARCH_REGISTRY.at(get_arch_name(arch));
    }
    
    // Expand patterns for all layers
    std::vector<TensorMapping> get_expanded_mappings() {
        std::vector<TensorMapping> result;
        
        for (const auto& mapping : config_->tensor_map) {
            if (mapping.per_layer) {
                // Expand {L} for each layer
                for (int L = 0; L < num_layers_; ++L) {
                    result.push_back({
                        .gguf_pattern = replace(mapping.gguf_pattern, "{L}", std::to_string(L)),
                        .internal_name = replace(mapping.internal_name, "{L}", std::to_string(L)),
                        .required = mapping.required,
                        .per_layer = false
                    });
                }
            } else {
                result.push_back(mapping);
            }
        }
        
        return result;
    }
    
    // Detect features by probing tensors
    ModelFeatures detect_features(const std::vector<GGUFTensor>& tensors) {
        ModelFeatures features;
        
        // Probe for QKV bias
        features.has_qkv_bias = find_tensor(tensors, "blk.0.attn_q.bias") != nullptr;
        
        // Probe for fused QKV
        features.qkv_fused = find_tensor(tensors, "blk.0.attn_qkv.weight") != nullptr;
        
        // Use arch defaults for others
        features.has_rope = config_->has_rope;
        features.norm_type = config_->norm_type;
        features.activation = config_->activation;
        
        return features;
    }
};
```

**Pattern**: Pattern expansion + feature probing → dynamic tensor loading

---

### Incorporate vLLM Patterns

**Why**: vLLM has cleaner weight iteration and better error handling

#### 1.4 Weight Iterator (from vLLM)

```cpp
// cuda/src/model/weight_iterator.h

class WeightIterator {
public:
    virtual ~WeightIterator() = default;
    
    struct WeightEntry {
        std::string name;
        std::vector<uint64_t> shape;
        QuantType quant_type;
        const void* data;
        size_t size_bytes;
    };
    
    virtual bool has_next() = 0;
    virtual WeightEntry next() = 0;
};

class GGUFWeightIterator : public WeightIterator {
public:
    GGUFWeightIterator(const io::MmapFile& mmap, const gguf::GGUFHeader& header)
        : mmap_(mmap), tensors_(header.tensors), current_(0) {}
    
    bool has_next() override {
        return current_ < tensors_.size();
    }
    
    WeightEntry next() override {
        const auto& tensor = tensors_[current_++];
        return {
            .name = tensor.name,
            .shape = tensor.dimensions,
            .quant_type = map_ggml_type(tensor.type),
            .data = mmap_.get_tensor_data(tensor.offset, tensor.size),
            .size_bytes = tensor.size
        };
    }
};
```

**Pattern**: Iterator abstraction → cleaner loading loop

---

## Part 2: Runtime Execution (GT-054, GT-055, GT-056)

### Use vLLM Patterns

**Why**: vLLM's PagedAttention and block manager are superior for memory efficiency

#### 2.1 Paged KV Cache (from vLLM)

```cpp
// cuda/src/kv_cache/paged_kv_cache.h

struct KVBlock {
    half* k_data;  // [BLOCK_SIZE, num_kv_heads, head_dim]
    half* v_data;  // [BLOCK_SIZE, num_kv_heads, head_dim]
    int ref_count;
    int block_id;
};

class PagedKVCache {
public:
    PagedKVCache(int num_layers, int num_kv_heads, int head_dim, int block_size)
        : num_layers_(num_layers),
          num_kv_heads_(num_kv_heads),
          head_dim_(head_dim),
          block_size_(block_size) {
        
        // Allocate block pool
        allocate_block_pool();
    }
    
    // Allocate blocks for a sequence
    std::vector<int> allocate_blocks(int num_tokens) {
        int num_blocks = (num_tokens + block_size_ - 1) / block_size_;
        std::vector<int> block_ids;
        
        for (int i = 0; i < num_blocks; ++i) {
            if (free_blocks_.empty()) {
                throw std::runtime_error("Out of KV cache blocks");
            }
            int block_id = free_blocks_.back();
            free_blocks_.pop_back();
            block_ids.push_back(block_id);
            blocks_[block_id].ref_count++;
        }
        
        return block_ids;
    }
    
    // Free blocks when sequence is done
    void free_blocks(const std::vector<int>& block_ids) {
        for (int block_id : block_ids) {
            blocks_[block_id].ref_count--;
            if (blocks_[block_id].ref_count == 0) {
                free_blocks_.push_back(block_id);
            }
        }
    }
    
    // Get block pointers for attention kernel
    void get_block_tables(
        const std::vector<std::vector<int>>& block_ids_per_seq,
        int** block_tables_out
    ) {
        // Flatten block IDs into table for kernel
        for (size_t seq = 0; seq < block_ids_per_seq.size(); ++seq) {
            for (size_t i = 0; i < block_ids_per_seq[seq].size(); ++i) {
                block_tables_out[seq][i] = block_ids_per_seq[seq][i];
            }
        }
    }
    
private:
    int num_layers_;
    int num_kv_heads_;
    int head_dim_;
    int block_size_;
    
    std::vector<KVBlock> blocks_;
    std::vector<int> free_blocks_;
    
    void allocate_block_pool() {
        // Calculate max blocks from available VRAM
        size_t available_vram = get_available_vram();
        size_t block_size_bytes = block_size_ * num_kv_heads_ * head_dim_ * sizeof(half) * 2;  // K + V
        int max_blocks = available_vram / block_size_bytes / num_layers_;
        
        blocks_.resize(max_blocks);
        free_blocks_.reserve(max_blocks);
        
        for (int i = 0; i < max_blocks; ++i) {
            // Allocate K and V for this block
            cudaMalloc(&blocks_[i].k_data, block_size_bytes / 2);
            cudaMalloc(&blocks_[i].v_data, block_size_bytes / 2);
            blocks_[i].ref_count = 0;
            blocks_[i].block_id = i;
            free_blocks_.push_back(i);
        }
    }
};
```

**Pattern**: Block pool + ref counting → efficient memory reuse

#### 2.2 Attention Kernel with Block Tables (from vLLM)

```cpp
// cuda/kernels/paged_attention.cu

__global__ void paged_attention_kernel(
    const half* __restrict__ q,           // [num_seqs, num_heads, head_dim]
    const half* __restrict__ k_cache,     // Block pool
    const half* __restrict__ v_cache,     // Block pool
    half* __restrict__ output,            // [num_seqs, num_heads, head_dim]
    const int* __restrict__ block_tables, // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens, // [num_seqs]
    int num_seqs,
    int num_heads,
    int head_dim,
    int block_size,
    float scale
) {
    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.x;
    
    if (seq_idx >= num_seqs) return;
    
    int context_len = context_lens[seq_idx];
    int num_blocks = (context_len + block_size - 1) / block_size;
    
    // Load query
    half q_vec[HEAD_DIM];
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_vec[i] = q[seq_idx * num_heads * head_dim + head_idx * head_dim + i];
    }
    
    float max_logit = -INFINITY;
    float sum_exp = 0.0f;
    
    // Iterate over blocks
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        int physical_block_id = block_tables[seq_idx * MAX_BLOCKS + block_idx];
        
        // Iterate over tokens in block
        int tokens_in_block = min(block_size, context_len - block_idx * block_size);
        for (int token_idx = 0; token_idx < tokens_in_block; ++token_idx) {
            // Compute Q·K
            float qk = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                int k_offset = physical_block_id * block_size * head_dim + token_idx * head_dim + i;
                qk += __half2float(q_vec[i]) * __half2float(k_cache[k_offset]);
            }
            qk *= scale;
            
            // Update softmax stats
            max_logit = fmaxf(max_logit, qk);
            sum_exp += expf(qk - max_logit);
        }
    }
    
    // Compute attention output (simplified)
    // ... (full implementation in actual kernel)
}
```

**Pattern**: Block tables + physical addressing → memory-efficient attention

---

## Part 3: Combined Architecture

### Our Implementation Strategy

#### Phase 1: Loading (GT-051, GT-052, GT-053) - Use llama.cpp

1. **Architecture Registry** - Data-driven, no hardcoding
2. **Metadata Key Mapper** - Dynamic key construction
3. **Tensor Mapper** - Pattern expansion + feature probing
4. **Weight Iterator** - Clean iteration (vLLM-inspired)

#### Phase 2: Runtime (GT-054, GT-055, GT-056) - Use vLLM

1. **Paged KV Cache** - Block pool + ref counting
2. **Block Manager** - Dynamic allocation
3. **Paged Attention Kernel** - Block tables
4. **Batch Scheduler** - Multi-request batching (future)

---

## Updated Story Breakdown

### GT-051: ✅ COMPLETE
- Implemented basic config parsing
- **Refactor needed**: Add MetadataKeyMapper for cleaner code

### GT-052: GGUF Weight Loading (UPDATED)
**New approach**:
1. Implement ArchitectureRegistry
2. Implement TensorMapper with pattern expansion
3. Implement GGUFWeightIterator
4. Load weights using iterator + tensor map
5. Detect features by probing (qkv_bias, qkv_fused)

**Estimated time**: 8-10 hours (was 6-8, more complex now)

### GT-053: BPE Tokenizer (UNCHANGED)
- Use llama.cpp patterns
- Extract from GGUF metadata
- **Estimated time**: 5-7 hours

### GT-054: Transformer Execution (UPDATED)
**New approach**:
1. Implement PagedKVCache (vLLM pattern)
2. Update attention kernels to use block tables
3. Wire transformer layers with paged cache

**Estimated time**: 6-8 hours (was 4-6, paged cache adds complexity)

### GT-055: LM Head (UNCHANGED)
- cuBLAS GEMM
- **Estimated time**: 2-3 hours

### GT-056: Wire Inference (UPDATED)
**New approach**:
1. Integrate PagedKVCache
2. Wire tokenizer → model → sampling
3. Handle block allocation/deallocation

**Estimated time**: 3-4 hours (was 2-3, block management adds complexity)

### GT-057: Test Cleanup (UNCHANGED)
- **Estimated time**: 1-2 hours

### NEW: GT-058: Refactor GT-051 with MetadataKeyMapper
**New story**:
1. Extract hardcoded key construction
2. Implement MetadataKeyMapper
3. Update parse_config_from_gguf() to use mapper
4. Tests still pass

**Estimated time**: 2-3 hours

---

## Total Revised Estimate

| Story | Old Estimate | New Estimate | Change |
|-------|--------------|--------------|--------|
| GT-051 | 2-3h | ✅ DONE | - |
| GT-052 | 6-8h | 8-10h | +2h (registry) |
| GT-053 | 5-7h | 5-7h | No change |
| GT-054 | 4-6h | 6-8h | +2h (paged cache) |
| GT-055 | 2-3h | 2-3h | No change |
| GT-056 | 2-3h | 3-4h | +1h (block mgmt) |
| GT-057 | 1-2h | 1-2h | No change |
| GT-058 | - | 2-3h | New story |
| **Total** | **22-31h** | **28-38h** | **+6-7h** |

**Why more time?**
- Architecture registry adds complexity but eliminates future hardcoding
- Paged KV cache is more complex but much more efficient
- One-time investment for long-term maintainability

---

## Benefits of This Approach

### From llama.cpp
- ✅ **No hardcoding** - Add new models by adding config, not code
- ✅ **Data-driven** - Architecture registry is the source of truth
- ✅ **Proven patterns** - Works for 20+ architectures
- ✅ **GGUF-native** - Designed for our use case

### From vLLM
- ✅ **Memory efficient** - Paged blocks reduce fragmentation
- ✅ **Batch-ready** - Can add batching later without refactor
- ✅ **Production-proven** - Used by major LLM services
- ✅ **Scalable** - Block pool grows/shrinks dynamically

### Combined
- ✅ **Best of both** - llama.cpp for loading, vLLM for runtime
- ✅ **Future-proof** - Easy to add new models and features
- ✅ **Maintainable** - Clear separation of concerns
- ✅ **Testable** - Each component can be tested independently

---

## Recommendation

**Accept the additional complexity** for these reasons:

1. **One-time cost**: 6-7 extra hours now saves weeks later
2. **No hardcoding**: Adding Phi-3, GPT-OSS-20B, etc. becomes trivial
3. **Production-ready**: vLLM's patterns are battle-tested
4. **Maintainable**: Clear architecture, easy to debug
5. **Extensible**: Batching, prefix caching, etc. become easy additions

**Alternative** (if time is critical):
- Keep GT-052 simple (current approach)
- Add GT-059: Refactor to registries (post-M0)
- Trade: Ship faster now, refactor later

---

**Created by**: Project Management Team 📋  
**Date**: 2025-10-05  
**Sources**: llama.cpp repo analysis + vLLM documentation  
**Decision needed**: Accept complexity now or refactor later?

---
Analyzed by Testing Team 🔍
