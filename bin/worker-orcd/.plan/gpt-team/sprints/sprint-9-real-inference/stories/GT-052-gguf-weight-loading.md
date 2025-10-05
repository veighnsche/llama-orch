# GT-052: GGUF Weight Loading to GPU

**Team**: GPT-Gamma 🤖  
**Sprint**: Sprint 9 - Real Inference  
**Size**: L (6-8 hours)  
**Priority**: P0 (M0 blocker)  
**Spec Ref**: M0-W-1220, M0-W-1221, M0-W-1222

---

## Story Description

Load model weights from GGUF file to GPU VRAM using memory-mapped I/O and chunked transfers. This is the core of real model loading.

---

## Current State (STUB)

**File**: `cuda/src/model/gpt_weights.cpp` line 251

```cpp
std::unique_ptr<GPTModelWeights> GPTWeightLoader::load_from_gguf(const std::string& path) {
    // Parse config from GGUF
    GPTConfig config = parse_config_from_gguf(path);
    
    // Create model weights
    auto model = std::make_unique<GPTModelWeights>();
    model->config = config;
    
    // TODO: Parse GGUF tensors
    std::vector<GGUFTensorInfo> tensors;
    
    // TODO: Load embeddings
    // load_embeddings(model.get(), path, tensors);
    
    // TODO: Load transformer layers
    model->layers.reserve(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        auto layer = std::make_unique<GPTLayerWeights>();
        // load_layer(layer.get(), i, path, tensors, config);  // TODO
        model->layers.push_back(std::move(layer));
    }
    
    // TODO: Load output head
    // load_output_head(model.get(), path, tensors);
    
    // Calculate total VRAM usage
    model->total_vram_bytes = calculate_vram_usage(config);
    
    return model;
}
```

**Problem**: Creates structure but doesn't actually load any weights to GPU.

---

## Acceptance Criteria

- [ ] Parse GGUF tensors from header
- [ ] Implement `load_embeddings()` - allocate GPU memory and copy
- [ ] Implement `load_layer()` - load all layer weights (12 tensors per layer)
- [ ] Implement `load_output_head()` - load final norm and LM head
- [ ] Implement `allocate_and_copy()` - helper to copy tensors to GPU
- [ ] Use memory-mapped I/O (no full file read into RAM)
- [ ] Use chunked VRAM transfer (1MB chunks)
- [ ] Verify VRAM allocation with `nvidia-smi`
- [ ] Track actual VRAM usage (not just estimate)
- [ ] Handle allocation failures gracefully
- [ ] Works for Qwen2.5-0.5B-Instruct (~400MB VRAM)
- [ ] Unit test verifies weights loaded to GPU

---

## Technical Details

### Implementation Plan

#### 1. Parse GGUF Tensors

```cpp
std::unique_ptr<GPTModelWeights> GPTWeightLoader::load_from_gguf(const std::string& path) {
    // Parse config
    GPTConfig config = parse_config_from_gguf(path);
    
    // Memory-map file
    auto mmap = io::MmapFile::open(path);
    auto header = gguf::parse_gguf_header(mmap.data(), mmap.size());
    
    // Create model
    auto model = std::make_unique<GPTModelWeights>();
    model->config = config;
    
    // Load weights
    load_embeddings(model.get(), mmap, header.tensors);
    
    model->layers.reserve(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        auto layer = std::make_unique<GPTLayerWeights>();
        load_layer(layer.get(), i, mmap, header.tensors, config);
        model->layers.push_back(std::move(layer));
    }
    
    load_output_head(model.get(), mmap, header.tensors);
    
    // Calculate actual VRAM usage
    model->total_vram_bytes = calculate_actual_vram_usage(model.get());
    
    return model;
}
```

#### 2. Implement load_embeddings()

```cpp
void GPTWeightLoader::load_embeddings(
    GPTModelWeights* model,
    const io::MmapFile& mmap,
    const std::vector<gguf::GGUFTensor>& tensors
) {
    // Find token embeddings tensor
    const gguf::GGUFTensor* token_emb = find_tensor(tensors, "token_embd.weight");
    if (!token_emb) {
        throw CudaError::model_load_failed("Missing token_embd.weight");
    }
    
    // Validate shape [hidden_dim, vocab_size]
    validate_tensor_shape(*token_emb, {model->config.hidden_dim, model->config.vocab_size});
    
    // Allocate and copy to GPU
    model->token_embeddings = allocate_and_copy(
        mmap,
        token_emb->offset + header.data_start,
        token_emb->size
    );
    
    // Find position embeddings (optional for some architectures)
    const gguf::GGUFTensor* pos_emb = find_tensor(tensors, "position_embd.weight");
    if (pos_emb) {
        validate_tensor_shape(*pos_emb, {model->config.hidden_dim, model->config.max_seq_len});
        model->position_embeddings = allocate_and_copy(
            mmap,
            pos_emb->offset + header.data_start,
            pos_emb->size
        );
    }
}
```

#### 3. Implement load_layer()

```cpp
void GPTWeightLoader::load_layer(
    GPTLayerWeights* layer,
    int layer_idx,
    const io::MmapFile& mmap,
    const std::vector<gguf::GGUFTensor>& tensors,
    const GPTConfig& config
) {
    std::string prefix = "blk." + std::to_string(layer_idx) + ".";
    
    // Attention norm
    layer->attn_norm_weight = load_tensor(mmap, tensors, prefix + "attn_norm.weight");
    layer->attn_norm_bias = load_tensor(mmap, tensors, prefix + "attn_norm.bias");
    
    // Attention QKV
    layer->attn_qkv_weight = load_tensor(mmap, tensors, prefix + "attn_qkv.weight");
    layer->attn_qkv_bias = load_tensor(mmap, tensors, prefix + "attn_qkv.bias");
    
    // Attention output
    layer->attn_out_weight = load_tensor(mmap, tensors, prefix + "attn_output.weight");
    layer->attn_out_bias = load_tensor(mmap, tensors, prefix + "attn_output.bias");
    
    // FFN norm
    layer->ffn_norm_weight = load_tensor(mmap, tensors, prefix + "ffn_norm.weight");
    layer->ffn_norm_bias = load_tensor(mmap, tensors, prefix + "ffn_norm.bias");
    
    // FFN weights
    layer->ffn_up_weight = load_tensor(mmap, tensors, prefix + "ffn_up.weight");
    layer->ffn_up_bias = load_tensor(mmap, tensors, prefix + "ffn_up.bias");
    layer->ffn_down_weight = load_tensor(mmap, tensors, prefix + "ffn_down.weight");
    layer->ffn_down_bias = load_tensor(mmap, tensors, prefix + "ffn_down.bias");
    
    // Track VRAM usage
    layer->total_vram_bytes = calculate_layer_vram(layer);
}
```

#### 4. Implement allocate_and_copy()

```cpp
void* GPTWeightLoader::allocate_and_copy(
    const io::MmapFile& mmap,
    size_t file_offset,
    size_t size_bytes
) {
    // Allocate GPU memory
    void* device_ptr = nullptr;
    cudaError_t err = cudaMalloc(&device_ptr, size_bytes);
    if (err != cudaSuccess) {
        throw CudaError::out_of_memory(
            "Failed to allocate " + std::to_string(size_bytes) + " bytes for tensor"
        );
    }
    
    // Copy in chunks (M0-W-1222 requirement)
    const size_t CHUNK_SIZE = 1024 * 1024;  // 1MB
    const uint8_t* host_ptr = static_cast<const uint8_t*>(mmap.data()) + file_offset;
    
    for (size_t offset = 0; offset < size_bytes; offset += CHUNK_SIZE) {
        size_t chunk_size = std::min(CHUNK_SIZE, size_bytes - offset);
        
        err = cudaMemcpy(
            static_cast<uint8_t*>(device_ptr) + offset,
            host_ptr + offset,
            chunk_size,
            cudaMemcpyHostToDevice
        );
        
        if (err != cudaSuccess) {
            cudaFree(device_ptr);
            throw CudaError::model_load_failed(
                "Failed to copy chunk to VRAM at offset " + std::to_string(offset)
            );
        }
    }
    
    return device_ptr;
}
```

#### 5. Helper Functions

```cpp
const gguf::GGUFTensor* GPTWeightLoader::find_tensor(
    const std::vector<gguf::GGUFTensor>& tensors,
    const std::string& name
) {
    for (const auto& tensor : tensors) {
        if (tensor.name == name) {
            return &tensor;
        }
    }
    return nullptr;
}

void* GPTWeightLoader::load_tensor(
    const io::MmapFile& mmap,
    const std::vector<gguf::GGUFTensor>& tensors,
    const std::string& name
) {
    const gguf::GGUFTensor* tensor = find_tensor(tensors, name);
    if (!tensor) {
        // Some tensors might be optional (e.g., bias)
        return nullptr;
    }
    
    return allocate_and_copy(mmap, tensor->offset + data_start, tensor->size);
}
```

---

## Testing Strategy

### Unit Test

```cpp
TEST(GPTWeightLoader, LoadFromGGUF_Qwen) {
    std::string path = "/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    auto model = GPTWeightLoader::load_from_gguf(path);
    
    // Verify structure
    ASSERT_NE(model, nullptr);
    ASSERT_NE(model->token_embeddings, nullptr);
    ASSERT_GT(model->layers.size(), 0);
    
    // Verify VRAM allocation
    EXPECT_GT(model->total_vram_bytes, 0);
    
    // Verify pointers are device memory
    cudaPointerAttributes attrs;
    cudaPointerGetAttributes(&attrs, model->token_embeddings);
    EXPECT_EQ(attrs.type, cudaMemoryTypeDevice);
    EXPECT_EQ(attrs.hostPointer, nullptr);  // No UMA
}
```

### Manual Verification

```bash
# Before loading
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

# Load model
./target/release/worker-orcd --model ... --gpu-device 0 --port 8080

# After loading
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

# Difference should be ~400MB for Qwen model
```

---

## Dependencies

**Upstream**: GT-051 (needs real config)  
**Downstream**: GT-054, GT-055 (need weights for inference)

---

## Definition of Done

- [ ] All TODOs removed from `load_from_gguf()`
- [ ] `load_embeddings()` implemented
- [ ] `load_layer()` implemented
- [ ] `load_output_head()` implemented
- [ ] `allocate_and_copy()` implemented
- [ ] Memory-mapped I/O used (no full file read)
- [ ] Chunked transfer implemented (1MB chunks)
- [ ] Unit tests pass
- [ ] `nvidia-smi` shows VRAM usage increase
- [ ] No memory leaks (valgrind or cuda-memcheck)
- [ ] Story marked complete

---

## Estimated Time

**Optimistic**: 6 hours  
**Realistic**: 6-8 hours  
**Pessimistic**: 10 hours (if tensor mapping is complex)

---

## Notes

### Why This Takes Time

- Need to handle 12+ tensors per layer
- Need to map GGUF tensor names to struct fields
- Need to handle optional tensors (some models don't have bias)
- Need to verify VRAM allocation for each tensor

### Existing Code to Use

- ✅ `io::MmapFile` - Memory-mapped I/O
- ✅ `gguf::parse_gguf_header()` - GGUF parsing
- ✅ `cudaMalloc()` - VRAM allocation
- ✅ `cudaMemcpy()` - H2D transfer
- ✅ `GPTLayerWeights` - Structure already defined

### Risks

- ⚠️ GGUF tensor names might vary between models
- ⚠️ Some tensors might be fused (e.g., QKV combined)
- ⚠️ Quantized tensors need special handling

**Mitigation**: Start with Qwen model, add GPT-OSS-20B support later

---

**Created by**: Project Management Team 📋  
**Assigned to**: GPT-Gamma 🤖  
**Status**: TODO  
**Related Fine**: FINE-001-20251005
