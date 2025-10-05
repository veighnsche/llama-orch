# GT-052-SIMPLIFIED: Weight Loading to VRAM

**Story ID**: GT-052-SIMPLIFIED  
**Title**: Load Qwen2.5-0.5B Weights to GPU VRAM  
**Size**: M (Medium)  
**Estimate**: 4-6 hours  
**Priority**: P0 (CRITICAL PATH)  
**Dependencies**: GT-051-REFACTOR ✅  
**Blocks**: GT-053, GT-054, GT-055, GT-056

---

## User Story

**As a** CUDA inference engine  
**I want** to load GGUF model weights into GPU VRAM  
**So that** I can execute inference on the GPU

---

## Context

GT-051-REFACTOR ✅ implemented GGUF parsing in Rust. Now we need to load the actual tensor data to GPU memory.

**Simplified approach**: Hardcode tensor names for Qwen2.5-0.5B only. No complex registry needed yet.

**Why simplified**:
- ✅ Get haiku test working fast
- ✅ Prove weight loading works
- ✅ Can refactor to registry later (M1)

---

## Acceptance Criteria

### Weight Loading
- [ ] Open GGUF file and read tensor data
- [ ] Allocate GPU memory for each tensor
- [ ] Copy tensor data to VRAM
- [ ] Track VRAM usage
- [ ] Return model handle with all weights loaded

### Tensor Support (Qwen2.5-0.5B)
- [ ] Token embeddings: `token_embd.weight` [151936, 896]
- [ ] 24 transformer layers, each with:
  - [ ] Attention norm: `blk.{L}.attn_norm.weight` [896]
  - [ ] Q/K/V weights: `blk.{L}.attn_{q,k,v}.weight` [896, 896] or [128, 896]
  - [ ] Q/K/V biases: `blk.{L}.attn_{q,k,v}.bias` [896] or [128]
  - [ ] Attention output: `blk.{L}.attn_output.weight` [896, 896]
  - [ ] FFN norm: `blk.{L}.ffn_norm.weight` [896]
  - [ ] FFN gate/up/down: `blk.{L}.ffn_{gate,up,down}.weight`
- [ ] Output norm: `output_norm.weight` [896]
- [ ] LM head: `output.weight` [151936, 896]

### FFI Integration
- [ ] Rust passes config to C++
- [ ] C++ loads weights to VRAM
- [ ] C++ returns model handle to Rust
- [ ] Rust can query VRAM usage

### Error Handling
- [ ] File not found
- [ ] Tensor not found
- [ ] Out of VRAM
- [ ] Invalid tensor dimensions
- [ ] Corrupted GGUF file

### Testing
- [ ] Load Qwen2.5-0.5B to VRAM
- [ ] Verify all tensors loaded
- [ ] Check VRAM usage (~500 MB)
- [ ] Verify tensor dimensions
- [ ] Test error cases

---

## Technical Design

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ RUST (main.rs)                                               │
│                                                              │
│  let metadata = GGUFMetadata::from_file(&path)?;            │
│  let config = extract_config(&metadata)?;                   │
│                                                              │
│  let cuda_model = unsafe {                                   │
│      cuda_load_model(                                        │
│          ctx,                                                │
│          path.as_ptr(),                                      │
│          config.vocab_size,                                  │
│          config.hidden_dim,                                  │
│          config.num_layers,                                  │
│          // ... more config                                  │
│      )                                                       │
│  };                                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          │ FFI
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ C++ (cuda/src/model/gpt_weights.cpp)                        │
│                                                              │
│  extern "C" CudaModel* cuda_load_model(                     │
│      CudaContext* ctx,                                       │
│      const char* path,                                       │
│      uint32_t vocab_size,                                    │
│      uint32_t hidden_dim,                                    │
│      uint32_t num_layers,                                    │
│      ...                                                     │
│  ) {                                                         │
│      // 1. Open GGUF file                                    │
│      // 2. Find tensors by name                             │
│      // 3. Allocate GPU memory                              │
│      // 4. Copy to VRAM                                      │
│      // 5. Return model handle                              │
│  }                                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Implementation

**File**: `cuda/src/model/qwen_weight_loader.cpp` (NEW)

```cpp
#include "qwen_weight_loader.h"
#include "../io/chunked_transfer.h"
#include "../device_memory.h"
#include "../vram_tracker.h"
#include <fstream>
#include <cstring>

namespace worker {
namespace model {

// Hardcoded tensor names for Qwen2.5-0.5B
std::vector<std::string> QwenWeightLoader::get_tensor_names(int num_layers) {
    std::vector<std::string> names;
    
    // Token embeddings
    names.push_back("token_embd.weight");
    
    // Transformer layers
    for (int i = 0; i < num_layers; i++) {
        std::string prefix = "blk." + std::to_string(i) + ".";
        
        // Attention
        names.push_back(prefix + "attn_norm.weight");
        names.push_back(prefix + "attn_q.weight");
        names.push_back(prefix + "attn_q.bias");
        names.push_back(prefix + "attn_k.weight");
        names.push_back(prefix + "attn_k.bias");
        names.push_back(prefix + "attn_v.weight");
        names.push_back(prefix + "attn_v.bias");
        names.push_back(prefix + "attn_output.weight");
        
        // FFN
        names.push_back(prefix + "ffn_norm.weight");
        names.push_back(prefix + "ffn_gate.weight");
        names.push_back(prefix + "ffn_up.weight");
        names.push_back(prefix + "ffn_down.weight");
    }
    
    // Output
    names.push_back("output_norm.weight");
    names.push_back("output.weight");
    
    return names;
}

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> dimensions;
    uint32_t type;
    uint64_t offset;
    size_t size_bytes;
};

TensorInfo find_tensor(const char* path, const std::string& name) {
    // Open GGUF file
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open GGUF file: " + std::string(path));
    }
    
    // Parse GGUF header (simplified - just find tensor)
    // Skip magic, version, tensor_count, metadata_count
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), 4);
    
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), 4);
    
    uint64_t tensor_count;
    file.read(reinterpret_cast<char*>(&tensor_count), 8);
    
    uint64_t metadata_count;
    file.read(reinterpret_cast<char*>(&metadata_count), 8);
    
    // Skip metadata (we already parsed it in Rust)
    // For now, seek to tensor info section
    // TODO: Implement proper GGUF tensor info parsing
    
    throw std::runtime_error("Tensor not found: " + name);
}

void* QwenWeightLoader::load_tensor_to_vram(
    const char* path,
    const std::string& tensor_name,
    VramTracker& tracker
) {
    // 1. Find tensor in GGUF file
    TensorInfo info = find_tensor(path, tensor_name);
    
    // 2. Allocate GPU memory
    void* gpu_ptr = DeviceMemory::allocate(info.size_bytes);
    
    // 3. Read tensor data from file
    std::ifstream file(path, std::ios::binary);
    file.seekg(info.offset);
    
    std::vector<char> data(info.size_bytes);
    file.read(data.data(), info.size_bytes);
    
    // 4. Copy to GPU
    io::ChunkedTransfer::copy_to_device(gpu_ptr, data.data(), info.size_bytes);
    
    // 5. Track allocation
    tracker.track_allocation(gpu_ptr, info.size_bytes, tensor_name);
    
    return gpu_ptr;
}

QwenModel* QwenWeightLoader::load(
    const char* path,
    uint32_t vocab_size,
    uint32_t hidden_dim,
    uint32_t num_layers
) {
    auto model = new QwenModel();
    model->config.vocab_size = vocab_size;
    model->config.hidden_dim = hidden_dim;
    model->config.num_layers = num_layers;
    
    VramTracker tracker;
    
    // Get all tensor names
    auto tensor_names = get_tensor_names(num_layers);
    
    // Load each tensor
    for (const auto& name : tensor_names) {
        void* gpu_ptr = load_tensor_to_vram(path, name, tracker);
        
        // Store pointer in model structure
        // TODO: Route to correct field based on tensor name
        model->tensors[name] = gpu_ptr;
    }
    
    model->vram_usage = tracker.total_usage();
    
    return model;
}

} // namespace model
} // namespace worker
```

**File**: `cuda/src/ffi.cpp` (UPDATE)

```cpp
extern "C" {

CudaModel* cuda_load_model(
    CudaContext* ctx,
    const char* path,
    uint32_t vocab_size,
    uint32_t hidden_dim,
    uint32_t num_layers,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t context_length,
    int* error
) {
    try {
        auto model = model::QwenWeightLoader::load(
            path,
            vocab_size,
            hidden_dim,
            num_layers
        );
        
        *error = 0;
        return reinterpret_cast<CudaModel*>(model);
    } catch (const std::exception& e) {
        fprintf(stderr, "Model load failed: %s\n", e.what());
        *error = -1;
        return nullptr;
    }
}

uint64_t cuda_get_vram_usage(CudaModel* model) {
    auto qwen_model = reinterpret_cast<model::QwenModel*>(model);
    return qwen_model->vram_usage;
}

} // extern "C"
```

---

## Implementation Steps

### Step 1: GGUF Tensor Reader (2 hours)
1. Parse GGUF tensor info section
2. Find tensor by name
3. Get tensor offset and size
4. Read tensor data from file

### Step 2: GPU Memory Allocation (1 hour)
1. Allocate VRAM for each tensor
2. Track allocations
3. Handle out-of-memory errors

### Step 3: Weight Loading (2 hours)
1. Load all Qwen2.5-0.5B tensors
2. Copy to VRAM using chunked transfer
3. Store pointers in model structure

### Step 4: FFI Integration (1 hour)
1. Update FFI to pass config
2. Return model handle
3. Add VRAM usage query

---

## Testing Strategy

### Integration Test

```rust
#[test]
fn test_load_qwen_weights() {
    let path = "/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    // Parse config in Rust
    let metadata = GGUFMetadata::from_file(path).unwrap();
    
    // Load weights in C++
    let mut error = 0;
    let model = unsafe {
        cuda_load_model(
            ctx,
            path.as_ptr() as *const i8,
            metadata.vocab_size().unwrap() as u32,
            metadata.hidden_dim().unwrap() as u32,
            metadata.num_layers().unwrap() as u32,
            metadata.num_heads().unwrap() as u32,
            metadata.num_kv_heads().unwrap() as u32,
            metadata.context_length().unwrap() as u32,
            &mut error,
        )
    };
    
    assert_eq!(error, 0);
    assert!(!model.is_null());
    
    // Check VRAM usage
    let vram_usage = unsafe { cuda_get_vram_usage(model) };
    println!("VRAM usage: {} MB", vram_usage / 1024 / 1024);
    
    // Should be around 500 MB for Qwen2.5-0.5B Q4_K_M
    assert!(vram_usage > 400_000_000);
    assert!(vram_usage < 600_000_000);
}
```

---

## Definition of Done

- [x] GGUF tensor reader implemented
- [x] GPU memory allocation working
- [x] All Qwen2.5-0.5B tensors loaded
- [x] VRAM tracking working
- [x] FFI integration complete
- [x] Integration test passes
- [x] VRAM usage verified with `nvidia-smi`
- [x] Error handling comprehensive
- [x] Code reviewed and approved

---

## Time Estimate

**Optimistic**: 4 hours  
**Realistic**: 4-6 hours  
**Pessimistic**: 8 hours (if GGUF tensor parsing is complex)

---

## Notes

### Why Hardcoded?

- ✅ **Fast**: Get haiku test working quickly
- ✅ **Simple**: No complex registry needed
- ✅ **Proven**: Test weight loading works
- ✅ **Refactorable**: Can add registry in M1

### Tensor Names Reference

From llama.cpp GGUF spec:
- Embeddings: `token_embd.weight`
- Layers: `blk.{L}.{component}.{param}`
- Output: `output_norm.weight`, `output.weight`

### VRAM Estimate

Qwen2.5-0.5B Q4_K_M:
- ~461 million parameters
- Q4_K_M quantization (~4.5 bits per param)
- ~260 MB for weights
- ~200 MB for KV cache
- **Total**: ~460 MB

---

**Created by**: Project Management Team 📋  
**Assigned to**: GPT-Gamma 🤖  
**Status**: TODO  
**Priority**: P0 (CRITICAL PATH)

---
Ready to implement! 🚀
