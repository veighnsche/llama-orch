# CUDA Model Loading (CUDA-5200)

**Status**: Draft  
**Module**: `model`  
**Files**: `src/model.cpp`, `include/model.hpp`  
**Conformance**: RFC-2119

---

## 0. Scope

The model module handles loading model weights from disk/RAM into GPU VRAM. It parses GGUF format, allocates VRAM, and manages model lifetime.

**Parent**: `00_cuda_overview.md`

---

## 1. Responsibilities

### [CUDA-5201] Model Loading
The module MUST load a single model from disk or RAM into VRAM.

### [CUDA-5202] GGUF Format Support
The module MUST support GGUF (GPT-Generated Unified Format) for M0.

### [CUDA-5203] VRAM Allocation
The module MUST allocate VRAM for model weights and track actual usage.

### [CUDA-5204] Single Model
The module MUST support exactly ONE model per worker process.

---

## 2. C API

### [CUDA-5210] Load Model
```c
// Load model from disk/RAM to VRAM
// Returns: Opaque model handle, or NULL on error
// vram_bytes_used: Set to actual VRAM bytes allocated
// error_code: Set to error code (0 = success)
CudaModel* cuda_load_model(
    CudaContext* ctx,
    const char* model_path,
    uint64_t* vram_bytes_used,
    int* error_code
);
```

### [CUDA-5211] Unload Model
```c
// Free model and release VRAM
void cuda_unload_model(CudaModel* model);
```

### [CUDA-5212] Query Model
```c
// Get VRAM usage
uint64_t cuda_model_get_vram_usage(CudaModel* model);

// Get model metadata
void cuda_model_get_metadata(
    CudaModel* model,
    CudaModelMetadata* metadata,
    int* error_code
);
```

---

## 3. C++ Implementation

### [CUDA-5220] Model Class
```cpp
// include/model.hpp
namespace worker_cuda {

struct ModelMetadata {
    std::string architecture;  // "llama", "gpt2", etc.
    std::string name;
    uint32_t context_length;
    uint32_t vocab_size;
    uint32_t embedding_length;
    uint32_t num_layers;
};

class Model {
public:
    // Load model from path
    Model(const Context& ctx, const std::string& path);
    ~Model();
    
    // Non-copyable, movable
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = default;
    Model& operator=(Model&&) = default;
    
    // Get VRAM usage
    uint64_t vram_bytes() const { return vram_bytes_; }
    
    // Get metadata
    const ModelMetadata& metadata() const { return metadata_; }
    
    // Get device pointers (for inference)
    const void* weights() const { return weights_.get(); }
    
private:
    void parse_gguf(const std::string& path);
    void allocate_vram();
    void copy_to_vram(const void* host_data, size_t size);
    
    ModelMetadata metadata_;
    std::unique_ptr<DeviceMemory> weights_;
    uint64_t vram_bytes_;
};

} // namespace worker_cuda
```

### [CUDA-5221] GGUF Parsing
```cpp
// src/model.cpp
namespace worker_cuda {

struct GGUFHeader {
    uint32_t magic;      // 'GGUF' (0x47475546)
    uint32_t version;    // Version 3
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
};

void Model::parse_gguf(const std::string& path) {
    // Open file
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw CudaError(CUDA_ERROR_MODEL_LOAD_FAILED,
                       "Failed to open model file");
    }
    
    // Read header
    GGUFHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    // Validate magic
    if (header.magic != 0x47475546) {
        throw CudaError(CUDA_ERROR_MODEL_LOAD_FAILED,
                       "Invalid GGUF magic bytes");
    }
    
    // Validate version
    if (header.version != 3) {
        throw CudaError(CUDA_ERROR_MODEL_LOAD_FAILED,
                       "Unsupported GGUF version");
    }
    
    // Parse metadata
    parse_metadata(file, header.metadata_kv_count);
    
    // Parse tensors
    parse_tensors(file, header.tensor_count);
}

void Model::allocate_vram() {
    // Calculate total size needed
    size_t total_size = calculate_total_size();
    
    // Allocate VRAM
    weights_ = std::make_unique<DeviceMemory>(total_size);
    vram_bytes_ = total_size;
}

void Model::copy_to_vram(const void* host_data, size_t size) {
    cudaError_t err = cudaMemcpy(
        weights_->get(),
        host_data,
        size,
        cudaMemcpyHostToDevice
    );
    
    if (err != cudaSuccess) {
        throw CudaError(CUDA_ERROR_MODEL_LOAD_FAILED,
                       "Failed to copy model to VRAM");
    }
}

} // namespace worker_cuda
```

### [CUDA-5222] VRAM Management
```cpp
// include/model.hpp (private helper)
class DeviceMemory {
public:
    explicit DeviceMemory(size_t bytes) {
        cudaError_t err = cudaMalloc(&ptr_, bytes);
        if (err != cudaSuccess) {
            throw CudaError(CUDA_ERROR_OUT_OF_MEMORY,
                           "Failed to allocate VRAM");
        }
        size_ = bytes;
    }
    
    ~DeviceMemory() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }
    
    // Non-copyable, movable
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    
    DeviceMemory(DeviceMemory&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    void* get() const { return ptr_; }
    size_t size() const { return size_; }
    
private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
};
```

---

## 4. FFI Implementation

### [CUDA-5230] C API Wrapper
```cpp
// src/ffi.cpp
extern "C" {

CudaModel* cuda_load_model(
    CudaContext* ctx,
    const char* model_path,
    uint64_t* vram_bytes_used,
    int* error_code
) {
    try {
        auto ctx_ptr = reinterpret_cast<worker_cuda::Context*>(ctx);
        auto model = std::make_unique<worker_cuda::Model>(
            *ctx_ptr,
            std::string(model_path)
        );
        
        *vram_bytes_used = model->vram_bytes();
        *error_code = CUDA_SUCCESS;
        return reinterpret_cast<CudaModel*>(model.release());
    } catch (const worker_cuda::CudaError& e) {
        *error_code = e.code();
        *vram_bytes_used = 0;
        return nullptr;
    } catch (...) {
        *error_code = CUDA_ERROR_UNKNOWN;
        *vram_bytes_used = 0;
        return nullptr;
    }
}

void cuda_unload_model(CudaModel* model) {
    if (model) {
        auto ptr = reinterpret_cast<worker_cuda::Model*>(model);
        delete ptr;
    }
}

uint64_t cuda_model_get_vram_usage(CudaModel* model) {
    if (!model) return 0;
    auto ptr = reinterpret_cast<worker_cuda::Model*>(model);
    return ptr->vram_bytes();
}

} // extern "C"
```

---

## 5. Loading Strategies

### [CUDA-5240] Disk Loading
```cpp
Model::Model(const Context& ctx, const std::string& path) {
    // 1. Memory-map file for efficient reading
    auto mapped_file = mmap_file(path);
    
    // 2. Parse GGUF format
    parse_gguf(mapped_file);
    
    // 3. Allocate VRAM
    allocate_vram();
    
    // 4. Copy weights to VRAM in chunks
    copy_weights_chunked(mapped_file);
    
    // 5. Unmap file
    munmap_file(mapped_file);
}
```

### [CUDA-5241] RAM-Staged Loading
```cpp
// For future optimization: model pre-staged in RAM by pool manager
Model::Model(const Context& ctx, const void* ram_ptr, size_t size) {
    // 1. Parse GGUF from RAM
    parse_gguf_from_memory(ram_ptr, size);
    
    // 2. Allocate VRAM
    allocate_vram();
    
    // 3. Copy directly from RAM to VRAM (faster)
    copy_to_vram(ram_ptr, size);
}
```

---

## 6. Error Handling

### [CUDA-5250] Error Codes
- `CUDA_SUCCESS` (0) — Model loaded successfully
- `CUDA_ERROR_OUT_OF_MEMORY` (2) — Insufficient VRAM
- `CUDA_ERROR_MODEL_LOAD_FAILED` (4) — Parse error or I/O error

### [CUDA-5251] Validation
The module MUST validate:
- File exists and is readable
- GGUF magic bytes are correct
- GGUF version is supported (version 3)
- Tensor count is reasonable (< 10,000)
- Total size fits in available VRAM

### [CUDA-5252] Cleanup on Error
If loading fails, the module MUST:
1. Free any partial VRAM allocations
2. Close file handles
3. Throw exception with detailed error

---

## 7. Performance

### [CUDA-5260] Memory-Mapped I/O
The module SHOULD use `mmap()` for efficient file reading without loading entire file into RAM.

### [CUDA-5261] Chunked Transfer
The module SHOULD copy model to VRAM in chunks (e.g., 1MB) to avoid large temporary buffers.

### [CUDA-5262] Progress Reporting
The module MAY report loading progress via callback for large models.

---

## 8. Testing

### [CUDA-5270] Unit Tests
```cpp
// tests/test_model.cpp
TEST(ModelTest, LoadValidModel) {
    int error_code;
    auto ctx = cuda_init(0, &error_code);
    ASSERT_NE(ctx, nullptr);
    
    uint64_t vram_bytes = 0;
    auto model = cuda_load_model(
        ctx,
        "test_model.gguf",
        &vram_bytes,
        &error_code
    );
    
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(error_code, CUDA_SUCCESS);
    ASSERT_GT(vram_bytes, 0);
    
    cuda_unload_model(model);
    cuda_destroy(ctx);
}

TEST(ModelTest, LoadInvalidFile) {
    int error_code;
    auto ctx = cuda_init(0, &error_code);
    
    uint64_t vram_bytes = 0;
    auto model = cuda_load_model(
        ctx,
        "nonexistent.gguf",
        &vram_bytes,
        &error_code
    );
    
    ASSERT_EQ(model, nullptr);
    ASSERT_EQ(error_code, CUDA_ERROR_MODEL_LOAD_FAILED);
    ASSERT_EQ(vram_bytes, 0);
    
    cuda_destroy(ctx);
}

TEST(ModelTest, GetVramUsage) {
    // Load model
    auto model = load_test_model();
    
    uint64_t usage = cuda_model_get_vram_usage(model);
    ASSERT_GT(usage, 0);
    
    cuda_unload_model(model);
}
```

---

## 9. GGUF Format Details

### [CUDA-5280] Header Structure
```cpp
struct GGUFHeader {
    uint32_t magic;              // 'GGUF' (0x47475546)
    uint32_t version;            // 3
    uint64_t tensor_count;       // Number of tensors
    uint64_t metadata_kv_count;  // Number of metadata entries
};
```

### [CUDA-5281] Metadata Keys
Required metadata:
- `general.architecture` — Model architecture ("llama", "gpt2", etc.)
- `general.name` — Model name
- `llama.context_length` — Context window size
- `llama.embedding_length` — Embedding dimensions
- `llama.block_count` — Number of layers

### [CUDA-5282] Tensor Format
```cpp
struct GGUFTensor {
    std::string name;
    uint32_t n_dimensions;
    uint64_t dimensions[4];
    uint32_t type;  // FP32, FP16, Q8_0, etc.
    uint64_t offset;
};
```

---

## 10. Memory Layout

### [CUDA-5290] VRAM Layout
```
┌─────────────────────────────────────┐
│ Model Weights (contiguous)          │
│ - Embedding weights                 │
│ - Layer 0 weights                   │
│ - Layer 1 weights                   │
│ - ...                               │
│ - Layer N weights                   │
│ - Output weights                    │
└─────────────────────────────────────┘
```

### [CUDA-5291] Alignment
All tensors MUST be aligned to 256-byte boundaries for optimal GPU access.

---

## 11. Traceability

**Code**: `cuda/src/model.cpp`, `cuda/include/model.hpp`  
**Tests**: `cuda/tests/test_model.cpp`  
**Parent**: `00_cuda_overview.md`  
**Spec IDs**: CUDA-5201 to CUDA-5291

---

**End of Specification**
