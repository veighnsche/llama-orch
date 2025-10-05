# Gap Analysis: Old CUDA Specs vs. M0 Leading Spec

**Date**: 2025-10-03  
**Status**: Analysis Complete  
**Old Specs Location**: `/home/vince/Projects/llama-orch/bin/worker-orcd/cuda/.specs/`  
**Leading Spec**: `/home/vince/Projects/llama-orch/bin/.specs/01_M0_worker_orcd.md`

---

## Executive Summary

The old CUDA specs (00-04) provide **valuable low-level implementation details** that are **missing from the M0 leading spec**. While the M0 spec is comprehensive at the architectural and API level, it lacks the granular CUDA implementation guidance found in the old specs.

**Recommendation**: **Retain and update** the old CUDA specs as implementation guides, synchronized with M0 requirements.

---

## Gap Categories

### ✅ **Gaps Found** (Old specs have details missing in M0)
### ⚠️ **Conflicts** (Old specs contradict M0)
### 🔄 **Overlaps** (Both specs cover, but old has more detail)

---

## 1. Architecture & Module Organization

### ✅ **GAP-1: Detailed C++ Class Design**

**Old Specs Have**:
- Complete C++ class hierarchies with RAII patterns
- Opaque handle implementation details
- Exception-to-error-code conversion patterns
- Move semantics and ownership rules

**M0 Spec Has**:
- High-level FFI boundary description
- General RAII principles
- Basic error handling approach

**Example from `01_context.md`**:
```cpp
class Context {
public:
    explicit Context(int gpu_device);
    ~Context();
    
    // Non-copyable, movable
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = default;
    Context& operator=(Context&&) = default;
    
    int device() const { return device_; }
    const cudaDeviceProp& properties() const { return props_; }
    
private:
    int device_;
    cudaDeviceProp props_;
};
```

**Missing in M0**: Concrete class design, move semantics, property accessors.

---

### ✅ **GAP-2: DeviceMemory RAII Wrapper**

**Old Specs Have** (`02_model.md`):
```cpp
class DeviceMemory {
public:
    explicit DeviceMemory(size_t bytes) {
        cudaError_t err = cudaMalloc(&ptr_, bytes);
        if (err != cudaSuccess) {
            throw CudaError(CUDA_ERROR_OUT_OF_MEMORY, "Failed to allocate VRAM");
        }
        size_ = bytes;
    }
    
    ~DeviceMemory() {
        if (ptr_) cudaFree(ptr_);
    }
    
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

**Missing in M0**: Complete RAII wrapper implementation with move semantics.

---

### ✅ **GAP-3: CudaStream RAII Wrapper**

**Old Specs Have** (`00_cuda_overview.md`):
```cpp
class CudaStream {
public:
    CudaStream();
    ~CudaStream() { cudaStreamDestroy(stream_); }
    cudaStream_t get() const { return stream_; }
private:
    cudaStream_t stream_;
};
```

**Missing in M0**: Stream management abstraction.

---

## 2. Error Handling Details

### ✅ **GAP-4: Exception-to-Error-Code Conversion Pattern**

**Old Specs Have** (`00_cuda_overview.md`, `01_context.md`):
```cpp
extern "C" CudaModel* cuda_load_model(..., int* error_code) {
    try {
        auto model = std::make_unique<Model>(...);
        *error_code = 0;
        return reinterpret_cast<CudaModel*>(model.release());
    } catch (const CudaError& e) {
        *error_code = e.code();
        return nullptr;
    } catch (...) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return nullptr;
    }
}
```

**M0 Has**: General principle of error code conversion.

**Missing in M0**: Concrete pattern for all FFI functions.

---

### ✅ **GAP-5: CudaError Exception Class**

**Old Specs Imply**:
```cpp
class CudaError : public std::exception {
public:
    CudaError(int code, const std::string& msg) 
        : code_(code), msg_(msg) {}
    
    int code() const { return code_; }
    const char* what() const noexcept override { return msg_.c_str(); }
    
private:
    int code_;
    std::string msg_;
};
```

**Missing in M0**: Exception class design.

---

### ✅ **GAP-6: Error Message Retrieval**

**Old Specs Have** (`00_cuda_overview.md`):
```cpp
extern "C" const char* cuda_error_message(int error_code) {
    switch (error_code) {
        case CUDA_SUCCESS: return "Success";
        case CUDA_ERROR_DEVICE_NOT_FOUND: return "CUDA device not found";
        case CUDA_ERROR_OUT_OF_MEMORY: return "Out of GPU memory";
        // ... etc
        default: return "Unknown error";
    }
}
```

**Missing in M0**: Error message retrieval function.

---

## 3. Model Loading Implementation

### ✅ **GAP-7: GGUF Header Parsing Details**

**Old Specs Have** (`02_model.md`):
```cpp
struct GGUFHeader {
    uint32_t magic;              // 'GGUF' (0x47475546)
    uint32_t version;            // 3
    uint64_t tensor_count;       // Number of tensors
    uint64_t metadata_kv_count;  // Number of metadata entries
};

void Model::parse_gguf(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw CudaError(CUDA_ERROR_MODEL_LOAD_FAILED, "Failed to open model file");
    }
    
    GGUFHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    if (header.magic != 0x47475546) {
        throw CudaError(CUDA_ERROR_MODEL_LOAD_FAILED, "Invalid GGUF magic bytes");
    }
    
    if (header.version != 3) {
        throw CudaError(CUDA_ERROR_MODEL_LOAD_FAILED, "Unsupported GGUF version");
    }
    
    parse_metadata(file, header.metadata_kv_count);
    parse_tensors(file, header.tensor_count);
}
```

**M0 Has**: High-level requirement for GGUF v3 support.

**Missing in M0**: Concrete parsing implementation.

---

### ✅ **GAP-8: Tensor Format Structure**

**Old Specs Have** (`02_model.md`):
```cpp
struct GGUFTensor {
    std::string name;
    uint32_t n_dimensions;
    uint64_t dimensions[4];
    uint32_t type;  // FP32, FP16, Q8_0, etc.
    uint64_t offset;
};
```

**Missing in M0**: Tensor metadata structure.

---

### ✅ **GAP-9: ModelMetadata Structure**

**Old Specs Have** (`02_model.md`):
```cpp
struct ModelMetadata {
    std::string architecture;  // "llama", "gpt2", etc.
    std::string name;
    uint32_t context_length;
    uint32_t vocab_size;
    uint32_t embedding_length;
    uint32_t num_layers;
};
```

**M0 Has**: Requirements for metadata fields.

**Missing in M0**: Concrete C++ structure.

---

### ✅ **GAP-10: Chunked Transfer Implementation**

**Old Specs Have** (`02_model.md`):
```cpp
const size_t CHUNK_SIZE = 1024 * 1024;  // 1MB
for (size_t offset = 0; offset < total_size; offset += CHUNK_SIZE) {
    size_t chunk_size = std::min(CHUNK_SIZE, total_size - offset);
    cudaMemcpy(device_ptr + offset, host_ptr + offset, chunk_size, cudaMemcpyHostToDevice);
}
```

**M0 Has**: Requirement for chunked transfer.

**Missing in M0**: Concrete implementation pattern.

---

### ✅ **GAP-11: Memory-Mapped File Handling**

**Old Specs Have** (`02_model.md`):
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

**M0 Has**: Requirement for mmap.

**Missing in M0**: Complete mmap workflow.

---

## 4. Inference Execution Details

### ✅ **GAP-12: InferenceConfig Structure**

**Old Specs Have** (`03_inference.md`):
```cpp
struct InferenceConfig {
    int max_tokens;
    float temperature;
    uint64_t seed;
    int top_k = 50;
    float top_p = 0.95f;
};
```

**M0 Has**: API parameters.

**Missing in M0**: C++ config structure with defaults.

---

### ✅ **GAP-13: InferenceResult Class Design**

**Old Specs Have** (`03_inference.md`):
```cpp
class InferenceResult {
public:
    InferenceResult(const Model& model, const std::string& prompt, const InferenceConfig& config);
    ~InferenceResult();
    
    // Non-copyable, non-movable (holds CUDA resources)
    InferenceResult(const InferenceResult&) = delete;
    InferenceResult& operator=(const InferenceResult&) = delete;
    InferenceResult(InferenceResult&&) = delete;
    InferenceResult& operator=(InferenceResult&&) = delete;
    
    bool next_token(std::string& token_out, int& token_index);
    bool is_done() const { return current_token_ >= config_.max_tokens; }
    
private:
    void tokenize_prompt(const std::string& prompt);
    void allocate_kv_cache();
    void run_forward_pass();
    int sample_token();
    std::string detokenize(int token_id);
    
    const Model& model_;
    InferenceConfig config_;
    std::vector<int> prompt_tokens_;
    int current_token_ = 0;
    std::unique_ptr<DeviceMemory> kv_cache_;
    std::unique_ptr<DeviceMemory> logits_;
    cudaStream_t stream_;
    std::mt19937_64 rng_;
};
```

**M0 Has**: High-level inference flow.

**Missing in M0**: Complete class design with state management.

---

### ✅ **GAP-14: KV Cache Allocation Formula**

**Old Specs Have** (`03_inference.md`):
```cpp
void InferenceResult::allocate_kv_cache() {
    // Calculate KV cache size
    // For each layer: 2 (K and V) * context_length * hidden_dim
    size_t cache_size = 2 * 
                       model_.metadata().num_layers *
                       model_.metadata().context_length *
                       model_.metadata().embedding_length *
                       sizeof(float);
    
    kv_cache_ = std::make_unique<DeviceMemory>(cache_size);
    cudaMemset(kv_cache_->get(), 0, cache_size);
}
```

**M0 Has**: Requirement for KV cache.

**Missing in M0**: Concrete allocation formula.

---

### ✅ **GAP-15: KV Cache Layout Diagram**

**Old Specs Have** (`03_inference.md`):
```
KV Cache Layout (per layer):
┌─────────────────────────────────────┐
│ Keys   [context_length × hidden_dim] │
├─────────────────────────────────────┤
│ Values [context_length × hidden_dim] │
└─────────────────────────────────────┘
```

**Missing in M0**: Memory layout visualization.

---

### ✅ **GAP-16: Kernel Error Checking Pattern**

**Old Specs Have** (`03_inference.md`):
```cpp
void InferenceResult::run_forward_pass() {
    // Launch kernel
    attention_kernel<<<grid, block, 0, stream_>>>(...);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw CudaError(CUDA_ERROR_KERNEL_LAUNCH_FAILED, cudaGetErrorString(err));
    }
    
    // Synchronize and check for execution errors
    err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
        throw CudaError(CUDA_ERROR_INFERENCE_FAILED, cudaGetErrorString(err));
    }
}
```

**M0 Has**: Requirement for error checking.

**Missing in M0**: Concrete error checking pattern.

---

### ✅ **GAP-17: Sampling Implementation Details**

**Old Specs Have** (`03_inference.md`):
```cpp
int InferenceResult::sample_token() {
    // Copy logits to host
    std::vector<float> host_logits(model_.metadata().vocab_size);
    cudaMemcpy(host_logits.data(), logits_->get(), 
               host_logits.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Apply temperature
    for (float& logit : host_logits) {
        logit /= config_.temperature;
    }
    
    // Softmax
    softmax(host_logits);
    
    // Sample with top-k/top-p
    return sample_top_k_top_p(host_logits, config_.top_k, config_.top_p, rng_);
}
```

**M0 Has**: High-level sampling requirement.

**Missing in M0**: Complete sampling implementation with top-k/top-p.

---

## 5. Health Monitoring Details

### ✅ **GAP-18: HealthStatus Structure**

**Old Specs Have** (`04_health.md`):
```cpp
struct HealthStatus {
    bool vram_resident;
    uint64_t vram_bytes_used;
    uint64_t vram_bytes_free;
    int temperature_celsius;
    bool has_ecc_errors;
};
```

**M0 Has**: Health endpoint JSON response.

**Missing in M0**: C++ health status structure.

---

### ✅ **GAP-19: VRAM Residency Check Implementation**

**Old Specs Have** (`04_health.md`):
```cpp
bool Health::check_vram_residency(const Model& model) {
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, model.weights());
    
    if (err != cudaSuccess) return false;
    
    // Verify pointer is device memory (not managed/host)
    if (attrs.type != cudaMemoryTypeDevice) return false;
    
    // Verify no host pointer (no UMA)
    if (attrs.hostPointer != nullptr) return false;
    
    return true;
}
```

**M0 Has**: Requirement for VRAM residency check.

**Missing in M0**: Concrete implementation using `cudaPointerGetAttributes`.

---

### ✅ **GAP-20: Process VRAM Usage Query**

**Old Specs Have** (`04_health.md`):
```cpp
uint64_t Health::get_process_vram_usage(const Context& ctx) {
    size_t free_bytes, total_bytes;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    
    if (err != cudaSuccess) return 0;
    
    return total_bytes - free_bytes;
}
```

**M0 Has**: Requirement for VRAM usage reporting.

**Missing in M0**: Implementation using `cudaMemGetInfo`.

---

### ✅ **GAP-21: NVML vs CUDA Runtime Clarification**

**Old Specs Have** (`04_health.md`):
```
NVML vs CUDA Runtime:
- CUDA Runtime: Can check VRAM residency, memory usage
- NVML: Required for temperature, power, utilization
- Decision: Worker uses CUDA Runtime only; pool manager uses NVML for system-wide monitoring
```

**M0 Has**: General health monitoring requirement.

**Missing in M0**: Clear delineation of CUDA Runtime vs NVML responsibilities.

---

## 6. Build System Details

### ✅ **GAP-22: CMake Configuration**

**Old Specs Have** (`00_cuda_overview.md`):
```cmake
cmake_minimum_required(VERSION 3.18)
project(worker_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Enable CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# Source files
add_library(worker_cuda STATIC
    src/ffi.cpp
    src/context.cpp
    src/model.cpp
    src/inference.cu
    src/health.cpp
    src/errors.cpp
    kernels/attention.cu
    kernels/matmul.cu
    kernels/sampling.cu
)

target_include_directories(worker_cuda PUBLIC include)
target_link_libraries(worker_cuda cudart)

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
```

**M0 Has**: Reference to CMake integration.

**Missing in M0**: Complete CMake configuration.

---

### ✅ **GAP-23: Cargo Build Integration**

**Old Specs Have** (`00_cuda_overview.md`):
```rust
// build.rs
fn main() {
    let dst = cmake::Config::new("cuda")
        .define("CMAKE_BUILD_TYPE", "Release")
        .build();
    
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=worker_cuda");
    println!("cargo:rustc-link-lib=cudart");
}
```

**M0 Has**: Reference to build.rs.

**Missing in M0**: Concrete build.rs implementation.

---

### ✅ **GAP-24: CUDA Architecture Targets**

**Old Specs Have**:
```cmake
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
```

**M0 Has**: General CUDA support requirement.

**Missing in M0**: Specific compute capability targets.

---

## 7. Testing Infrastructure

### ✅ **GAP-25: C++ Unit Test Examples**

**Old Specs Have** (`01_context.md`, `02_model.md`, `03_inference.md`):
```cpp
// tests/test_context.cpp
TEST(ContextTest, InitializeValidDevice) {
    int error_code;
    auto ctx = cuda_init(0, &error_code);
    ASSERT_NE(ctx, nullptr);
    ASSERT_EQ(error_code, CUDA_SUCCESS);
    cuda_destroy(ctx);
}

TEST(ContextTest, InitializeInvalidDevice) {
    int error_code;
    auto ctx = cuda_init(999, &error_code);
    ASSERT_EQ(ctx, nullptr);
    ASSERT_EQ(error_code, CUDA_ERROR_INVALID_DEVICE);
}

// tests/test_model.cpp
TEST(ModelTest, LoadValidModel) {
    int error_code;
    auto ctx = cuda_init(0, &error_code);
    ASSERT_NE(ctx, nullptr);
    
    uint64_t vram_bytes = 0;
    auto model = cuda_load_model(ctx, "test_model.gguf", &vram_bytes, &error_code);
    
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(error_code, CUDA_SUCCESS);
    ASSERT_GT(vram_bytes, 0);
    
    cuda_unload_model(model);
    cuda_destroy(ctx);
}

// tests/test_inference.cpp
TEST(InferenceTest, ReproducibleOutputForTesting) {
    auto output1 = run_inference("Hello", /*seed=*/42, /*temp=*/0.0f);
    auto output2 = run_inference("Hello", /*seed=*/42, /*temp=*/0.0f);
    ASSERT_EQ(output1, output2);
}
```

**M0 Has**: High-level test requirements.

**Missing in M0**: Concrete C++ test examples using Google Test.

---

### ✅ **GAP-26: Rust Integration Test Examples**

**Old Specs Have** (`00_cuda_overview.md`):
```rust
// tests/cuda_integration.rs
#[test]
fn test_load_model() {
    let ctx = unsafe { cuda_init(0, &mut 0) };
    assert!(!ctx.is_null());
    
    let mut vram_bytes = 0;
    let model = unsafe {
        cuda_load_model(ctx, c"model.gguf".as_ptr(), &mut vram_bytes, &mut 0)
    };
    assert!(!model.is_null());
    assert!(vram_bytes > 0);
}
```

**M0 Has**: High-level integration test requirements.

**Missing in M0**: Concrete Rust FFI test examples.

---

## 8. Kernel Implementation Guidance

### ✅ **GAP-27: Kernel Organization Structure**

**Old Specs Have** (`00_cuda_overview.md`):
```
kernels/
├── attention.cu      # Attention mechanism kernels
├── matmul.cu         # Matrix multiplication kernels
├── sampling.cu       # Token sampling kernels
├── rope.cu           # Rotary position embeddings
└── common.cuh        # Shared kernel utilities
```

**M0 Has**: List of required kernels.

**Missing in M0**: File organization structure.

---

### ✅ **GAP-28: Kernel Launch Pattern**

**Old Specs Have** (`00_cuda_overview.md`):
```cpp
void run_forward_pass(const Model& model, InferenceState& state) {
    dim3 grid(num_blocks);
    dim3 block(threads_per_block);
    attention_kernel<<<grid, block, 0, state.stream>>>(
        model.weights(),
        state.kv_cache(),
        state.output()
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw CudaError(err);
    }
}
```

**M0 Has**: Requirement for kernel execution.

**Missing in M0**: Standard kernel launch pattern.

---

### ✅ **GAP-29: Attention Kernel Skeleton**

**Old Specs Have** (`03_inference.md`):
```cuda
__global__ void attention_kernel(
    const float* weights,
    const float* input,
    float* kv_cache,
    int position,
    float* output
) {
    // Multi-head self-attention
    // - Query, Key, Value projections
    // - Scaled dot-product attention
    // - Output projection
    
    // Implementation details...
}
```

**M0 Has**: Requirement for attention kernel.

**Missing in M0**: Kernel signature and structure.

---

### ✅ **GAP-30: Softmax Kernel Skeleton**

**Old Specs Have** (`03_inference.md`):
```cuda
__global__ void softmax_kernel(float* logits, int vocab_size) {
    // Numerically stable softmax
    // - Find max
    // - Exp and sum
    // - Normalize
    
    // Implementation details...
}
```

**M0 Has**: Requirement for sampling.

**Missing in M0**: Softmax kernel structure.

---

## 9. Performance Optimization Guidance

### ✅ **GAP-31: Memory Coalescing Guidance**

**Old Specs Have** (`00_cuda_overview.md`):
```
Kernels MUST access global memory in coalesced patterns for optimal bandwidth.
```

**Missing in M0**: Explicit memory coalescing requirement.

---

### ✅ **GAP-32: Stream Parallelism Guidance**

**Old Specs Have** (`00_cuda_overview.md`):
```
Use CUDA streams for overlapping computation and memory transfers where possible.
```

**Missing in M0**: Stream parallelism guidance.

---

### ✅ **GAP-33: Kernel Fusion Guidance**

**Old Specs Have** (`00_cuda_overview.md`, `03_inference.md`):
```
Fuse small kernels to reduce launch overhead:
- Combine RoPE + attention
- Combine LayerNorm + FFN
```

**M0 Has**: General performance requirements.

**Missing in M0**: Specific kernel fusion opportunities.

---

## 10. Conflicts & Contradictions

### ⚠️ **CONFLICT-1: Architecture Detection**

**Old Specs** (`02_model.md`):
- No mention of architecture detection from GGUF metadata
- Assumes single architecture (Llama-style)

**M0 Spec** (M0-W-1212):
- **MUST** detect architecture from GGUF `general.architecture` field
- Support both Llama and GPT architectures
- Use ModelAdapter pattern

**Resolution**: M0 spec is correct. Old specs predate multi-architecture support.

---

### ⚠️ **CONFLICT-2: Tokenization**

**Old Specs**:
- No tokenization details (assumes external tokenizer)

**M0 Spec** (§8):
- **MUST** implement two tokenizer backends: `hf-json` and `gguf-bpe`
- Pure Rust implementation
- No external dependencies

**Resolution**: M0 spec is correct. Old specs predate tokenization requirements.

---

### ⚠️ **CONFLICT-3: MXFP4 Support**

**Old Specs**:
- No mention of MXFP4 quantization
- Only references generic quantization types (Q8_0, Q4_0)

**M0 Spec** (M0-W-1201):
- **MUST** support MXFP4 for GPT-OSS-20B
- In-kernel dequantization
- Architecture-aware weight mapping

**Resolution**: M0 spec is correct. Old specs predate MXFP4 requirements.

---

### ⚠️ **CONFLICT-4: Test Reproducibility Scope**

**Old Specs** (`03_inference.md`):
- Presents reproducibility as a general feature
- No distinction between testing and production

**M0 Spec** (§3):
- **Clarifies**: Reproducibility is for **TESTING ONLY**
- Temperature 0.0-2.0 is the product feature
- NOT a product guarantee

**Resolution**: M0 spec is correct. Old specs lacked this critical clarification.

---

## 11. Overlaps (Old Specs Have More Detail)

### 🔄 **OVERLAP-1: VRAM-Only Enforcement**

**Both Cover**: VRAM-only policy

**Old Specs Add** (`01_context.md`):
```cpp
// Enforce VRAM-only mode
cudaDeviceSetLimit(cudaLimitMallocHeapSize, 0);  // Disable UMA
cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
```

**Value**: Concrete CUDA API calls.

---

### 🔄 **OVERLAP-2: Error Handling**

**Both Cover**: Error codes and handling

**Old Specs Add** (`00_cuda_overview.md`):
- Complete error code enum
- Error message retrieval function
- Exception-to-error-code conversion pattern

**Value**: Implementation-ready error handling.

---

### 🔄 **OVERLAP-3: Model Loading**

**Both Cover**: GGUF loading, mmap, chunked transfer

**Old Specs Add** (`02_model.md`):
- Complete parsing implementation
- Tensor structure definitions
- Memory layout diagrams

**Value**: Implementation-ready model loader.

---

## 12. Recommendations

### ✅ **Keep and Update Old Specs**

**Rationale**:
1. Old specs provide **implementation-level details** missing in M0
2. M0 spec focuses on **requirements and architecture**
3. Old specs serve as **implementation guides** for developers

**Action Items**:
1. **Update old specs** to align with M0 requirements:
   - Add architecture detection (M0-W-1212)
   - Add ModelAdapter pattern (M0-W-1213-1215)
   - Add MXFP4 support (M0-W-1201, M0-W-1435)
   - Add tokenization references (§8)
   - Clarify test reproducibility scope (§3)

2. **Add cross-references**:
   - Old specs should reference M0 spec IDs
   - M0 spec should reference old specs for implementation details

3. **Create implementation index**:
   - `00_cuda_overview.md` → Architecture & patterns
   - `01_context.md` → Context management implementation
   - `02_model.md` → Model loading implementation
   - `03_inference.md` → Inference execution implementation
   - `04_health.md` → Health monitoring implementation

---

### 📋 **Update Checklist for Old Specs**

#### `00_cuda_overview.md`
- [ ] Add ModelAdapter factory pattern
- [ ] Add architecture detection overview
- [ ] Add MXFP4 kernel references
- [ ] Update kernel organization to include GPT-specific kernels

#### `01_context.md`
- [ ] No changes needed (still accurate)

#### `02_model.md`
- [ ] Add architecture detection from GGUF metadata
- [ ] Add ModelAdapter selection logic
- [ ] Add MXFP4 tensor type handling
- [ ] Add weight mapping for both Llama and GPT architectures

#### `03_inference.md`
- [ ] Add ModelAdapter usage
- [ ] Add architecture-specific forward pass references
- [ ] Add MXFP4 dequantization details
- [ ] Clarify test reproducibility (NOT a product guarantee)
- [ ] Add tokenization integration points

#### `04_health.md`
- [ ] Add `quant_kind` field (MXFP4/Q4_K_M/Q4_0)
- [ ] Add `tokenizer_kind` field (hf-json/gguf-bpe)
- [ ] Add `vocab_size` field
- [ ] No other changes needed

---

## 13. Summary

### Gaps Found: **33 implementation details missing in M0**
### Conflicts: **4 contradictions (M0 is correct in all cases)**
### Overlaps: **3 areas where old specs add value**

### Final Verdict: **RETAIN AND UPDATE OLD SPECS**

The old CUDA specs are **valuable implementation guides** that complement the M0 leading spec. They should be:
1. **Updated** to align with M0 requirements (architecture adapters, MXFP4, tokenization)
2. **Cross-referenced** with M0 spec IDs
3. **Maintained** as implementation documentation

**Next Steps**:
1. Update old specs per checklist above
2. Add "Implementation Guide" header to each old spec
3. Reference M0 spec as the leading architectural document
4. Keep old specs as detailed implementation references

---

**End of Gap Analysis**
