# Worker CUDA Implementation

**Language**: C++17 + CUDA  
**Purpose**: GPU operations for worker-orcd binary  
**FFI**: Exposes C API to Rust layer

---

## Directory Structure

```
cuda/
├── README.md                    # This file
├── CMakeLists.txt              # CMake build configuration
├── .specs/
│   ├── 00_cuda_overview.md     # Architecture overview
│   ├── 01_context.md           # Context management spec
│   ├── 02_model.md             # Model loading spec
│   ├── 03_inference.md         # Inference execution spec
│   └── 04_health.md            # Health monitoring spec
├── include/
│   ├── worker_cuda.h           # Public C API (FFI boundary)
│   ├── context.hpp             # Context management (C++)
│   ├── model.hpp               # Model operations (C++)
│   ├── inference.hpp           # Inference operations (C++)
│   ├── health.hpp              # Health checks (C++)
│   ├── errors.hpp              # Error codes and handling
│   └── types.hpp               # Common types
├── src/
│   ├── ffi.cpp                 # C API implementation
│   ├── context.cpp             # Context management
│   ├── model.cpp               # Model loading
│   ├── inference.cu            # Inference kernels (CUDA)
│   ├── health.cpp              # Health monitoring
│   ├── errors.cpp              # Error handling
│   └── utils.cpp               # Utility functions
├── kernels/
│   ├── attention.cu            # Attention kernels
│   ├── matmul.cu               # Matrix multiplication
│   ├── sampling.cu             # Token sampling
│   └── common.cuh              # Common kernel utilities
├── tests/
│   ├── test_context.cpp        # Context tests
│   ├── test_model.cpp          # Model loading tests
│   ├── test_inference.cpp      # Inference tests
│   └── test_health.cpp         # Health check tests
└── vendor/
    └── ggml/                   # GGML library (if needed)
```

---

## Build System

Uses **CMake** for C++/CUDA compilation, integrated with Cargo via `build.rs`.

### CMake Configuration
```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(worker_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Source files
add_library(worker_cuda STATIC
    src/ffi.cpp
    src/context.cpp
    src/model.cpp
    src/inference.cu
    src/health.cpp
    src/errors.cpp
    src/utils.cpp
    kernels/attention.cu
    kernels/matmul.cu
    kernels/sampling.cu
)

target_include_directories(worker_cuda PUBLIC include)
target_link_libraries(worker_cuda cudart)
```

### Rust Build Integration
```rust
// build.rs (in worker-orcd root)
use std::env;
use std::path::PathBuf;

fn main() {
    let cuda_dir = PathBuf::from("cuda");
    
    // Build CUDA library with CMake
    let dst = cmake::Config::new(&cuda_dir)
        .define("CMAKE_BUILD_TYPE", "Release")
        .build();
    
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=worker_cuda");
    println!("cargo:rustc-link-lib=cudart");
    
    // Rebuild if CUDA sources change
    println!("cargo:rerun-if-changed=cuda/src");
    println!("cargo:rerun-if-changed=cuda/include");
    println!("cargo:rerun-if-changed=cuda/kernels");
}
```

---

## Module Organization

### Context Management (`context.cpp`)
- Initialize CUDA device
- Create and manage CUDA context
- Set VRAM-only enforcement flags
- Device capability queries

### Model Loading (`model.cpp`)
- Parse GGUF format
- Allocate VRAM for weights
- Copy model to GPU
- Model metadata extraction

### Inference (`inference.cu`)
- Token generation loop
- KV cache management
- Kernel orchestration
- Sampling strategies

### Health Monitoring (`health.cpp`)
- VRAM residency verification
- Memory leak detection
- GPU error checking

---

## FFI Boundary

### C API (worker_cuda.h)
Exposes pure C functions for Rust FFI:
```c
extern "C" {
    CudaContext* cuda_init(int gpu_device, int* error_code);
    CudaModel* cuda_load_model(CudaContext* ctx, const char* path, ...);
    // ... etc
}
```

### C++ Implementation
Internal C++ classes with RAII, exceptions, etc:
```cpp
namespace worker_cuda {
    class Context { ... };
    class Model { ... };
    class Inference { ... };
}
```

---

## Testing

### Unit Tests (Google Test)
```bash
cd cuda
mkdir build && cd build
cmake -DBUILD_TESTING=ON ..
make
ctest
```

### Integration Tests
Rust integration tests call into CUDA via FFI.

---

## Dependencies

- **CUDA Toolkit**: 11.8+ (for CUDA 11.x) or 12.x
- **CMake**: 3.18+
- **C++ Compiler**: GCC 9+ or Clang 10+
- **GGML** (optional): For GGUF parsing

---

## Development Workflow

1. **Edit CUDA code**: Modify `.cpp`, `.cu`, `.hpp` files
2. **Build**: `cargo build` (runs CMake automatically)
3. **Test CUDA**: `cd cuda/build && ctest`
4. **Test Rust**: `cargo test`

---

## Specs

See `.specs/` directory for detailed specifications of each module.
