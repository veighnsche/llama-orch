# CUDA Integration Guide

**Audience**: worker-orcd developers  
**Status**: Normative for Rust ↔ C++/CUDA FFI

---

## Overview

`worker-orcd` is a **hybrid binary** combining:
- **Rust**: HTTP/RPC server, lifecycle, observability
- **C++/CUDA**: Model loading, inference kernels, GPU memory

This document defines the integration pattern, build process, and FFI contract.

---

## Directory Layout

```
bin/worker-orcd/
├── Cargo.toml
├── build.rs                    ← Compiles CUDA code
├── README.md
├── src/                        ← Rust implementation
│   ├── main.rs
│   ├── server.rs               ← HTTP/RPC server
│   ├── worker.rs               ← Worker lifecycle
│   ├── ffi.rs                  ← FFI bindings to CUDA
│   └── config.rs
├── cuda/                       ← C++/CUDA implementation
│   ├── CMakeLists.txt          ← CUDA build config
│   ├── README.md               ← CUDA-specific docs
│   ├── include/
│   │   ├── inference.h         ← Public C API
│   │   ├── model_loader.h
│   │   └── types.h             ← Shared types
│   └── src/
│       ├── inference.cu        ← Inference kernels
│       ├── model_loader.cu     ← Model loading
│       ├── memory.cu           ← GPU memory management
│       └── kernels/
│           ├── attention.cu
│           └── sampling.cu
└── tests/
    ├── ffi_smoke.rs            ← FFI boundary tests
    └── integration.rs
```

---

## Language Boundary

### Rust Responsibilities

✅ **HTTP/RPC server**: Axum-based API for pool-managerd  
✅ **Lifecycle management**: Spawn, readiness callback, shutdown  
✅ **Configuration**: Parse config, validate, pass to CUDA  
✅ **Observability**: Metrics (Prometheus), structured logging (tracing)  
✅ **Error handling**: Convert CUDA errors to Rust `Result`  
✅ **Memory safety**: Ensure CUDA resources are freed (RAII wrappers)

### C++/CUDA Responsibilities

✅ **Model loading**: Parse GGUF, allocate VRAM, load weights  
✅ **Inference kernels**: Attention, MLP, sampling, KV cache  
✅ **GPU memory**: cudaMalloc, cudaFree, VRAM tracking  
✅ **Determinism**: Seeded RNG, reproducible sampling  
✅ **Performance**: Optimized kernels, batching, streams

---

## FFI Contract

### C API (cuda/include/inference.h)

```c
#ifndef WORKER_ORCD_INFERENCE_H
#define WORKER_ORCD_INFERENCE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to model instance
typedef struct WorkerModel WorkerModel;

// Error codes
typedef enum {
    WORKER_OK = 0,
    WORKER_ERR_CUDA = 1,
    WORKER_ERR_OOM = 2,
    WORKER_ERR_INVALID_MODEL = 3,
    WORKER_ERR_INFERENCE_FAILED = 4,
} WorkerError;

// Model configuration
typedef struct {
    const char* model_path;
    uint32_t gpu_device;
    uint32_t slots_total;
    uint32_t context_size;
    bool use_mmap;
} WorkerModelConfig;

// Inference request
typedef struct {
    const int32_t* tokens;
    size_t tokens_len;
    uint32_t max_tokens;
    uint32_t seed;
    float temperature;
    float top_p;
} WorkerInferenceRequest;

// Inference response (single token)
typedef struct {
    int32_t token;
    float logit;
    bool is_eos;
} WorkerInferenceResponse;

// API functions
WorkerError worker_model_load(
    const WorkerModelConfig* config,
    WorkerModel** out_model,
    uint64_t* out_vram_bytes
);

WorkerError worker_model_infer(
    WorkerModel* model,
    const WorkerInferenceRequest* request,
    WorkerInferenceResponse* response
);

WorkerError worker_model_free(WorkerModel* model);

const char* worker_error_string(WorkerError error);

#ifdef __cplusplus
}
#endif

#endif // WORKER_ORCD_INFERENCE_H
```

### Rust FFI Bindings (src/ffi.rs)

```rust
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;

// Re-export C types
#[repr(C)]
pub struct WorkerModel {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum WorkerError {
    Ok = 0,
    Cuda = 1,
    Oom = 2,
    InvalidModel = 3,
    InferenceFailed = 4,
}

#[repr(C)]
pub struct WorkerModelConfig {
    pub model_path: *const c_char,
    pub gpu_device: u32,
    pub slots_total: u32,
    pub context_size: u32,
    pub use_mmap: bool,
}

#[repr(C)]
pub struct WorkerInferenceRequest {
    pub tokens: *const i32,
    pub tokens_len: usize,
    pub max_tokens: u32,
    pub seed: u32,
    pub temperature: f32,
    pub top_p: f32,
}

#[repr(C)]
pub struct WorkerInferenceResponse {
    pub token: i32,
    pub logit: f32,
    pub is_eos: bool,
}

extern "C" {
    pub fn worker_model_load(
        config: *const WorkerModelConfig,
        out_model: *mut *mut WorkerModel,
        out_vram_bytes: *mut u64,
    ) -> WorkerError;

    pub fn worker_model_infer(
        model: *mut WorkerModel,
        request: *const WorkerInferenceRequest,
        response: *mut WorkerInferenceResponse,
    ) -> WorkerError;

    pub fn worker_model_free(model: *mut WorkerModel) -> WorkerError;

    pub fn worker_error_string(error: WorkerError) -> *const c_char;
}

// Safe Rust wrapper
pub struct Model {
    handle: *mut WorkerModel,
}

impl Model {
    pub fn load(config: &ModelConfig) -> Result<(Self, u64), String> {
        let c_path = CString::new(config.model_path.as_str())
            .map_err(|e| format!("Invalid path: {}", e))?;

        let c_config = WorkerModelConfig {
            model_path: c_path.as_ptr(),
            gpu_device: config.gpu_device,
            slots_total: config.slots_total,
            context_size: config.context_size,
            use_mmap: config.use_mmap,
        };

        let mut handle = ptr::null_mut();
        let mut vram_bytes = 0u64;

        let err = unsafe {
            worker_model_load(&c_config, &mut handle, &mut vram_bytes)
        };

        if err != WorkerError::Ok {
            return Err(error_to_string(err));
        }

        Ok((Self { handle }, vram_bytes))
    }

    pub fn infer(&mut self, request: &InferenceRequest) -> Result<InferenceResponse, String> {
        let c_request = WorkerInferenceRequest {
            tokens: request.tokens.as_ptr(),
            tokens_len: request.tokens.len(),
            max_tokens: request.max_tokens,
            seed: request.seed,
            temperature: request.temperature,
            top_p: request.top_p,
        };

        let mut c_response = WorkerInferenceResponse {
            token: 0,
            logit: 0.0,
            is_eos: false,
        };

        let err = unsafe {
            worker_model_infer(self.handle, &c_request, &mut c_response)
        };

        if err != WorkerError::Ok {
            return Err(error_to_string(err));
        }

        Ok(InferenceResponse {
            token: c_response.token,
            logit: c_response.logit,
            is_eos: c_response.is_eos,
        })
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                worker_model_free(self.handle);
            }
        }
    }
}

unsafe impl Send for Model {}
// NOT Sync - CUDA contexts are thread-local

fn error_to_string(err: WorkerError) -> String {
    unsafe {
        let c_str = worker_error_string(err);
        CStr::from_ptr(c_str).to_string_lossy().into_owned()
    }
}

// High-level Rust types
pub struct ModelConfig {
    pub model_path: String,
    pub gpu_device: u32,
    pub slots_total: u32,
    pub context_size: u32,
    pub use_mmap: bool,
}

pub struct InferenceRequest {
    pub tokens: Vec<i32>,
    pub max_tokens: u32,
    pub seed: u32,
    pub temperature: f32,
    pub top_p: f32,
}

pub struct InferenceResponse {
    pub token: i32,
    pub logit: f32,
    pub is_eos: bool,
}
```

---

## Build Process

### build.rs (Rust build script)

```rust
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let cuda_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("cuda");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Build CUDA code with CMake
    let status = Command::new("cmake")
        .arg("-S")
        .arg(&cuda_dir)
        .arg("-B")
        .arg(&out_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .status()
        .expect("Failed to run cmake");

    assert!(status.success(), "CMake configuration failed");

    let status = Command::new("cmake")
        .arg("--build")
        .arg(&out_dir)
        .arg("--config")
        .arg("Release")
        .status()
        .expect("Failed to build CUDA code");

    assert!(status.success(), "CMake build failed");

    // Link the static library
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=worker_cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");

    // Rerun if CUDA sources change
    println!("cargo:rerun-if-changed=cuda/");
}
```

### CMakeLists.txt (CUDA build config)

```cmake
cmake_minimum_required(VERSION 3.18)
project(worker_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Source files
set(SOURCES
    src/inference.cu
    src/model_loader.cu
    src/memory.cu
    src/kernels/attention.cu
    src/kernels/sampling.cu
)

# Static library
add_library(worker_cuda STATIC ${SOURCES})

target_include_directories(worker_cuda
    PUBLIC include
    PRIVATE src
)

target_link_libraries(worker_cuda
    CUDA::cudart
    CUDA::cublas
)

# Compiler flags
target_compile_options(worker_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        --generate-line-info
    >
)

# Set CUDA architectures (adjust for target GPUs)
set_target_properties(worker_cuda PROPERTIES
    CUDA_ARCHITECTURES "75;80;86;89"  # Turing, Ampere, Ada
)
```

---

## Error Handling

### CUDA → Rust Error Propagation

```rust
// src/worker.rs
use crate::ffi::{Model, ModelConfig};

pub struct Worker {
    model: Model,
    vram_bytes: u64,
}

impl Worker {
    pub async fn new(config: WorkerConfig) -> Result<Self, WorkerError> {
        let model_config = ModelConfig {
            model_path: config.model_path.clone(),
            gpu_device: config.gpu_device,
            slots_total: config.slots_total,
            context_size: 4096,
            use_mmap: true,
        };

        let (model, vram_bytes) = Model::load(&model_config)
            .map_err(|e| WorkerError::ModelLoadFailed(e))?;

        tracing::info!(
            vram_bytes = vram_bytes,
            model_path = %config.model_path,
            "Model loaded successfully"
        );

        Ok(Self { model, vram_bytes })
    }

    pub async fn infer(&mut self, request: InferenceRequest) -> Result<Vec<i32>, WorkerError> {
        let mut tokens = Vec::new();

        loop {
            let response = self.model.infer(&request)
                .map_err(|e| WorkerError::InferenceFailed(e))?;

            tokens.push(response.token);

            if response.is_eos || tokens.len() >= request.max_tokens as usize {
                break;
            }
        }

        Ok(tokens)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error("Model load failed: {0}")]
    ModelLoadFailed(String),

    #[error("Inference failed: {0}")]
    InferenceFailed(String),
}
```

---

## Testing

### FFI Smoke Test

```rust
// tests/ffi_smoke.rs
use worker_orcd::ffi::{Model, ModelConfig};

#[test]
fn test_ffi_model_load() {
    let model_path = std::env::var("LLORCH_TEST_MODEL_PATH")
        .unwrap_or_else(|_| "../../.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf".into());

    let config = ModelConfig {
        model_path,
        gpu_device: 0,
        slots_total: 1,
        context_size: 2048,
        use_mmap: true,
    };

    let (model, vram_bytes) = Model::load(&config).expect("Model load failed");
    assert!(vram_bytes > 0);
    assert!(vram_bytes < 2_000_000_000); // < 2GB for Qwen-0.5B
}
```

---

## Best Practices

### DO

✅ **Keep FFI surface minimal**: Only expose what Rust needs  
✅ **Use C ABI**: `extern "C"` for all public functions  
✅ **Opaque handles**: Never expose C++ classes directly  
✅ **Error codes**: Return enum, not exceptions  
✅ **RAII in Rust**: Wrap handles in `Drop` types  
✅ **Document ownership**: Who allocates? Who frees?  
✅ **Test FFI boundary**: Dedicated smoke tests  

### DON'T

❌ **Pass std::string**: Use `const char*`  
❌ **Pass std::vector**: Use `const T*, size_t`  
❌ **Throw exceptions**: C++ exceptions don't cross FFI  
❌ **Assume thread-safety**: CUDA contexts are thread-local  
❌ **Leak memory**: Every `load` needs a `free`  
❌ **Use C++ types in headers**: Stick to C types  

---

## References

- **Rust FFI Guide**: https://doc.rust-lang.org/nomicon/ffi.html
- **CUDA Best Practices**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **cudarc** (Rust CUDA bindings): https://github.com/coreylowman/cudarc
- **bindgen** (auto-generate bindings): https://rust-lang.github.io/rust-bindgen/

---

## Related

- **Worker README**: `../README.md`
- **Test Models**: `../../.docs/testing/TEST_MODELS.md`
- **Build System**: `../build.rs`, `../cuda/CMakeLists.txt`

---

**Status**: Normative for worker-orcd CUDA integration.
