# FT-007: Rust FFI Bindings

**Team**: Foundation-Alpha  
**Sprint**: Sprint 2 - FFI Layer  
**Size**: M (2 days)  
**Days**: 12 - 13  
**Spec Ref**: M0-W-1050, CUDA-4011

---

## Story Description

Implement Rust FFI bindings for C API defined in FT-006. This provides safe Rust wrappers around unsafe FFI calls with proper error handling and resource management.

---

## Acceptance Criteria

- [ ] Rust bindings for all FFI functions from worker_ffi.h
- [ ] Safe wrapper types: `CudaContext`, `CudaModel`, `InferenceResult`
- [ ] RAII pattern: resources freed automatically on drop
- [ ] Error codes converted to Rust `Result<T, CudaError>`
- [ ] All FFI calls are `unsafe` blocks with safety comments
- [ ] Unit tests validate bindings with mock C functions
- [ ] Integration tests validate bindings with real CUDA (if available)
- [ ] Documentation for every public function
- [ ] No memory leaks (verified with valgrind or similar)

---

## Dependencies

### Upstream (Blocks This Story)
- **CRITICAL**: FT-006 FFI interface definition (Expected completion: Day 11)

### Downstream (This Story Blocks)
- FT-010: CUDA context init needs Rust bindings
- FT-024: HTTP-FFI-CUDA integration test needs bindings

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/src/cuda/ffi.rs` - Raw FFI declarations
- `bin/worker-orcd/src/cuda/context.rs` - CudaContext wrapper
- `bin/worker-orcd/src/cuda/model.rs` - CudaModel wrapper
- `bin/worker-orcd/src/cuda/inference.rs` - InferenceResult wrapper
- `bin/worker-orcd/src/cuda/error.rs` - Error types and conversion
- `bin/worker-orcd/src/cuda/mod.rs` - CUDA module exports

### Key Interfaces
```rust
use std::ffi::{CStr, CString};
use std::ptr;

// Raw FFI declarations (unsafe)
mod ffi {
    use std::os::raw::{c_char, c_int};
    
    #[repr(C)]
    pub struct CudaContext {
        _private: [u8; 0],
    }
    
    #[repr(C)]
    pub struct CudaModel {
        _private: [u8; 0],
    }
    
    #[repr(C)]
    pub struct InferenceResult {
        _private: [u8; 0],
    }
    
    extern "C" {
        pub fn cuda_init(gpu_device: c_int, error_code: *mut c_int) -> *mut CudaContext;
        pub fn cuda_destroy(ctx: *mut CudaContext);
        pub fn cuda_get_device_count() -> c_int;
        
        pub fn cuda_load_model(
            ctx: *mut CudaContext,
            model_path: *const c_char,
            vram_bytes_used: *mut u64,
            error_code: *mut c_int,
        ) -> *mut CudaModel;
        pub fn cuda_unload_model(model: *mut CudaModel);
        pub fn cuda_model_get_vram_usage(model: *mut CudaModel) -> u64;
        
        pub fn cuda_inference_start(
            model: *mut CudaModel,
            prompt: *const c_char,
            max_tokens: c_int,
            temperature: f32,
            seed: u64,
            error_code: *mut c_int,
        ) -> *mut InferenceResult;
        pub fn cuda_inference_next_token(
            result: *mut InferenceResult,
            token_out: *mut c_char,
            token_buffer_size: c_int,
            token_index: *mut c_int,
            error_code: *mut c_int,
        ) -> bool;
        pub fn cuda_inference_free(result: *mut InferenceResult);
        
        pub fn cuda_check_vram_residency(model: *mut CudaModel, error_code: *mut c_int) -> bool;
        pub fn cuda_get_vram_usage(model: *mut CudaModel) -> u64;
        pub fn cuda_get_process_vram_usage(ctx: *mut CudaContext) -> u64;
        pub fn cuda_check_device_health(ctx: *mut CudaContext, error_code: *mut c_int) -> bool;
        
        pub fn cuda_error_message(error_code: c_int) -> *const c_char;
    }
}

// Safe Rust wrappers
pub struct CudaContext {
    ptr: *mut ffi::CudaContext,
}

impl CudaContext {
    pub fn new(gpu_device: i32) -> Result<Self, CudaError> {
        let mut error_code = 0;
        let ptr = unsafe { ffi::cuda_init(gpu_device, &mut error_code) };
        
        if ptr.is_null() {
            return Err(CudaError::from_code(error_code));
        }
        
        Ok(Self { ptr })
    }
    
    pub fn device_count() -> i32 {
        unsafe { ffi::cuda_get_device_count() }
    }
    
    pub fn load_model(&self, model_path: &str) -> Result<CudaModel, CudaError> {
        let path_cstr = CString::new(model_path)
            .map_err(|_| CudaError::InvalidParameter("model_path contains null byte".into()))?;
        
        let mut vram_bytes = 0;
        let mut error_code = 0;
        
        // SAFETY: ptr is valid (checked in constructor), path_cstr is valid CString
        let model_ptr = unsafe {
            ffi::cuda_load_model(
                self.ptr,
                path_cstr.as_ptr(),
                &mut vram_bytes,
                &mut error_code,
            )
        };
        
        if model_ptr.is_null() {
            return Err(CudaError::from_code(error_code));
        }
        
        Ok(CudaModel {
            ptr: model_ptr,
            vram_bytes,
        })
    }
    
    pub fn process_vram_usage(&self) -> u64 {
        // SAFETY: ptr is valid (checked in constructor)
        unsafe { ffi::cuda_get_process_vram_usage(self.ptr) }
    }
    
    pub fn check_device_health(&self) -> Result<bool, CudaError> {
        let mut error_code = 0;
        // SAFETY: ptr is valid (checked in constructor)
        let healthy = unsafe { ffi::cuda_check_device_health(self.ptr, &mut error_code) };
        
        if error_code != 0 {
            return Err(CudaError::from_code(error_code));
        }
        
        Ok(healthy)
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr is valid and only dropped once
            unsafe { ffi::cuda_destroy(self.ptr) };
        }
    }
}

// Send + Sync: Each context is single-threaded, but can be moved between threads
unsafe impl Send for CudaContext {}

pub struct CudaModel {
    ptr: *mut ffi::CudaModel,
    vram_bytes: u64,
}

impl CudaModel {
    pub fn vram_usage(&self) -> u64 {
        self.vram_bytes
    }
    
    pub fn check_vram_residency(&self) -> Result<bool, CudaError> {
        let mut error_code = 0;
        // SAFETY: ptr is valid (checked in constructor)
        let resident = unsafe { ffi::cuda_check_vram_residency(self.ptr, &mut error_code) };
        
        if error_code != 0 {
            return Err(CudaError::from_code(error_code));
        }
        
        Ok(resident)
    }
    
    pub fn start_inference(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        seed: u64,
    ) -> Result<InferenceResult, CudaError> {
        let prompt_cstr = CString::new(prompt)
            .map_err(|_| CudaError::InvalidParameter("prompt contains null byte".into()))?;
        
        let mut error_code = 0;
        
        // SAFETY: ptr is valid, prompt_cstr is valid CString
        let result_ptr = unsafe {
            ffi::cuda_inference_start(
                self.ptr,
                prompt_cstr.as_ptr(),
                max_tokens as i32,
                temperature,
                seed,
                &mut error_code,
            )
        };
        
        if result_ptr.is_null() {
            return Err(CudaError::from_code(error_code));
        }
        
        Ok(InferenceResult { ptr: result_ptr })
    }
}

impl Drop for CudaModel {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr is valid and only dropped once
            unsafe { ffi::cuda_unload_model(self.ptr) };
        }
    }
}

unsafe impl Send for CudaModel {}

pub struct InferenceResult {
    ptr: *mut ffi::InferenceResult,
}

impl InferenceResult {
    pub fn next_token(&mut self) -> Result<Option<(String, u32)>, CudaError> {
        let mut token_buffer = vec![0u8; 256];
        let mut token_index = 0;
        let mut error_code = 0;
        
        // SAFETY: ptr is valid, token_buffer is valid
        let has_token = unsafe {
            ffi::cuda_inference_next_token(
                self.ptr,
                token_buffer.as_mut_ptr() as *mut i8,
                token_buffer.len() as i32,
                &mut token_index,
                &mut error_code,
            )
        };
        
        if error_code != 0 {
            return Err(CudaError::from_code(error_code));
        }
        
        if !has_token {
            return Ok(None);
        }
        
        // Find null terminator
        let null_pos = token_buffer.iter().position(|&b| b == 0).unwrap_or(token_buffer.len());
        let token_str = String::from_utf8_lossy(&token_buffer[..null_pos]).into_owned();
        
        Ok(Some((token_str, token_index as u32)))
    }
}

impl Drop for InferenceResult {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr is valid and only dropped once
            unsafe { ffi::cuda_inference_free(self.ptr) };
        }
    }
}

unsafe impl Send for InferenceResult {}

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("Invalid device: {0}")]
    InvalidDevice(String),
    
    #[error("Out of memory: {0}")]
    OutOfMemory(String),
    
    #[error("Model load failed: {0}")]
    ModelLoadFailed(String),
    
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
    
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Kernel launch failed: {0}")]
    KernelLaunchFailed(String),
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl CudaError {
    fn from_code(code: i32) -> Self {
        // SAFETY: cuda_error_message returns static string
        let msg_ptr = unsafe { ffi::cuda_error_message(code) };
        let msg = if msg_ptr.is_null() {
            "Unknown error".to_string()
        } else {
            unsafe { CStr::from_ptr(msg_ptr) }
                .to_string_lossy()
                .into_owned()
        };
        
        match code {
            1 => Self::InvalidDevice(msg),
            2 => Self::OutOfMemory(msg),
            3 | 4 => Self::ModelLoadFailed(msg),
            5 => Self::InferenceFailed(msg),
            6 => Self::InvalidParameter(msg),
            7 => Self::KernelLaunchFailed(msg),
            _ => Self::Unknown(msg),
        }
    }
}
```

### Implementation Notes
- All FFI calls wrapped in `unsafe` blocks with safety comments
- RAII pattern: Drop implementations free C++ resources
- Error codes converted to typed Rust errors with `thiserror`
- CString validation prevents null bytes in strings
- Buffer sizes validated before FFI calls
- No panics in FFI boundary (return Result instead)
- Send trait implemented carefully (contexts are single-threaded)
- Opaque pointer types prevent Rust from dereferencing C++ internals

---

## Testing Strategy

### Unit Tests
- Test CudaContext::new() with valid device ID
- Test CudaContext::new() with invalid device ID returns error
- Test CudaContext::device_count() returns positive number
- Test CudaModel::vram_usage() returns non-zero
- Test InferenceResult::next_token() returns tokens
- Test error code conversion to CudaError variants
- Test Drop implementations (use miri or valgrind)

### Integration Tests
- Test full inference flow: init → load → infer → next_token → drop
- Test VRAM residency check
- Test process VRAM usage query
- Test device health check
- Test error propagation from C++ to Rust
- Test resource cleanup on panic (use catch_unwind)

### Manual Verification
1. Build with CUDA feature: `cargo build --features cuda`
2. Run unit tests: `cargo test --features cuda`
3. Run integration tests: `cargo test --features cuda --test integration`
4. Check for memory leaks: `valgrind --leak-check=full ./target/debug/worker-orcd`

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (7+ tests)
- [ ] Integration tests passing (6+ tests)
- [ ] Documentation updated (module docs, safety comments)
- [ ] No memory leaks (verified with valgrind)
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §4.2 Rust Layer Responsibilities (M0-W-1050)
- Spec: `bin/.specs/01_M0_worker_orcd.md` §4.2 FFI Boundaries (CUDA-4011)
- Related Stories: FT-006 (FFI interface), FT-010 (CUDA context init)
- Rust FFI: https://doc.rust-lang.org/nomicon/ffi.html

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋

---

## 🎀 Narration Opportunities

**From**: Narration-Core Team

### Events to Narrate

1. **FFI call failure** (with error code)
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: "ffi_call",
       target: function_name.to_string(),
       error_kind: Some(error.code().to_string()),
       human: format!("FFI call to {} failed: {}", function_name, error),
       ..Default::default()
   });
   ```

2. **Resource cleanup on drop**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: "resource_cleanup",
       target: resource_type.to_string(),
       human: format!("Cleaning up {} resource", resource_type),
       ..Default::default()
   });
   ```

**Why this matters**: FFI boundary is where Rust and C++ meet. Narration helps diagnose FFI-related issues like null pointers, memory leaks, and error propagation.

---
*Narration guidance added by Narration-Core Team 🎀*

---

## 🔍 Testing Team Requirements

**From**: Testing Team (Pre-Development Audit)

### Unit Testing Requirements
- **Test CudaContext::new() with valid device ID** (happy path)
- **Test CudaContext::new() with invalid device ID returns error** (error handling)
- **Test CudaContext::device_count() returns positive number** (device query)
- **Test CudaModel::vram_usage() returns non-zero** (VRAM tracking)
- **Test InferenceResult::next_token() returns tokens** (token generation)
- **Test error code conversion to CudaError variants** (all error codes)
- **Test Drop implementations** (use miri or valgrind)
- **Test CString validation prevents null bytes** (safety check)
- **Property test**: All valid device IDs accepted

### Integration Testing Requirements
- **Test full inference flow: init → load → infer → next_token → drop** (end-to-end)
- **Test VRAM residency check** (model loaded in VRAM)
- **Test process VRAM usage query** (total VRAM used)
- **Test device health check** (GPU status)
- **Test error propagation from C++ to Rust** (exception conversion)
- **Test resource cleanup on panic** (use catch_unwind)
- **Test concurrent contexts on different devices** (multi-GPU if available)

### BDD Testing Requirements (VERY IMPORTANT)
- **Scenario**: CUDA context initialization
  - Given a valid GPU device ID
  - When I create a CudaContext
  - Then the context should be initialized successfully
  - And device properties should be accessible
- **Scenario**: Model loading
  - Given a CudaContext and valid model path
  - When I load the model
  - Then the model should be loaded to VRAM
  - And VRAM usage should be reported
- **Scenario**: Inference execution
  - Given a loaded model
  - When I start inference with a prompt
  - Then I should receive an InferenceResult
  - And next_token() should return tokens
- **Scenario**: Error handling
  - Given an invalid device ID
  - When I create a CudaContext
  - Then I should receive CudaError::InvalidDevice
  - And error message should be descriptive

### Critical Paths to Test
- FFI call safety (unsafe blocks with safety comments)
- RAII pattern (Drop implementations)
- Error code to Result conversion
- CString validation (null byte prevention)
- Memory ownership (Rust owns nothing from C++)

### Edge Cases
- NULL pointers from C++ (handle gracefully)
- Very long model paths
- Very long prompts
- Token buffer overflow
- Panic during FFI call
- Double-free prevention (Drop called multiple times)

---
Test opportunities identified by Testing Team 🔍
