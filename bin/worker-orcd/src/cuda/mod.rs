//! CUDA FFI bindings for worker-orcd
//!
//! This module provides safe Rust wrappers around the C API
//! exposed by the CUDA C++ implementation.
//!
//! When built WITHOUT the `cuda` feature, this module provides
//! stub implementations for development on CUDA-less devices.

#[cfg(feature = "cuda")]
use std::ffi::{c_char, c_int, CString};

#[cfg(not(feature = "cuda"))]
#[allow(unused_imports)]
use std::ffi::CString;

// ============================================================================
// Opaque Handle Types
// ============================================================================

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

// ============================================================================
// Error Codes
// ============================================================================

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaErrorCode {
    Success = 0,
    DeviceNotFound = 1,
    OutOfMemory = 2,
    InvalidDevice = 3,
    ModelLoadFailed = 4,
    InferenceFailed = 5,
    VramResidencyFailed = 6,
    KernelLaunchFailed = 7,
    InvalidParameter = 8,
    Unknown = 99,
}

impl From<i32> for CudaErrorCode {
    fn from(code: i32) -> Self {
        match code {
            0 => Self::Success,
            1 => Self::DeviceNotFound,
            2 => Self::OutOfMemory,
            3 => Self::InvalidDevice,
            4 => Self::ModelLoadFailed,
            5 => Self::InferenceFailed,
            6 => Self::VramResidencyFailed,
            7 => Self::KernelLaunchFailed,
            8 => Self::InvalidParameter,
            _ => Self::Unknown,
        }
    }
}

// ============================================================================
// Error Type
// ============================================================================

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("CUDA device not found")]
    DeviceNotFound,
    
    #[error("Out of GPU memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory { requested: u64, available: u64 },
    
    #[error("Invalid CUDA device: {0}")]
    InvalidDevice(i32),
    
    #[error("Model load failed: {0}")]
    ModelLoadFailed(String),
    
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
    
    #[error("VRAM residency check failed")]
    VramResidencyFailed,
    
    #[error("CUDA kernel launch failed: {0}")]
    KernelLaunchFailed(String),
    
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Unknown CUDA error: code {0}")]
    Unknown(i32),
}

impl CudaError {
    pub fn from_code(code: i32) -> Self {
        let error_code = CudaErrorCode::from(code);
        match error_code {
            CudaErrorCode::Success => unreachable!("Success is not an error"),
            CudaErrorCode::DeviceNotFound => Self::DeviceNotFound,
            CudaErrorCode::OutOfMemory => Self::OutOfMemory {
                requested: 0,
                available: 0,
            },
            CudaErrorCode::InvalidDevice => Self::InvalidDevice(0),
            CudaErrorCode::ModelLoadFailed => Self::ModelLoadFailed("Unknown".to_string()),
            CudaErrorCode::InferenceFailed => Self::InferenceFailed("Unknown".to_string()),
            CudaErrorCode::VramResidencyFailed => Self::VramResidencyFailed,
            CudaErrorCode::KernelLaunchFailed => Self::KernelLaunchFailed("Unknown".to_string()),
            CudaErrorCode::InvalidParameter => Self::InvalidParameter("Unknown".to_string()),
            CudaErrorCode::Unknown => Self::Unknown(code),
        }
    }
}

// ============================================================================
// Raw FFI Declarations
// ============================================================================

#[cfg(feature = "cuda")]
extern "C" {
    // Error handling
    fn cuda_error_message(error_code: c_int) -> *const c_char;
    
    // Context management
    fn cuda_init(gpu_device: c_int, error_code: *mut c_int) -> *mut CudaContext;
    fn cuda_destroy(ctx: *mut CudaContext);
    fn cuda_get_device_count() -> c_int;
    
    // Model loading
    fn cuda_load_model(
        ctx: *mut CudaContext,
        model_path: *const c_char,
        vram_bytes_used: *mut u64,
        error_code: *mut c_int,
    ) -> *mut CudaModel;
    fn cuda_unload_model(model: *mut CudaModel);
    fn cuda_model_get_vram_usage(model: *mut CudaModel) -> u64;
    
    // Inference
    fn cuda_inference_start(
        model: *mut CudaModel,
        prompt: *const c_char,
        max_tokens: c_int,
        temperature: f32,
        seed: u64,
        error_code: *mut c_int,
    ) -> *mut InferenceResult;
    fn cuda_inference_next_token(
        result: *mut InferenceResult,
        token_out: *mut c_char,
        token_buffer_size: c_int,
        token_index: *mut c_int,
        error_code: *mut c_int,
    ) -> bool;
    fn cuda_inference_free(result: *mut InferenceResult);
    
    // Health monitoring
    fn cuda_check_vram_residency(model: *mut CudaModel, error_code: *mut c_int) -> bool;
    fn cuda_get_vram_usage(model: *mut CudaModel) -> u64;
    fn cuda_get_process_vram_usage(ctx: *mut CudaContext) -> u64;
    fn cuda_check_device_health(ctx: *mut CudaContext, error_code: *mut c_int) -> bool;
}

// ============================================================================
// Safe Wrappers
// ============================================================================

pub mod safe {
    use super::*;
    
    /// Safe wrapper for CUDA context
    pub struct ContextHandle {
        #[cfg(feature = "cuda")]
        ptr: *mut CudaContext,
        #[cfg(not(feature = "cuda"))]
        _gpu_device: i32,
    }
    
    impl ContextHandle {
        pub fn new(gpu_device: i32) -> Result<Self, CudaError> {
            #[cfg(feature = "cuda")]
            {
                let mut error_code = 0;
                let ptr = unsafe { cuda_init(gpu_device, &mut error_code) };
                
                if ptr.is_null() {
                    return Err(CudaError::from_code(error_code));
                }
                
                Ok(Self { ptr })
            }
            
            #[cfg(not(feature = "cuda"))]
            {
                tracing::warn!(
                    gpu_device,
                    "CUDA stub: ContextHandle::new() - built without CUDA support"
                );
                Err(CudaError::DeviceNotFound)
            }
        }
        
        #[cfg(feature = "cuda")]
        pub fn as_ptr(&self) -> *mut CudaContext {
            self.ptr
        }
        
        #[cfg(not(feature = "cuda"))]
        pub fn as_ptr(&self) -> *mut CudaContext {
            std::ptr::null_mut()
        }
        
        pub fn device_count() -> i32 {
            #[cfg(feature = "cuda")]
            {
                unsafe { cuda_get_device_count() }
            }
            
            #[cfg(not(feature = "cuda"))]
            {
                0
            }
        }
    }
    
    impl Drop for ContextHandle {
        fn drop(&mut self) {
            #[cfg(feature = "cuda")]
            {
                unsafe { cuda_destroy(self.ptr) };
            }
        }
    }
    
    // Safety: CudaContext is thread-safe (CUDA context is per-process)
    unsafe impl Send for ContextHandle {}
    unsafe impl Sync for ContextHandle {}
    
    /// Safe wrapper for CUDA model
    pub struct ModelHandle {
        #[cfg(feature = "cuda")]
        ptr: *mut CudaModel,
        vram_bytes: u64,
        #[cfg(not(feature = "cuda"))]
        _model_path: String,
    }
    
    impl ModelHandle {
        pub fn load(
            _ctx: &ContextHandle,
            model_path: &str,
        ) -> Result<Self, CudaError> {
            #[cfg(feature = "cuda")]
            {
                let path_cstr = CString::new(model_path)
                    .map_err(|_| CudaError::InvalidParameter("Invalid path".to_string()))?;
                
                let mut vram_bytes = 0;
                let mut error_code = 0;
                
                let ptr = unsafe {
                    cuda_load_model(
                        _ctx.as_ptr(),
                        path_cstr.as_ptr(),
                        &mut vram_bytes,
                        &mut error_code,
                    )
                };
                
                if ptr.is_null() {
                    return Err(CudaError::from_code(error_code));
                }
                
                Ok(Self { ptr, vram_bytes })
            }
            
            #[cfg(not(feature = "cuda"))]
            {
                tracing::warn!(
                    model_path,
                    "CUDA stub: ModelHandle::load() - built without CUDA support"
                );
                Err(CudaError::ModelLoadFailed(
                    "Built without CUDA support".to_string()
                ))
            }
        }
        
        pub fn vram_bytes(&self) -> u64 {
            self.vram_bytes
        }
        
        #[cfg(feature = "cuda")]
        pub fn as_ptr(&self) -> *mut CudaModel {
            self.ptr
        }
        
        #[cfg(not(feature = "cuda"))]
        pub fn as_ptr(&self) -> *mut CudaModel {
            std::ptr::null_mut()
        }
        
        pub fn check_vram_residency(&self) -> Result<bool, CudaError> {
            #[cfg(feature = "cuda")]
            {
                let mut error_code = 0;
                let resident = unsafe {
                    cuda_check_vram_residency(self.ptr, &mut error_code)
                };
                
                if error_code != 0 {
                    return Err(CudaError::from_code(error_code));
                }
                
                Ok(resident)
            }
            
            #[cfg(not(feature = "cuda"))]
            {
                Ok(false)
            }
        }
    }
    
    impl Drop for ModelHandle {
        fn drop(&mut self) {
            #[cfg(feature = "cuda")]
            {
                unsafe { cuda_unload_model(self.ptr) };
            }
        }
    }
    
    // Safety: ModelHandle is thread-safe (model is immutable after load)
    unsafe impl Send for ModelHandle {}
    unsafe impl Sync for ModelHandle {}
    
    /// Safe wrapper for inference session
    pub struct InferenceHandle {
        #[cfg(feature = "cuda")]
        ptr: *mut InferenceResult,
        #[cfg(not(feature = "cuda"))]
        _stub: (),
    }
    
    impl InferenceHandle {
        pub fn start(
            _model: &ModelHandle,
            prompt: &str,
            max_tokens: i32,
            temperature: f32,
            seed: u64,
        ) -> Result<Self, CudaError> {
            #[cfg(feature = "cuda")]
            {
                let prompt_cstr = CString::new(prompt)
                    .map_err(|_| CudaError::InvalidParameter("Invalid prompt".to_string()))?;
                
                let mut error_code = 0;
                
                let ptr = unsafe {
                    cuda_inference_start(
                        _model.as_ptr(),
                        prompt_cstr.as_ptr(),
                        max_tokens,
                        temperature,
                        seed,
                        &mut error_code,
                    )
                };
                
                if ptr.is_null() {
                    return Err(CudaError::from_code(error_code));
                }
                
                Ok(Self { ptr })
            }
            
            #[cfg(not(feature = "cuda"))]
            {
                tracing::warn!(
                    prompt_len = prompt.len(),
                    max_tokens,
                    temperature,
                    seed,
                    "CUDA stub: InferenceHandle::start() - built without CUDA support"
                );
                Err(CudaError::InferenceFailed(
                    "Built without CUDA support".to_string()
                ))
            }
        }
        
        pub fn next_token(&mut self) -> Result<Option<(String, i32)>, CudaError> {
            #[cfg(feature = "cuda")]
            {
                use std::ffi::c_char;
                use std::ffi::c_int;
                
                let mut token_buffer = vec![0u8; 256];
                let mut token_index = 0;
                let mut error_code = 0;
                
                let has_token = unsafe {
                    cuda_inference_next_token(
                        self.ptr,
                        token_buffer.as_mut_ptr() as *mut c_char,
                        token_buffer.len() as c_int,
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
                let token_str = String::from_utf8_lossy(&token_buffer[..null_pos]).to_string();
                
                Ok(Some((token_str, token_index)))
            }
            
            #[cfg(not(feature = "cuda"))]
            {
                Ok(None)
            }
        }
    }
    
    impl Drop for InferenceHandle {
        fn drop(&mut self) {
            #[cfg(feature = "cuda")]
            {
                unsafe { cuda_inference_free(self.ptr) };
            }
        }
    }
    
    // Safety: InferenceHandle is NOT thread-safe (single-threaded inference)
    // Do not implement Send/Sync
}
