//! Error types and conversion for CUDA FFI
//!
//! This module defines Rust error types that correspond to CUDA error codes
//! and provides conversion from C error codes to typed Rust errors.

use std::ffi::CStr;
use std::os::raw::c_int;

// ============================================================================
// Error Code Enum
// ============================================================================

/// CUDA error codes (matches C enum)
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaErrorCode {
    Success = 0,
    InvalidDevice = 1,
    OutOfMemory = 2,
    ModelLoadFailed = 3,
    InferenceFailed = 4,
    InvalidParameter = 5,
    KernelLaunchFailed = 6,
    VramResidencyFailed = 7,
    DeviceNotFound = 8,
    Unknown = 99,
}

impl From<c_int> for CudaErrorCode {
    fn from(code: c_int) -> Self {
        match code {
            0 => Self::Success,
            1 => Self::InvalidDevice,
            2 => Self::OutOfMemory,
            3 => Self::ModelLoadFailed,
            4 => Self::InferenceFailed,
            5 => Self::InvalidParameter,
            6 => Self::KernelLaunchFailed,
            7 => Self::VramResidencyFailed,
            8 => Self::DeviceNotFound,
            _ => Self::Unknown,
        }
    }
}

// ============================================================================
// Rust Error Type
// ============================================================================

/// Typed CUDA errors with context
#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("Invalid CUDA device: {0}")]
    InvalidDevice(String),

    #[error("Out of VRAM: {0}")]
    OutOfMemory(String),

    #[error("Model load failed: {0}")]
    ModelLoadFailed(String),

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("CUDA kernel launch failed: {0}")]
    KernelLaunchFailed(String),

    #[error("VRAM residency check failed: {0}")]
    VramResidencyFailed(String),

    #[error("CUDA device not found: {0}")]
    DeviceNotFound(String),

    #[error("Unknown CUDA error (code {code}): {message}")]
    Unknown { code: i32, message: String },
}

impl CudaError {
    /// Convert C error code to typed Rust error with message
    ///
    /// # Safety
    ///
    /// Calls unsafe FFI function `cuda_error_message` which returns
    /// a pointer to a static string.
    pub fn from_code(code: c_int) -> Self {
        use super::ffi;

        // SAFETY: cuda_error_message returns static string (never NULL)
        let msg_ptr = unsafe { ffi::cuda_error_message(code) };
        let msg = if msg_ptr.is_null() {
            format!("Error code {}", code)
        } else {
            // SAFETY: msg_ptr is valid static string from C
            unsafe { CStr::from_ptr(msg_ptr) }
                .to_string_lossy()
                .into_owned()
        };

        let error_code = CudaErrorCode::from(code);
        match error_code {
            CudaErrorCode::Success => {
                unreachable!("Success code should not be converted to error")
            }
            CudaErrorCode::InvalidDevice => Self::InvalidDevice(msg),
            CudaErrorCode::OutOfMemory => Self::OutOfMemory(msg),
            CudaErrorCode::ModelLoadFailed => Self::ModelLoadFailed(msg),
            CudaErrorCode::InferenceFailed => Self::InferenceFailed(msg),
            CudaErrorCode::InvalidParameter => Self::InvalidParameter(msg),
            CudaErrorCode::KernelLaunchFailed => Self::KernelLaunchFailed(msg),
            CudaErrorCode::VramResidencyFailed => Self::VramResidencyFailed(msg),
            CudaErrorCode::DeviceNotFound => Self::DeviceNotFound(msg),
            CudaErrorCode::Unknown => Self::Unknown { code, message: msg },
        }
    }

    /// Get error code for this error
    pub fn code(&self) -> CudaErrorCode {
        match self {
            Self::InvalidDevice(_) => CudaErrorCode::InvalidDevice,
            Self::OutOfMemory(_) => CudaErrorCode::OutOfMemory,
            Self::ModelLoadFailed(_) => CudaErrorCode::ModelLoadFailed,
            Self::InferenceFailed(_) => CudaErrorCode::InferenceFailed,
            Self::InvalidParameter(_) => CudaErrorCode::InvalidParameter,
            Self::KernelLaunchFailed(_) => CudaErrorCode::KernelLaunchFailed,
            Self::VramResidencyFailed(_) => CudaErrorCode::VramResidencyFailed,
            Self::DeviceNotFound(_) => CudaErrorCode::DeviceNotFound,
            Self::Unknown { .. } => CudaErrorCode::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_conversion() {
        assert_eq!(CudaErrorCode::from(0), CudaErrorCode::Success);
        assert_eq!(CudaErrorCode::from(1), CudaErrorCode::InvalidDevice);
        assert_eq!(CudaErrorCode::from(2), CudaErrorCode::OutOfMemory);
        assert_eq!(CudaErrorCode::from(3), CudaErrorCode::ModelLoadFailed);
        assert_eq!(CudaErrorCode::from(4), CudaErrorCode::InferenceFailed);
        assert_eq!(CudaErrorCode::from(5), CudaErrorCode::InvalidParameter);
        assert_eq!(CudaErrorCode::from(6), CudaErrorCode::KernelLaunchFailed);
        assert_eq!(CudaErrorCode::from(7), CudaErrorCode::VramResidencyFailed);
        assert_eq!(CudaErrorCode::from(8), CudaErrorCode::DeviceNotFound);
        assert_eq!(CudaErrorCode::from(99), CudaErrorCode::Unknown);
        assert_eq!(CudaErrorCode::from(100), CudaErrorCode::Unknown);
    }

    #[test]
    fn test_cuda_error_from_code() {
        let err = CudaError::from_code(1);
        assert!(matches!(err, CudaError::InvalidDevice(_)));
        assert_eq!(err.code(), CudaErrorCode::InvalidDevice);

        let err = CudaError::from_code(2);
        assert!(matches!(err, CudaError::OutOfMemory(_)));
        assert_eq!(err.code(), CudaErrorCode::OutOfMemory);

        let err = CudaError::from_code(99);
        assert!(matches!(err, CudaError::Unknown { .. }));
        assert_eq!(err.code(), CudaErrorCode::Unknown);
    }

    #[test]
    fn test_error_code_roundtrip() {
        let codes = vec![1, 2, 3, 4, 5, 6, 7, 8, 99];
        for code in codes {
            let err = CudaError::from_code(code);
            let err_code = err.code();
            assert_eq!(CudaErrorCode::from(code), err_code);
        }
    }
}

// ---
// Built by Foundation-Alpha 🏗️
