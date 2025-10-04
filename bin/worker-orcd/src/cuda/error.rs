//! Error types and conversion for CUDA FFI
//!
//! This module defines Rust error types that correspond to CUDA error codes
//! and provides conversion from C error codes to typed Rust errors.
//!
//! # HTTP Status Code Mapping
//!
//! - `InvalidParameter` → 400 Bad Request
//! - `OutOfMemory` → 503 Service Unavailable (retriable)
//! - All other errors → 500 Internal Server Error
//!
//! # Retriable Errors
//!
//! Only `OutOfMemory` errors are retriable. The orchestrator can retry
//! these requests on a different worker with available VRAM.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
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
    /// # Arguments
    ///
    /// * `code` - C error code from CUDA FFI
    ///
    /// # Returns
    ///
    /// Typed CudaError variant with error message from C layer
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

    /// Get stable string code for API responses
    ///
    /// Returns a stable string identifier for this error type.
    /// These codes are part of the API contract and must not change.
    pub fn code_str(&self) -> &'static str {
        match self {
            Self::InvalidDevice(_) => "INVALID_DEVICE",
            Self::OutOfMemory(_) => "VRAM_OOM",
            Self::ModelLoadFailed(_) => "MODEL_LOAD_FAILED",
            Self::InferenceFailed(_) => "INFERENCE_FAILED",
            Self::InvalidParameter(_) => "INVALID_PARAMETER",
            Self::KernelLaunchFailed(_) => "KERNEL_LAUNCH_FAILED",
            Self::VramResidencyFailed(_) => "VRAM_RESIDENCY_FAILED",
            Self::DeviceNotFound(_) => "DEVICE_NOT_FOUND",
            Self::Unknown { .. } => "UNKNOWN",
        }
    }

    /// Get HTTP status code for this error
    ///
    /// Maps CUDA errors to appropriate HTTP status codes:
    /// - `InvalidParameter` → 400 Bad Request (client error)
    /// - `OutOfMemory` → 503 Service Unavailable (retriable)
    /// - All others → 500 Internal Server Error
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::InvalidParameter(_) => StatusCode::BAD_REQUEST,
            Self::OutOfMemory(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::InvalidDevice(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::ModelLoadFailed(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::InferenceFailed(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::KernelLaunchFailed(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::VramResidencyFailed(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::DeviceNotFound(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Unknown { .. } => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Check if error is retriable by orchestrator
    ///
    /// Only `OutOfMemory` errors are retriable. The orchestrator can
    /// retry these requests on a different worker with available VRAM.
    pub fn is_retriable(&self) -> bool {
        matches!(self, Self::OutOfMemory(_))
    }
}

// ============================================================================
// HTTP Response Integration
// ============================================================================

/// Error response JSON structure
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    /// Stable error code (e.g., "VRAM_OOM", "INVALID_DEVICE")
    pub code: String,
    /// Human-readable error message
    pub message: String,
    /// Whether error is retriable (only present if true)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retriable: Option<bool>,
}

impl IntoResponse for CudaError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let code = self.code_str().to_string();
        let message = self.to_string();
        let retriable = if self.is_retriable() { Some(true) } else { None };

        // Log error with structured fields
        tracing::error!(
            error_code = %code,
            error_message = %message,
            status_code = %status.as_u16(),
            retriable = retriable.unwrap_or(false),
            "CUDA error occurred"
        );

        let body = Json(ErrorResponse {
            code,
            message,
            retriable,
        });

        (status, body).into_response()
    }
}

// ============================================================================
// SSE Error Events
// ============================================================================

/// SSE error event structure
#[derive(Debug, Serialize)]
pub struct SseError {
    /// Stable error code
    pub code: String,
    /// Human-readable error message
    pub message: String,
}

impl From<CudaError> for SseError {
    fn from(err: CudaError) -> Self {
        Self {
            code: err.code_str().to_string(),
            message: err.to_string(),
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

    // ========================================================================
    // HTTP Status Code Tests
    // ========================================================================

    #[test]
    fn test_invalid_parameter_returns_400() {
        let err = CudaError::from_code(5);
        assert_eq!(err.status_code(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_out_of_memory_returns_503() {
        let err = CudaError::from_code(2);
        assert_eq!(err.status_code(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_other_errors_return_500() {
        let codes = vec![1, 3, 4, 6, 7, 8, 99];
        for code in codes {
            let err = CudaError::from_code(code);
            assert_eq!(
                err.status_code(),
                StatusCode::INTERNAL_SERVER_ERROR,
                "Error code {} should return 500",
                code
            );
        }
    }

    // ========================================================================
    // String Code Tests
    // ========================================================================

    #[test]
    fn test_code_str_returns_stable_codes() {
        assert_eq!(
            CudaError::from_code(1).code_str(),
            "INVALID_DEVICE"
        );
        assert_eq!(CudaError::from_code(2).code_str(), "VRAM_OOM");
        assert_eq!(
            CudaError::from_code(3).code_str(),
            "MODEL_LOAD_FAILED"
        );
        assert_eq!(
            CudaError::from_code(4).code_str(),
            "INFERENCE_FAILED"
        );
        assert_eq!(
            CudaError::from_code(5).code_str(),
            "INVALID_PARAMETER"
        );
        assert_eq!(
            CudaError::from_code(6).code_str(),
            "KERNEL_LAUNCH_FAILED"
        );
        assert_eq!(
            CudaError::from_code(7).code_str(),
            "VRAM_RESIDENCY_FAILED"
        );
        assert_eq!(
            CudaError::from_code(8).code_str(),
            "DEVICE_NOT_FOUND"
        );
        assert_eq!(CudaError::from_code(99).code_str(), "UNKNOWN");
    }

    // ========================================================================
    // Retriable Error Tests
    // ========================================================================

    #[test]
    fn test_only_oom_is_retriable() {
        let oom = CudaError::from_code(2);
        assert!(oom.is_retriable(), "OOM should be retriable");

        let non_retriable_codes = vec![1, 3, 4, 5, 6, 7, 8, 99];
        for code in non_retriable_codes {
            let err = CudaError::from_code(code);
            assert!(
                !err.is_retriable(),
                "Error code {} should not be retriable",
                code
            );
        }
    }

    // ========================================================================
    // ErrorResponse Serialization Tests
    // ========================================================================

    #[test]
    fn test_error_response_serialization() {
        let response = ErrorResponse {
            code: "VRAM_OOM".to_string(),
            message: "Out of VRAM".to_string(),
            retriable: Some(true),
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["code"], "VRAM_OOM");
        assert_eq!(json["message"], "Out of VRAM");
        assert_eq!(json["retriable"], true);
    }

    #[test]
    fn test_error_response_omits_retriable_when_none() {
        let response = ErrorResponse {
            code: "INFERENCE_FAILED".to_string(),
            message: "Inference failed".to_string(),
            retriable: None,
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["code"], "INFERENCE_FAILED");
        assert_eq!(json["message"], "Inference failed");
        assert!(json.get("retriable").is_none());
    }

    // ========================================================================
    // SSE Error Conversion Tests
    // ========================================================================

    #[test]
    fn test_sse_error_conversion() {
        let cuda_err = CudaError::from_code(2);
        let sse_err: SseError = cuda_err.into();

        assert_eq!(sse_err.code, "VRAM_OOM");
        assert!(!sse_err.message.is_empty());
    }

    #[test]
    fn test_sse_error_serialization() {
        let sse_err = SseError {
            code: "INFERENCE_FAILED".to_string(),
            message: "Inference failed".to_string(),
        };

        let json = serde_json::to_value(&sse_err).unwrap();
        assert_eq!(json["code"], "INFERENCE_FAILED");
        assert_eq!(json["message"], "Inference failed");
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_unknown_error_code_maps_to_unknown() {
        let err = CudaError::from_code(999);
        assert!(matches!(err, CudaError::Unknown { .. }));
        assert_eq!(err.code_str(), "UNKNOWN");
        assert_eq!(err.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(!err.is_retriable());
    }

    #[test]
    fn test_error_display_includes_message() {
        let err = CudaError::from_code(2);
        let display = format!("{}", err);
        assert!(display.contains("VRAM") || display.contains("memory"));
    }

    // ========================================================================
    // Property Tests
    // ========================================================================

    #[test]
    fn test_all_errors_have_non_empty_code_str() {
        let codes = vec![1, 2, 3, 4, 5, 6, 7, 8, 99];
        for code in codes {
            let err = CudaError::from_code(code);
            let code_str = err.code_str();
            assert!(!code_str.is_empty(), "Error code {} has empty string code", code);
            assert!(code_str.chars().all(|c| c.is_ascii_uppercase() || c == '_'),
                "Error code {} has invalid string code: {}", code, code_str);
        }
    }

    #[test]
    fn test_all_errors_have_valid_status_codes() {
        let codes = vec![1, 2, 3, 4, 5, 6, 7, 8, 99];
        for code in codes {
            let err = CudaError::from_code(code);
            let status = err.status_code();
            assert!(
                status == StatusCode::BAD_REQUEST
                    || status == StatusCode::INTERNAL_SERVER_ERROR
                    || status == StatusCode::SERVICE_UNAVAILABLE,
                "Error code {} has invalid status code: {}",
                code,
                status
            );
        }
    }
}

// ---
// Built by Foundation-Alpha 🏗️
