//! Integration tests for CUDA error to HTTP response conversion
//!
//! Tests that CUDA errors are properly converted to HTTP responses
//! with correct status codes, error codes, and retriable flags.

use axum::{
    body::to_bytes,
    http::StatusCode,
    response::IntoResponse,
};
use serde_json::Value;
use worker_orcd::cuda::CudaError;

// ============================================================================
// HTTP Status Code Integration Tests
// ============================================================================

#[tokio::test]
async fn test_invalid_parameter_error_returns_400() {
    // Create invalid parameter error
    let error = CudaError::from_code(5);
    
    // Convert to HTTP response
    let response = error.into_response();
    
    // Verify status code
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    
    // Extract and verify body
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    
    assert_eq!(json["code"], "INVALID_PARAMETER");
    assert!(json["message"].as_str().unwrap().contains("parameter"));
    assert!(json.get("retriable").is_none() || json["retriable"] == false);
}

#[tokio::test]
async fn test_out_of_memory_error_returns_503() {
    // Create OOM error
    let error = CudaError::from_code(2);
    
    // Convert to HTTP response
    let response = error.into_response();
    
    // Verify status code
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    
    // Extract and verify body
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    
    assert_eq!(json["code"], "VRAM_OOM");
    assert!(json["message"].as_str().unwrap().contains("memory") 
            || json["message"].as_str().unwrap().contains("VRAM"));
    assert_eq!(json["retriable"], true);
}

#[tokio::test]
async fn test_inference_failed_error_returns_500() {
    // Create inference failed error
    let error = CudaError::from_code(4);
    
    // Convert to HTTP response
    let response = error.into_response();
    
    // Verify status code
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    
    // Extract and verify body
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    
    assert_eq!(json["code"], "INFERENCE_FAILED");
    assert!(json["message"].as_str().unwrap().contains("Inference"));
    assert!(json.get("retriable").is_none() || json["retriable"] == false);
}

#[tokio::test]
async fn test_model_load_failed_error_returns_500() {
    // Create model load failed error
    let error = CudaError::from_code(3);
    
    // Convert to HTTP response
    let response = error.into_response();
    
    // Verify status code
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    
    // Extract and verify body
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    
    assert_eq!(json["code"], "MODEL_LOAD_FAILED");
    assert!(json["message"].as_str().unwrap().contains("Model") 
            || json["message"].as_str().unwrap().contains("load"));
    assert!(json.get("retriable").is_none() || json["retriable"] == false);
}

// ============================================================================
// Error Response Structure Tests
// ============================================================================

#[tokio::test]
async fn test_error_response_has_required_fields() {
    let error = CudaError::from_code(1);
    let response = error.into_response();
    
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    
    // Verify required fields exist
    assert!(json.get("code").is_some());
    assert!(json.get("message").is_some());
    
    // Verify field types
    assert!(json["code"].is_string());
    assert!(json["message"].is_string());
    
    // Verify non-empty
    assert!(!json["code"].as_str().unwrap().is_empty());
    assert!(!json["message"].as_str().unwrap().is_empty());
}

#[tokio::test]
async fn test_retriable_flag_only_present_for_oom() {
    // OOM error should have retriable: true
    let oom = CudaError::from_code(2);
    let response = oom.into_response();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["retriable"], true);
    
    // Non-retriable errors should omit retriable field or set to false
    let non_retriable_codes = vec![1, 3, 4, 5, 6, 7, 8];
    for code in non_retriable_codes {
        let error = CudaError::from_code(code);
        let response = error.into_response();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        
        // Either absent or false
        if let Some(retriable) = json.get("retriable") {
            assert_eq!(retriable, false, "Error code {} should not be retriable", code);
        }
    }
}

// ============================================================================
// Error Code Stability Tests
// ============================================================================

#[tokio::test]
async fn test_error_codes_are_stable() {
    // Verify error codes match spec (these are part of API contract)
    let expected_codes = vec![
        (1, "INVALID_DEVICE"),
        (2, "VRAM_OOM"),
        (3, "MODEL_LOAD_FAILED"),
        (4, "INFERENCE_FAILED"),
        (5, "INVALID_PARAMETER"),
        (6, "KERNEL_LAUNCH_FAILED"),
        (7, "VRAM_RESIDENCY_FAILED"),
        (8, "DEVICE_NOT_FOUND"),
        (99, "UNKNOWN"),
    ];
    
    for (code, expected_str) in expected_codes {
        let error = CudaError::from_code(code);
        let response = error.into_response();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        
        assert_eq!(
            json["code"], expected_str,
            "Error code {} should map to {}",
            code, expected_str
        );
    }
}

// ============================================================================
// HTTP Status Code Mapping Tests
// ============================================================================

#[tokio::test]
async fn test_all_errors_have_valid_http_status() {
    let codes = vec![1, 2, 3, 4, 5, 6, 7, 8, 99];
    
    for code in codes {
        let error = CudaError::from_code(code);
        let response = error.into_response();
        let status = response.status();
        
        // Verify status is one of the expected codes
        assert!(
            status == StatusCode::BAD_REQUEST
                || status == StatusCode::INTERNAL_SERVER_ERROR
                || status == StatusCode::SERVICE_UNAVAILABLE,
            "Error code {} has unexpected status: {}",
            code,
            status
        );
    }
}

// ============================================================================
// Error Context Preservation Tests
// ============================================================================

#[tokio::test]
async fn test_error_message_includes_context() {
    let error = CudaError::from_code(2);
    let response = error.into_response();
    
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    
    let message = json["message"].as_str().unwrap();
    
    // Message should include context from C++ layer
    assert!(
        message.contains("VRAM") || message.contains("memory"),
        "Error message should include context: {}",
        message
    );
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[tokio::test]
async fn test_unknown_error_code_returns_500() {
    let error = CudaError::from_code(999);
    let response = error.into_response();
    
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    
    assert_eq!(json["code"], "UNKNOWN");
}

#[tokio::test]
async fn test_error_response_is_valid_json() {
    let codes = vec![1, 2, 3, 4, 5, 6, 7, 8, 99];
    
    for code in codes {
        let error = CudaError::from_code(code);
        let response = error.into_response();
        
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        
        // Should parse as valid JSON
        let result: Result<Value, _> = serde_json::from_slice(&body);
        assert!(
            result.is_ok(),
            "Error code {} should produce valid JSON",
            code
        );
    }
}

// ============================================================================
// Content-Type Tests
// ============================================================================

#[tokio::test]
async fn test_error_response_has_json_content_type() {
    let error = CudaError::from_code(2);
    let response = error.into_response();
    
    let content_type = response.headers().get("content-type");
    assert!(content_type.is_some());
    
    let content_type_str = content_type.unwrap().to_str().unwrap();
    assert!(content_type_str.contains("application/json"));
}

// ---
// Built by Foundation-Alpha 🏗️
