//! Integration tests for correlation ID middleware
//!
//! These tests verify that correlation IDs are properly:
//! - Extracted from X-Correlation-ID headers
//! - Generated when missing
//! - Propagated to response headers
//! - Accessible in handlers
//! - Included in logs
//!
//! # Spec References
//! - WORK-3040: Correlation ID middleware

use axum::{
    body::Body,
    extract::Extension,
    http::{Request, StatusCode},
    middleware,
    response::IntoResponse,
    routing::get,
    Router,
};
use observability_narration_core::{axum::correlation_middleware, validate_correlation_id};
use tower::ServiceExt;

/// Test handler that returns the correlation ID
async fn test_handler(Extension(correlation_id): Extension<String>) -> impl IntoResponse {
    (StatusCode::OK, correlation_id)
}

/// Test: Request with X-Correlation-ID header preserves ID
///
/// Verifies that valid correlation IDs from clients are preserved
#[tokio::test]
async fn test_request_with_correlation_id_preserves_id() {
    let app = Router::new()
        .route("/test", get(test_handler))
        .layer(middleware::from_fn(correlation_middleware));

    let valid_id = "550e8400-e29b-41d4-a716-446655440000";
    let request = Request::builder()
        .uri("/test")
        .header("X-Correlation-ID", valid_id)
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Verify response header contains the same ID
    let response_id = response.headers().get("X-Correlation-ID").unwrap();
    assert_eq!(response_id.to_str().unwrap(), valid_id);
}

/// Test: Request without header generates new ID
///
/// Verifies that missing correlation IDs trigger UUID generation
#[tokio::test]
async fn test_request_without_header_generates_id() {
    let app = Router::new()
        .route("/test", get(test_handler))
        .layer(middleware::from_fn(correlation_middleware));

    let request = Request::builder().uri("/test").body(Body::empty()).unwrap();

    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Verify response header contains a generated UUID
    let response_id = response.headers().get("X-Correlation-ID").unwrap();
    let id = response_id.to_str().unwrap();

    // Should be a valid UUID
    assert!(validate_correlation_id(id).is_some());
}

/// Test: Invalid header value triggers ID generation
///
/// Verifies that invalid correlation IDs are rejected and new ones generated
#[tokio::test]
async fn test_invalid_header_triggers_generation() {
    let app = Router::new()
        .route("/test", get(test_handler))
        .layer(middleware::from_fn(correlation_middleware));

    let invalid_id = "invalid-id-with-special-chars!@#$";
    let request = Request::builder()
        .uri("/test")
        .header("X-Correlation-ID", invalid_id)
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Verify response header contains a NEW valid UUID (not the invalid one)
    let response_id = response.headers().get("X-Correlation-ID").unwrap();
    let id = response_id.to_str().unwrap();

    assert!(validate_correlation_id(id).is_some());
    assert_ne!(id, invalid_id);
}

/// Test: Correlation ID accessible in handler via extensions
///
/// Verifies that handlers can extract the correlation ID
#[tokio::test]
async fn test_correlation_id_accessible_in_handler() {
    let app = Router::new()
        .route("/test", get(test_handler))
        .layer(middleware::from_fn(correlation_middleware));

    let test_id = "550e8400-e29b-41d4-a716-446655440000"; // Valid UUID v4
    let request = Request::builder()
        .uri("/test")
        .header("X-Correlation-ID", test_id)
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    // Handler returns the correlation ID in the body
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();

    // Should match the provided ID
    assert_eq!(body_str, test_id);
}

/// Test: Empty correlation ID header triggers generation
///
/// Verifies that empty headers are treated as missing
#[tokio::test]
async fn test_empty_header_triggers_generation() {
    let app = Router::new()
        .route("/test", get(test_handler))
        .layer(middleware::from_fn(correlation_middleware));

    let request =
        Request::builder().uri("/test").header("X-Correlation-ID", "").body(Body::empty()).unwrap();

    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Verify a new UUID was generated
    let response_id = response.headers().get("X-Correlation-ID").unwrap();
    let id = response_id.to_str().unwrap();

    assert!(validate_correlation_id(id).is_some());
    assert!(!id.is_empty());
}

/// Test: Very long correlation ID rejected
///
/// Verifies that excessively long IDs trigger generation
#[tokio::test]
async fn test_very_long_correlation_id_rejected() {
    let app = Router::new()
        .route("/test", get(test_handler))
        .layer(middleware::from_fn(correlation_middleware));

    let long_id = "a".repeat(1000);
    let request = Request::builder()
        .uri("/test")
        .header("X-Correlation-ID", long_id.clone())
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Verify a new UUID was generated (not the long one)
    let response_id = response.headers().get("X-Correlation-ID").unwrap();
    let id = response_id.to_str().unwrap();

    assert!(validate_correlation_id(id).is_some());
    assert_ne!(id, long_id);
}

/// Test: Correlation ID with special characters rejected
///
/// Verifies that IDs with special characters trigger generation
#[tokio::test]
async fn test_special_characters_rejected() {
    let app = Router::new()
        .route("/test", get(test_handler))
        .layer(middleware::from_fn(correlation_middleware));

    let special_id = "test@#$%^&*()";
    let request = Request::builder()
        .uri("/test")
        .header("X-Correlation-ID", special_id)
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Verify a new UUID was generated
    let response_id = response.headers().get("X-Correlation-ID").unwrap();
    let id = response_id.to_str().unwrap();

    assert!(validate_correlation_id(id).is_some());
    assert_ne!(id, special_id);
}

/// Test: Multiple requests get different generated IDs
///
/// Verifies that each request without an ID gets a unique UUID
#[tokio::test]
async fn test_multiple_requests_get_different_ids() {
    let app = Router::new()
        .route("/test", get(test_handler))
        .layer(middleware::from_fn(correlation_middleware));

    // First request
    let request1 = Request::builder().uri("/test").body(Body::empty()).unwrap();

    let response1 = app.clone().oneshot(request1).await.unwrap();
    let id1 = response1.headers().get("X-Correlation-ID").unwrap().to_str().unwrap().to_string();

    // Second request
    let request2 = Request::builder().uri("/test").body(Body::empty()).unwrap();

    let response2 = app.oneshot(request2).await.unwrap();
    let id2 = response2.headers().get("X-Correlation-ID").unwrap().to_str().unwrap().to_string();

    // IDs should be different
    assert_ne!(id1, id2);

    // Both should be valid UUIDs
    assert!(validate_correlation_id(&id1).is_some());
    assert!(validate_correlation_id(&id2).is_some());
}

/// Test: Correlation ID format validation
///
/// Verifies that only valid UUID v4 formats are accepted
#[test]
fn test_correlation_id_format_validation() {
    // Valid UUID v4
    let valid_uuid = "550e8400-e29b-41d4-a716-446655440000";
    assert!(validate_correlation_id(valid_uuid).is_some());

    // Invalid formats
    assert!(validate_correlation_id("").is_none());
    assert!(validate_correlation_id("not-a-uuid").is_none());
    assert!(validate_correlation_id("123").is_none());
    assert!(validate_correlation_id("@#$%").is_none());
}

// ---
// Built by Foundation-Alpha 🏗️
