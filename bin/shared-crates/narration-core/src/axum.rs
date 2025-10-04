//! Axum middleware for correlation ID extraction and generation.
//!
//! Provides middleware that:
//! - Extracts correlation ID from `X-Correlation-ID` header
//! - Validates the ID format
//! - Generates a new ID if missing or invalid
//! - Stores the ID in request extensions
//! - Adds the ID to response headers
//!
//! # Example
//! ```rust,ignore
//! use axum::{Router, routing::post, middleware};
//! use observability_narration_core::axum::correlation_middleware;
//!
//! let app = Router::new()
//!     .route("/execute", post(handler))
//!     .layer(middleware::from_fn(correlation_middleware));
//! ```

use crate::{generate_correlation_id, validate_correlation_id};
use axum::{extract::Request, http::HeaderValue, middleware::Next, response::Response};

/// Axum middleware for correlation ID extraction/generation.
///
/// This middleware:
/// 1. Extracts `X-Correlation-ID` from incoming request headers
/// 2. Validates the ID format (UUID v4)
/// 3. Generates a new ID if missing or invalid
/// 4. Stores the ID in request extensions for handler access
/// 5. Adds the ID to response headers
///
/// # Example
/// ```rust,ignore
/// use axum::{
///     Router,
///     routing::post,
///     middleware,
///     extract::Extension,
/// };
/// use observability_narration_core::axum::correlation_middleware;
///
/// async fn handler(Extension(correlation_id): Extension<String>) -> &'static str {
///     // correlation_id is automatically extracted
///     "OK"
/// }
///
///     .route("/execute", post(handler))
///     .layer(middleware::from_fn(correlation_middleware));
/// ```
pub async fn correlation_middleware(mut req: Request, next: Next) -> Response {
    // Extract and validate correlation ID from header
    let correlation_id = req.headers()
        .get("X-Correlation-ID")
        .and_then(|v| v.to_str().ok())
        .and_then(|id| validate_correlation_id(id))
        .map(|id| id.to_string())
        .unwrap_or_else(generate_correlation_id);

    // Store in request extensions for handler access
    req.extensions_mut().insert(correlation_id.clone());

    // Process request
    let mut response = next.run(req).await;

    // Add correlation ID to response headers
    if let Ok(header_value) = HeaderValue::from_str(&correlation_id) {
        response.headers_mut().insert("X-Correlation-ID", header_value);
    }

    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        extract::Extension,
        http::{Request, StatusCode},
        middleware,
        response::IntoResponse,
        routing::get,
        Router,
    };
    use tower::util::ServiceExt;

    async fn test_handler(Extension(correlation_id): Extension<String>) -> impl IntoResponse {
        (StatusCode::OK, correlation_id)
    }

    #[tokio::test]
    async fn test_middleware_extracts_valid_correlation_id() {
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
        assert_eq!(response.headers().get("X-Correlation-ID").unwrap(), valid_id);
    }

    #[tokio::test]
    async fn test_middleware_generates_id_when_missing() {
        let app = Router::new()
            .route("/test", get(test_handler))
            .layer(middleware::from_fn(correlation_middleware));

        let request = Request::builder().uri("/test").body(Body::empty()).unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let header = response.headers().get("X-Correlation-ID").unwrap();
        let id = header.to_str().unwrap();

        // Should be a valid UUID
        assert!(validate_correlation_id(id).is_some());
    }

    #[tokio::test]
    async fn test_middleware_generates_id_when_invalid() {
        let app = Router::new()
            .route("/test", get(test_handler))
            .layer(middleware::from_fn(correlation_middleware));

        let request = Request::builder()
            .uri("/test")
            .header("X-Correlation-ID", "invalid-id")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let header = response.headers().get("X-Correlation-ID").unwrap();
        let id = header.to_str().unwrap();

        // Should be a valid UUID (not the invalid one)
        assert!(validate_correlation_id(id).is_some());
        assert_ne!(id, "invalid-id");
    }
}
