//! Authentication middleware for pool-managerd
//!
//! Implements Bearer token authentication using the auth-min library
//! with timing-safe comparison and token fingerprinting for audit logs.

use auth_min::{parse_bearer, timing_safe_eq, token_fp6};
use axum::{extract::Request, http::StatusCode, middleware::Next, response::Response};

/// Authentication middleware for pool-managerd endpoints
///
/// Validates Bearer tokens using timing-safe comparison to prevent timing attacks.
/// Logs authentication events with token fingerprints for audit trails.
///
/// # Exemptions
///
/// - `/health` endpoint is exempt from authentication
///
/// # Environment Variables
///
/// - `LLORCH_API_TOKEN`: Required for non-loopback binds (enforced at startup)
///
/// # Returns
///
/// - `Ok(Response)` if authenticated or exempt
/// - `Err(StatusCode::UNAUTHORIZED)` if authentication fails
pub async fn auth_middleware(req: Request, next: Next) -> Result<Response, StatusCode> {
    // Skip auth for health checks
    if req.uri().path() == "/health" {
        return Ok(next.run(req).await);
    }

    // Read token from environment
    let expected_token =
        std::env::var("LLORCH_API_TOKEN").ok().filter(|t| !t.is_empty()).ok_or_else(|| {
            tracing::error!("LLORCH_API_TOKEN not configured");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Extract Authorization header
    let auth = req.headers().get(http::header::AUTHORIZATION).and_then(|v| v.to_str().ok());

    // Parse Bearer token
    let received_token = parse_bearer(auth).ok_or_else(|| {
        tracing::warn!(
            path = %req.uri().path(),
            "Missing or invalid Bearer token"
        );
        StatusCode::UNAUTHORIZED
    })?;

    // Timing-safe comparison
    if !timing_safe_eq(received_token.as_bytes(), expected_token.as_bytes()) {
        let fp6 = token_fp6(&received_token);
        tracing::warn!(
            identity = %format!("token:{}", fp6),
            path = %req.uri().path(),
            event = "auth_failed",
            "Authentication failed: invalid token"
        );
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Success - log with fingerprint
    let fp6 = token_fp6(&received_token);
    tracing::debug!(
        identity = %format!("token:{}", fp6),
        path = %req.uri().path(),
        event = "authenticated",
        "Request authenticated"
    );

    Ok(next.run(req).await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        middleware,
        routing::get,
        Router,
    };
    use tower::ServiceExt;

    async fn test_handler() -> &'static str {
        "ok"
    }

    fn create_test_app() -> Router {
        Router::new()
            .route("/health", get(test_handler))
            .route("/pools/test/status", get(test_handler))
            .layer(middleware::from_fn(auth_middleware))
    }

    #[tokio::test]
    async fn test_health_endpoint_no_auth_required() {
        let app = create_test_app();

        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_protected_endpoint_requires_token() {
        std::env::set_var("LLORCH_API_TOKEN", "test-token-1234567890");

        let app = create_test_app();

        // No token -> 401
        let response = app
            .clone()
            .oneshot(Request::builder().uri("/pools/test/status").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        std::env::remove_var("LLORCH_API_TOKEN");
    }

    #[tokio::test]
    async fn test_valid_token_accepted() {
        std::env::set_var("LLORCH_API_TOKEN", "test-token-1234567890");

        let app = create_test_app();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/pools/test/status")
                    .header("Authorization", "Bearer test-token-1234567890")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        std::env::remove_var("LLORCH_API_TOKEN");
    }

    #[tokio::test]
    async fn test_invalid_token_rejected() {
        std::env::set_var("LLORCH_API_TOKEN", "test-token-1234567890");

        let app = create_test_app();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/pools/test/status")
                    .header("Authorization", "Bearer wrong-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        std::env::remove_var("LLORCH_API_TOKEN");
    }
}
