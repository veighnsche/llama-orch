// Created by: TEAM-102
// Purpose: Authentication middleware using auth-min shared crate
// Implements: Bearer token validation, timing-safe comparison, token fingerprinting

use auth_min::{parse_bearer, timing_safe_eq, token_fp6};
use axum::{
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

use crate::http::routes::AppState;

/// TEAM-102: Authentication middleware
/// 
/// Validates Bearer tokens using timing-safe comparison.
/// Logs authentication events with token fingerprints (never raw tokens).
/// 
/// Returns 401 Unauthorized if:
/// - No Authorization header
/// - Invalid Bearer token format
/// - Token doesn't match expected value
pub async fn auth_middleware(
    State(state): State<AppState>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, impl IntoResponse> {
    // TEAM-102: Parse Authorization header
    let auth_header = req
        .headers()
        .get("authorization")
        .and_then(|h| h.to_str().ok());

    // TEAM-102: Parse Bearer token (RFC 6750 compliant)
    let token = match parse_bearer(auth_header) {
        Some(t) => t,
        None => {
            tracing::warn!("auth failed: missing or invalid Authorization header");
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(json!({
                    "error": {
                        "code": "MISSING_API_KEY",
                        "message": "Missing API key"
                    }
                })),
            ));
        }
    };

    // TEAM-102: Timing-safe comparison (prevents CWE-208 timing attacks)
    if !timing_safe_eq(token.as_bytes(), state.expected_token.as_bytes()) {
        let fp = token_fp6(&token);
        tracing::warn!(
            identity = %format!("token:{}", fp),
            "auth failed: invalid token"
        );
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(json!({
                "error": {
                    "code": "INVALID_API_KEY",
                    "message": "Invalid or missing API key"
                }
            })),
        ));
    }

    // TEAM-102: Success - log with fingerprint (never raw token)
    let fp = token_fp6(&token);
    tracing::info!(identity = %format!("token:{}", fp), "authenticated");

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
    use std::sync::Arc;
    use crate::registry::WorkerRegistry;
    use model_catalog::ModelCatalog;
    use crate::provisioner::ModelProvisioner;
    use rbee_hive::download_tracker::DownloadTracker;
    use std::path::PathBuf;

    async fn test_handler() -> &'static str {
        "OK"
    }

    fn create_test_app(expected_token: String) -> Router {
        let state = AppState {
            registry: Arc::new(WorkerRegistry::new()),
            model_catalog: Arc::new(ModelCatalog::new(":memory:".to_string())),
            provisioner: Arc::new(ModelProvisioner::new(PathBuf::from("/tmp"))),
            download_tracker: Arc::new(DownloadTracker::new()),
            server_addr: "127.0.0.1:9200".parse().unwrap(),
            expected_token, // TEAM-102
        };

        Router::new()
            .route("/test", get(test_handler))
            .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
            .with_state(state)
    }

    #[tokio::test]
    async fn test_auth_success() {
        let app = create_test_app("test-token-12345".to_string());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/test")
                    .header("Authorization", "Bearer test-token-12345")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_missing_header() {
        let app = create_test_app("test-token-12345".to_string());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_invalid_token() {
        let app = create_test_app("correct-token".to_string());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/test")
                    .header("Authorization", "Bearer wrong-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_invalid_format() {
        let app = create_test_app("test-token".to_string());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/test")
                    .header("Authorization", "InvalidFormat")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
}
