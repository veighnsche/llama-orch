//! Middleware stubs (auth, correlation-id, error mapping).
//! These are now functional layers applied in the router.

use axum::{
    body::Body,
    http::{HeaderMap, HeaderValue, Request},
    middleware::Next,
    response::Response,
};
use http::StatusCode;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct MiddlewareConfig {
    pub require_api_key: bool,
}

impl Default for MiddlewareConfig {
    fn default() -> Self {
        Self { require_api_key: true }
    }
}

/// Correlation-Id middleware: echo incoming header or generate a UUID.
pub async fn correlation_id_layer(
    mut req: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let corr = extract_or_generate_correlation_id(req.headers());
    // Attach to request extensions so handlers/services can use it if needed
    req.extensions_mut().insert(corr.clone());
    let mut res = next.run(req).await;
    res.headers_mut().insert("X-Correlation-Id", HeaderValue::from_str(&corr).unwrap());
    Ok(res)
}

fn extract_or_generate_correlation_id(headers: &HeaderMap) -> String {
    headers
        .get("X-Correlation-Id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string())
}

// TODO(SECURITY): Migrate X-API-Key to Bearer token authentication using auth-min
// 
// Current implementation uses hardcoded "valid" key which is insecure for production.
// Should migrate to proper Bearer token authentication with:
// 1. auth_min::parse_bearer() for header parsing
// 2. auth_min::timing_safe_eq() for token comparison
// 3. auth_min::token_fp6() for logging
// 4. LLORCH_API_TOKEN environment variable
//
// This middleware should eventually be replaced by bearer_identity_layer
// or unified into a single auth middleware using auth-min.
//
// See: .specs/12_auth-min-hardening.md (SEC-AUTH-3001)
/// API key middleware: enforce `X-API-Key: valid` on all routes except `/metrics`.
pub async fn api_key_layer(req: Request<Body>, next: Next) -> Result<Response, StatusCode> {
    if !should_require_api_key(req.uri().path()) {
        return Ok(next.run(req).await);
    }
    // Helper to build an error response with correlation id header. Since correlation_id_layer
    // may not run on early returns, compute/echo correlation id here from request headers.
    let build_err = |status: StatusCode| {
        let corr = extract_or_generate_correlation_id(req.headers());
        let mut resp = Response::builder().status(status).body(Body::empty()).unwrap();
        resp.headers_mut()
            .insert("X-Correlation-Id", HeaderValue::from_str(&corr).unwrap());
        resp
    };
    match req.headers().get("X-API-Key") {
        None => Ok(build_err(StatusCode::UNAUTHORIZED)),
        Some(v) => match v.to_str().ok() {
            Some("valid") => Ok(next.run(req).await),
            _ => Ok(build_err(StatusCode::FORBIDDEN)),
        },
    }
}

fn should_require_api_key(path: &str) -> bool {
    // Exempt metrics and root health checks if any in the future
    path != "/metrics"
}
