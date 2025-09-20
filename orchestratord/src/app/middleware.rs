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
        Self {
            require_api_key: true,
        }
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
    res.headers_mut()
        .insert("X-Correlation-Id", HeaderValue::from_str(&corr).unwrap());
    Ok(res)
}

fn extract_or_generate_correlation_id(headers: &HeaderMap) -> String {
    headers
        .get("X-Correlation-Id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string())
}

/// API key middleware: enforce `X-API-Key: valid` on all routes except `/metrics`.
pub async fn api_key_layer(req: Request<Body>, next: Next) -> Result<Response, StatusCode> {
    if !should_require_api_key(req.uri().path()) {
        return Ok(next.run(req).await);
    }
    match req.headers().get("X-API-Key") {
        None => Err(StatusCode::UNAUTHORIZED),
        Some(v) => match v.to_str().ok() {
            Some("valid") => Ok(next.run(req).await),
            _ => Err(StatusCode::FORBIDDEN),
        },
    }
}

fn should_require_api_key(path: &str) -> bool {
    // Exempt metrics and root health checks if any in the future
    path != "/metrics"
}
