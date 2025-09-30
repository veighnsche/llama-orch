//! Middleware for orchestratord.
//!
//! Provides correlation ID tracking for request tracing.
//! Authentication is handled by bearer_auth_middleware in app/auth_min.rs.

use axum::{
    body::Body,
    http::{HeaderMap, HeaderValue, Request},
    middleware::Next,
    response::Response,
};
use http::StatusCode;
use uuid::Uuid;

/// Correlation-Id middleware: echo incoming header or generate a UUID.
///
/// Attaches X-Correlation-Id to both request extensions and response headers
/// for distributed tracing and log correlation.
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
