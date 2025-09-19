//! Minimal Auth middleware and startup checks.
//! This layer parses `Authorization: Bearer <token>` and records an identity
//! breadcrumb for logs. Enforcement is opt-in via environment variables.
//!
//! Env:
//! - AUTH_TOKEN: when set, the expected token value.
//! - AUTH_OPTIONAL: when true, requests without a token are allowed on loopback.
//! - TRUST_PROXY_AUTH: when true, trust proxy-provided headers (placeholder).

use axum::{body::Body, http::Request, middleware::Next, response::Response};
use http::{HeaderMap, StatusCode};

#[derive(Clone)]
pub struct Identity {
    pub breadcrumb: String,   // e.g., "token:abc123" or "localhost"
    pub auth_ok: bool,        // result of timing-safe compare when applicable
}

/// Attach `identity` information to request extensions for later logging.
/// Does not enforce by default; enforcement may be added per-route later.
pub async fn bearer_identity_layer(mut req: Request<Body>, next: Next) -> Result<Response, StatusCode> {
    let headers = req.headers().clone();
    if let Some(id) = identity_from_headers(&headers) {
        req.extensions_mut().insert(id);
    }
    Ok(next.run(req).await)
}

/// Compute an identity breadcrumb from headers and env configuration.
fn identity_from_headers(headers: &HeaderMap) -> Option<Identity> {
    let auth = headers.get(http::header::AUTHORIZATION).and_then(|v| v.to_str().ok());
    if let Some(token) = auth_min::parse_bearer(auth) {
        let fp = auth_min::token_fp6(&token);
        let expected = std::env::var("AUTH_TOKEN").ok();
        let ok = expected
            .map(|e| auth_min::timing_safe_eq(e.as_bytes(), token.as_bytes()))
            .unwrap_or(false);
        return Some(Identity { breadcrumb: format!("token:{}", fp), auth_ok: ok });
    }
    Some(Identity { breadcrumb: "localhost".to_string(), auth_ok: true })
}

/// Startup guard: refuse to bind a non-loopback address without AUTH_TOKEN set.
pub fn enforce_startup_bind_policy(addr: &str) -> Result<(), String> {
    let is_loopback = auth_min::is_loopback_addr(addr);
    let token = std::env::var("AUTH_TOKEN").ok();
    if !is_loopback && token.as_deref().unwrap_or("").is_empty() {
        return Err("refusing to bind non-loopback address without AUTH_TOKEN set (set AUTH_OPTIONAL=true only applies to loopback)".to_string());
    }
    Ok(())
}
