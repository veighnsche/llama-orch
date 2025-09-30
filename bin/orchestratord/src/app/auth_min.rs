//! Bearer token authentication middleware using auth-min.
//!
//! Provides unified Bearer token authentication for all orchestratord endpoints.
//! Uses timing-safe comparison and token fingerprinting for security.
//!
//! Environment Variables:
//! - LLORCH_API_TOKEN: Required token for authentication (min 16 chars)
//! - Loopback binds (127.0.0.1, ::1): Token optional
//! - Non-loopback binds: Token REQUIRED (enforced at startup)

use axum::{body::Body, http::Request, middleware::Next, response::Response};
use http::{HeaderMap, StatusCode};
use tracing::{debug, warn};

#[derive(Clone)]
pub struct Identity {
    pub breadcrumb: String, // e.g., "token:abc123" or "localhost"
    pub auth_ok: bool,      // result of timing-safe compare when applicable
}

/// Bearer token authentication middleware.
///
/// Validates Bearer tokens using auth-min with timing-safe comparison.
/// Exempts /metrics endpoint from authentication.
///
/// Returns 401 Unauthorized if:
/// - Token is missing or invalid
/// - Token doesn't match LLORCH_API_TOKEN
pub async fn bearer_auth_middleware(
    mut req: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    // Exempt /metrics from authentication
    if req.uri().path() == "/metrics" {
        return Ok(next.run(req).await);
    }

    let headers = req.headers().clone();

    // Read expected token from environment
    let expected_token = std::env::var("LLORCH_API_TOKEN").ok().filter(|t| !t.is_empty());

    // If no token configured, allow all requests (backward compat for loopback)
    let expected_token = match expected_token {
        Some(t) => t,
        None => {
            // No token required - allow request
            let id = Identity { breadcrumb: "localhost".to_string(), auth_ok: true };
            req.extensions_mut().insert(id);
            return Ok(next.run(req).await);
        }
    };

    // Extract and parse Bearer token
    let auth = headers.get(http::header::AUTHORIZATION).and_then(|v| v.to_str().ok());

    let received_token = match auth_min::parse_bearer(auth) {
        Some(token) => token,
        None => {
            warn!(
                path = %req.uri().path(),
                "Missing or invalid Bearer token"
            );
            return Err(StatusCode::UNAUTHORIZED);
        }
    };

    // Timing-safe comparison
    if !auth_min::timing_safe_eq(received_token.as_bytes(), expected_token.as_bytes()) {
        let fp6 = auth_min::token_fp6(&received_token);
        warn!(
            identity = %format!("token:{}", fp6),
            path = %req.uri().path(),
            event = "auth_failed",
            "Authentication failed: invalid token"
        );
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Success - attach identity and log
    let fp6 = auth_min::token_fp6(&received_token);
    let id = Identity { breadcrumb: format!("token:{}", fp6), auth_ok: true };

    debug!(
        identity = %id.breadcrumb,
        path = %req.uri().path(),
        event = "authenticated",
        "Request authenticated"
    );

    req.extensions_mut().insert(id);
    Ok(next.run(req).await)
}

/// Startup guard: refuse to bind a non-loopback address without LLORCH_API_TOKEN set.
pub fn enforce_startup_bind_policy(addr: &str) -> Result<(), String> {
    let is_loopback = auth_min::is_loopback_addr(addr);
    let token = std::env::var("LLORCH_API_TOKEN").ok();
    if !is_loopback && token.as_deref().unwrap_or("").is_empty() {
        return Err(
            "refusing to bind non-loopback address without LLORCH_API_TOKEN set".to_string()
        );
    }
    Ok(())
}
