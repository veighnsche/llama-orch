//! Common types and utilities for API handlers.
//!
//! Authentication is now handled by bearer_auth_middleware in app/auth_min.rs.
//! This module provides helper functions for extracting request metadata.

use http::HeaderMap;

/// Extract correlation ID from request headers.
///
/// Returns the X-Correlation-Id header value, or a default UUID if not present.
/// The correlation_id_layer middleware ensures this header is always set.
pub fn correlation_id_from(headers: &HeaderMap) -> String {
    headers
        .get("X-Correlation-Id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("11111111-1111-4111-8111-111111111111")
        .to_string()
}
