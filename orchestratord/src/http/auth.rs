use http::HeaderMap;

/// Require an API key in `X-API-Key` header. Planning-only stub that accepts "valid".
pub fn require_api_key(headers: &HeaderMap) -> Result<(), http::StatusCode> {
    match headers.get("X-API-Key").and_then(|v| v.to_str().ok()) {
        None => Err(http::StatusCode::UNAUTHORIZED),
        Some("valid") => Ok(()),
        Some(_) => Err(http::StatusCode::FORBIDDEN),
    }
}
