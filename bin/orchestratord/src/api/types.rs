use http::{HeaderMap, StatusCode};

// TODO(SECURITY): Replace X-API-Key with Bearer token authentication using auth-min
//
// This function should be migrated to use:
// 1. auth_min::parse_bearer() for Authorization: Bearer <token> headers
// 2. auth_min::timing_safe_eq() for secure token comparison
// 3. auth_min::token_fp6() for audit logging
//
// Current X-API-Key approach is a placeholder. Production should use Bearer tokens.
//
// See: .specs/12_auth-min-hardening.md (SEC-AUTH-3001)
pub fn require_api_key(headers: &HeaderMap) -> Result<(), StatusCode> {
    match headers.get("X-API-Key") {
        None => Err(StatusCode::UNAUTHORIZED),
        Some(v) => {
            let s = v.to_str().unwrap_or("");
            if s == "valid" {
                Ok(())
            } else {
                Err(StatusCode::FORBIDDEN)
            }
        }
    }
}

pub fn correlation_id_from(headers: &HeaderMap) -> String {
    headers
        .get("X-Correlation-Id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("11111111-1111-4111-8111-111111111111")
        .to_string()
}
