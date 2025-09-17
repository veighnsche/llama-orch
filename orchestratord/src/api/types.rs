use http::{HeaderMap, StatusCode};

pub fn require_api_key(headers: &HeaderMap) -> Result<(), StatusCode> {
    match headers.get("X-API-Key") {
        None => Err(StatusCode::UNAUTHORIZED),
        Some(v) => {
            let s = v.to_str().unwrap_or("");
            if s == "valid" { Ok(()) } else { Err(StatusCode::FORBIDDEN) }
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
