use http::HeaderMap;

/// Require an API key in `X-API-Key` header. Planning-only stub that accepts "valid".
pub fn require_api_key(headers: &HeaderMap) -> Result<(), http::StatusCode> {
    match headers.get("X-API-Key").and_then(|v| v.to_str().ok()) {
        None => Err(http::StatusCode::UNAUTHORIZED),
        Some("valid") => Ok(()),
        Some(_) => Err(http::StatusCode::FORBIDDEN),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ORCH-AUTH-0001: require X-API-Key, accept only "valid" in planning stub
    #[test]
    fn test_orch_auth_0001_require_api_key() {
        let mut h = HeaderMap::new();
        // missing header -> 401
        assert_eq!(require_api_key(&h), Err(http::StatusCode::UNAUTHORIZED));
        // wrong value -> 403
        h.insert("X-API-Key", "nope".parse().unwrap());
        assert_eq!(require_api_key(&h), Err(http::StatusCode::FORBIDDEN));
        // valid -> Ok
        *h.get_mut("X-API-Key").unwrap() = "valid".parse().unwrap();
        assert!(require_api_key(&h).is_ok());
    }
}
