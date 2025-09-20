use http::HeaderMap;
use orchestratord::api::types::{correlation_id_from, require_api_key};

#[test]
fn require_api_key_missing_is_401() {
    let headers = HeaderMap::new();
    let err = require_api_key(&headers).unwrap_err();
    assert_eq!(err, http::StatusCode::UNAUTHORIZED);
}

#[test]
fn require_api_key_invalid_is_403() {
    let mut headers = HeaderMap::new();
    headers.insert("X-API-Key", "nope".parse().unwrap());
    let err = require_api_key(&headers).unwrap_err();
    assert_eq!(err, http::StatusCode::FORBIDDEN);
}

#[test]
fn require_api_key_valid_ok() {
    let mut headers = HeaderMap::new();
    headers.insert("X-API-Key", "valid".parse().unwrap());
    assert!(require_api_key(&headers).is_ok());
}

#[test]
fn correlation_id_from_prefers_header() {
    let mut headers = HeaderMap::new();
    headers.insert("X-Correlation-Id", "abc-123".parse().unwrap());
    assert_eq!(correlation_id_from(&headers), "abc-123");
}

#[test]
fn correlation_id_from_default_when_missing() {
    let headers = HeaderMap::new();
    let v = correlation_id_from(&headers);
    assert_eq!(v, "11111111-1111-4111-8111-111111111111");
}
