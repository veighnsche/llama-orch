/// Parse a Retry-After header value into milliseconds.
///
/// Supports the "delta-seconds" form per RFC 7231 (e.g., "120"). Returns `None` if parsing fails.
/// Date formats are not currently supported.
pub fn parse_retry_after(s: &str) -> Option<u64> {
    let trimmed = s.trim();
    if let Ok(secs) = trimmed.parse::<u64>() {
        return Some(secs.saturating_mul(1000));
    }
    None
}

/// Returns true if the given HTTP status code is considered non-retriable by default.
///
/// Defaults: 400/401/403/404/422 are non-retriable. This can be extended per adapter needs.
pub fn is_non_retriable_status(status: u16) -> bool {
    matches!(status, 400 | 401 | 403 | 404 | 422)
}

/// Returns true if the given HTTP status code is considered retriable by default.
///
/// Defaults: 429 and all 5xx are retriable. Also retriable for IO/timeouts (handled at error level, not by status).
pub fn is_retriable_status(status: u16) -> bool {
    status == 429 || (500..=599).contains(&status)
}
