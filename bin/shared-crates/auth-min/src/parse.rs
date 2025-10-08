//! Bearer token parsing from HTTP Authorization headers
//!
//! Provides robust parsing of `Authorization: Bearer <token>` headers with
//! proper validation and error handling.
//!
//! # References
//!
//! - RFC 6750: The OAuth 2.0 Authorization Framework: Bearer Token Usage
//! - `.specs/12_auth-min-hardening.md` (SEC-AUTH-2003)

/// Parse a Bearer token from an Authorization header value.
///
/// Extracts the token from an `Authorization: Bearer <token>` header,
/// handling whitespace and validation.
///
/// # Format
///
/// Expects the header value to be in the format: `Bearer <token>`
/// - The `Bearer` prefix is case-sensitive (per RFC 6750)
/// - Whitespace around the token is trimmed
/// - Empty tokens after the prefix are rejected
///
/// # Examples
///
/// ```
/// use auth_min::parse_bearer;
///
/// // Valid Bearer token
/// assert_eq!(
///     parse_bearer(Some("Bearer abc123")),
///     Some("abc123".to_string())
/// );
///
/// // Whitespace is trimmed
/// assert_eq!(
///     parse_bearer(Some("  Bearer  abc123  ")),
///     Some("abc123".to_string())
/// );
///
/// // Missing Bearer prefix
/// assert_eq!(parse_bearer(Some("abc123")), None);
///
/// // Empty token
/// assert_eq!(parse_bearer(Some("Bearer ")), None);
///
/// // None input
/// assert_eq!(parse_bearer(None), None);
/// ```
///
/// # Returns
///
/// - `Some(token)` if the header is valid and contains a non-empty token
/// - `None` if the header is missing, malformed, or contains an empty token
#[must_use]
pub fn parse_bearer(header_val: Option<&str>) -> Option<String> {
    let s = header_val?;

    // Validate header length to prevent DoS
    const MAX_HEADER_LEN: usize = 8192; // 8KB max header size
    if s.len() > MAX_HEADER_LEN {
        return None;
    }

    let s = s.trim();

    // Check for Bearer prefix (case-sensitive per RFC 6750)
    let rest = s.strip_prefix("Bearer ")?;

    // Trim whitespace from token
    let token = rest.trim();

    // Reject empty tokens
    if token.is_empty() {
        return None;
    }

    // Validate token doesn't contain control characters (security hardening)
    if token.chars().any(|c| c.is_control()) {
        return None;
    }

    Some(token.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_bearer_token() {
        let result = parse_bearer(Some("Bearer abc123"));
        assert_eq!(result, Some("abc123".to_string()));
    }

    #[test]
    fn test_bearer_with_whitespace() {
        let result = parse_bearer(Some("  Bearer  abc123  "));
        assert_eq!(result, Some("abc123".to_string()));
    }

    #[test]
    fn test_bearer_with_token_whitespace() {
        let result = parse_bearer(Some("Bearer   abc123   "));
        assert_eq!(result, Some("abc123".to_string()));
    }

    #[test]
    fn test_missing_bearer_prefix() {
        let result = parse_bearer(Some("abc123"));
        assert_eq!(result, None);
    }

    #[test]
    fn test_empty_token() {
        let result = parse_bearer(Some("Bearer "));
        assert_eq!(result, None);
    }

    #[test]
    fn test_empty_token_with_whitespace() {
        let result = parse_bearer(Some("Bearer    "));
        assert_eq!(result, None);
    }

    #[test]
    fn test_none_input() {
        let result = parse_bearer(None);
        assert_eq!(result, None);
    }

    #[test]
    fn test_empty_string() {
        let result = parse_bearer(Some(""));
        assert_eq!(result, None);
    }

    #[test]
    fn test_only_whitespace() {
        let result = parse_bearer(Some("   "));
        assert_eq!(result, None);
    }

    #[test]
    fn test_case_sensitive_bearer() {
        // "bearer" (lowercase) should not match
        let result = parse_bearer(Some("bearer abc123"));
        assert_eq!(result, None);

        // "BEARER" (uppercase) should not match
        let result = parse_bearer(Some("BEARER abc123"));
        assert_eq!(result, None);
    }

    #[test]
    fn test_token_with_spaces() {
        // Tokens with internal spaces are valid (will be trimmed at edges only)
        let result = parse_bearer(Some("Bearer token with spaces"));
        assert_eq!(result, Some("token with spaces".to_string()));
    }

    #[test]
    fn test_long_token() {
        let long_token = "a".repeat(256);
        let header = format!("Bearer {}", long_token);
        let result = parse_bearer(Some(&header));
        assert_eq!(result, Some(long_token));
    }

    #[test]
    fn test_special_characters() {
        // Tokens can contain special characters
        let result = parse_bearer(Some("Bearer abc-123_xyz.456"));
        assert_eq!(result, Some("abc-123_xyz.456".to_string()));
    }
}
