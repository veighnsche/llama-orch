//! Hardening tests for edge cases and DoS prevention

use crate::{enforce_startup_bind_policy, is_loopback_addr, parse_bearer, token_fp6};

#[test]
fn test_parse_bearer_rejects_control_characters() {
    // Tokens with control characters should be rejected
    assert_eq!(parse_bearer(Some("Bearer token\x00with\x00nulls")), None);
    assert_eq!(parse_bearer(Some("Bearer token\nwith\nnewlines")), None);
    assert_eq!(parse_bearer(Some("Bearer token\rwith\rcarriage")), None);
    assert_eq!(parse_bearer(Some("Bearer token\twith\ttabs")), None);
}

#[test]
fn test_parse_bearer_handles_very_long_headers() {
    // Very long headers should be rejected (DoS prevention)
    let long_header = format!("Bearer {}", "a".repeat(10000));
    assert_eq!(parse_bearer(Some(&long_header)), None);
}

#[test]
fn test_token_fp6_handles_very_long_tokens() {
    // Very long tokens should still produce valid fingerprints
    let long_token = "a".repeat(10000);
    let fp = token_fp6(&long_token);

    // Should still be 6 chars
    assert_eq!(fp.len(), 6);

    // Should be valid hex
    assert!(fp.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn test_token_fp6_empty_vs_nonempty() {
    // Empty and non-empty tokens should produce different fingerprints
    let fp_empty = token_fp6("");
    let fp_space = token_fp6(" ");
    let fp_a = token_fp6("a");

    assert_ne!(fp_empty, fp_space);
    assert_ne!(fp_empty, fp_a);
    assert_ne!(fp_space, fp_a);
}

#[test]
fn test_is_loopback_rejects_very_long_addresses() {
    // Very long addresses should be rejected (DoS prevention)
    let long_addr = "127.0.0.1:".to_string() + &"8".repeat(1000);
    assert!(!is_loopback_addr(&long_addr));
}

#[test]
fn test_is_loopback_handles_malformed_addresses() {
    // Malformed addresses should not panic
    assert!(!is_loopback_addr(":::::::"));
    assert!(!is_loopback_addr("[[[[["));
    assert!(!is_loopback_addr("]]]]]"));
    assert!(!is_loopback_addr("127.0.0.1:::::8080"));
}

#[test]
fn test_bind_policy_rejects_empty_address() {
    let result = enforce_startup_bind_policy("");
    assert!(result.is_err());
}

#[test]
fn test_bind_policy_rejects_very_long_address() {
    let long_addr = "0.0.0.0:".to_string() + &"8".repeat(1000);
    let result = enforce_startup_bind_policy(&long_addr);
    assert!(result.is_err());
}

#[test]
fn test_bind_policy_rejects_short_token() {
    // Set a token that's too short
    std::env::set_var("LLORCH_API_TOKEN", "short");

    let result = enforce_startup_bind_policy("0.0.0.0:8080");
    assert!(result.is_err());

    std::env::remove_var("LLORCH_API_TOKEN");
}

#[test]
fn test_bind_policy_accepts_minimum_length_token() {
    // 16 chars is minimum
    std::env::set_var("LLORCH_API_TOKEN", "1234567890123456");

    let result = enforce_startup_bind_policy("0.0.0.0:8080");
    assert!(result.is_ok());

    std::env::remove_var("LLORCH_API_TOKEN");
}

#[test]
fn test_timing_safe_eq_with_empty_slices() {
    use crate::timing_safe_eq;

    // Empty slices should compare equal
    assert!(timing_safe_eq(b"", b""));

    // Empty vs non-empty should be false
    assert!(!timing_safe_eq(b"", b"a"));
    assert!(!timing_safe_eq(b"a", b""));
}

#[test]
fn test_timing_safe_eq_with_single_byte() {
    use crate::timing_safe_eq;

    // Single byte comparisons
    assert!(timing_safe_eq(b"a", b"a"));
    assert!(!timing_safe_eq(b"a", b"b"));
    assert!(!timing_safe_eq(b"\x00", b"\xff"));
}

#[test]
fn test_parse_bearer_unicode_tokens() {
    // Unicode tokens should be accepted (UTF-8 is valid)
    let result = parse_bearer(Some("Bearer token-with-émojis-🔒"));
    assert!(result.is_some());
    assert_eq!(result.unwrap(), "token-with-émojis-🔒");
}

#[test]
fn test_token_fp6_unicode() {
    // Unicode tokens should produce valid fingerprints
    let fp = token_fp6("token-with-émojis-🔒");
    assert_eq!(fp.len(), 6);
    assert!(fp.chars().all(|c| c.is_ascii_hexdigit()));
}
