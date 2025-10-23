// TEAM-108 AUDIT: 100% of file reviewed (153/153 lines)
// Date: 2025-10-18
// Status: ✅ PASS - No blocking issues found
// Findings: SHA-256 hash, 6-char output (24 bits), 8KB max token size, known vector test present
// Issues: None

//! Token fingerprinting for safe logging
//!
//! Provides SHA-256 based token fingerprints (fp6) that are safe to include in logs
//! without exposing the actual token value.
//!
//! # Security Properties
//!
//! - **Non-reversible**: SHA-256 is a one-way hash function
//! - **Collision resistant**: 24-bit space (16.7M combinations) sufficient for correlation
//! - **Log-safe**: 6 hex characters are concise and don't leak sensitive data
//!
//! # References
//!
//! - `.specs/12_auth-min-hardening.md` (SEC-AUTH-2002)

use sha2::{Digest, Sha256};

/// Generate a 6-character hexadecimal fingerprint of a token.
///
/// This function creates a non-reversible fingerprint of a token using SHA-256,
/// returning the first 6 hexadecimal characters. This is safe to include in logs
/// for correlation and auditing purposes.
///
/// # Security
///
/// The fingerprint is derived from SHA-256, making it:
/// - **Non-reversible**: Cannot recover the original token from the fingerprint
/// - **Collision resistant**: 24-bit space provides sufficient uniqueness for audit trails
/// - **Log-safe**: Short enough for grep/correlation without PII concerns
///
/// # Examples
///
/// ```
/// use auth_min::token_fp6;
///
/// let token = "secret-token-abc123";
/// let fp6 = token_fp6(token);
///
/// // Fingerprint is deterministic
/// assert_eq!(fp6, token_fp6(token));
///
/// // Different tokens produce different fingerprints
/// assert_ne!(fp6, token_fp6("different-token"));
///
/// // Safe to log
/// println!("Authenticated: token:{}", fp6);
/// ```
///
/// # Format
///
/// Returns a 6-character lowercase hexadecimal string (e.g., "a3f2c1").
#[must_use]
pub fn token_fp6(token: &str) -> String {
    // Validate input length to prevent DoS via extremely long tokens
    const MAX_TOKEN_LEN: usize = 8192; // 8KB max token size
    if token.len() > MAX_TOKEN_LEN {
        // For extremely long inputs, hash in chunks to prevent memory issues
        // This is defense-in-depth; callers should validate token length
        let truncated = &token[..MAX_TOKEN_LEN];
        let mut hasher = Sha256::new();
        hasher.update(truncated.as_bytes());
        hasher.update(b"[truncated]"); // Marker for truncation
        let digest = hasher.finalize();
        let hex = hex::encode(digest);
        return hex[0..6].to_string();
    }

    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    let digest = hasher.finalize();
    let hex = hex::encode(digest);

    // Take first 6 characters (24 bits)
    hex[0..6].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic() {
        let token = "secret-token-abc123";
        let fp1 = token_fp6(token);
        let fp2 = token_fp6(token);
        assert_eq!(fp1, fp2, "Fingerprint should be deterministic");
    }

    #[test]
    fn test_different_tokens_different_fingerprints() {
        let token1 = "secret-token-abc123";
        let token2 = "different-token-xyz789";
        let fp1 = token_fp6(token1);
        let fp2 = token_fp6(token2);
        assert_ne!(fp1, fp2, "Different tokens should produce different fingerprints");
    }

    #[test]
    fn test_fingerprint_length() {
        let token = "any-token";
        let fp = token_fp6(token);
        assert_eq!(fp.len(), 6, "Fingerprint should be exactly 6 characters");
    }

    #[test]
    fn test_fingerprint_format() {
        let token = "test-token";
        let fp = token_fp6(token);

        // Should be lowercase hexadecimal
        assert!(
            fp.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "Fingerprint should be lowercase hex: {}",
            fp
        );
    }

    #[test]
    fn test_empty_token() {
        let token = "";
        let fp = token_fp6(token);
        assert_eq!(fp.len(), 6, "Empty token should still produce 6-char fingerprint");
    }

    #[test]
    fn test_collision_resistance() {
        // Test that similar tokens produce different fingerprints
        let tokens = vec!["token-a", "token-b", "token-c", "atoken-", "btoken-", "ctoken-"];

        let mut fingerprints = std::collections::HashSet::new();
        for token in &tokens {
            let fp = token_fp6(token);
            assert!(
                fingerprints.insert(fp.clone()),
                "Collision detected for token: {} (fp: {})",
                token,
                fp
            );
        }
    }

    #[test]
    fn test_known_vectors() {
        // Test with known SHA-256 values to ensure correctness
        let token = "test";
        let fp = token_fp6(token);

        // SHA-256("test") = 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08
        // First 6 chars: 9f86d0
        assert_eq!(fp, "9f86d0");
    }
}
