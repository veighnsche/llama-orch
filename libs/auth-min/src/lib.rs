//! auth-min — minimal auth utilities shared across crates.
//!
//! Provides:
//! - timing-safe token compare
//! - token fingerprint (fp6) for logs
//! - bearer header parsing
//! - loopback bind detection
//! - proxy auth trust gate (TRUST_PROXY_AUTH)

use sha2::{Digest, Sha256};

/// Constant-time comparison to mitigate timing leaks.
pub fn timing_safe_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// Fingerprint the token into a 6-char lowercase hex suffix (fp6) for logs.
pub fn token_fp6(token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    let digest = hasher.finalize();
    let hex = hex::encode(digest);
    hex[0..6].to_string()
}

/// Parse `Authorization: Bearer <token>`; returns token string if valid.
pub fn parse_bearer(header_val: Option<&str>) -> Option<String> {
    let s = header_val?;
    let s = s.trim();
    if let Some(rest) = s.strip_prefix("Bearer ") {
        let t = rest.trim();
        if !t.is_empty() {
            return Some(t.to_string());
        }
    }
    None
}

/// Determine if an address string is loopback-only (127.0.0.1 or [::1]).
pub fn is_loopback_addr(addr: &str) -> bool {
    // Accept common forms like "127.0.0.1:8080" or "::1:8080"
    if let Some((host, _port)) = addr.rsplit_once(':') {
        return host == "127.0.0.1" || host == "::1" || host == "[::1]";
    }
    addr == "127.0.0.1" || addr == "::1" || addr == "[::1]"
}

/// Gate flag for trusting proxy-provided auth headers.
pub fn trust_proxy_auth() -> bool {
    std::env::var("TRUST_PROXY_AUTH")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}
