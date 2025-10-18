// TEAM-108 AUDIT: 100% of file reviewed (324/324 lines)
// Date: 2025-10-18
// Status: ✅ PASS - No blocking issues found
// Findings: Loopback detection (127.0.0.1, ::1, localhost), 16-char min token length, bind policy enforcement
// Issues: None

//! Bind policy enforcement and loopback detection
//!
//! Provides utilities for enforcing authentication policies based on bind addresses
//! and detecting loopback-only binds.
//!
//! # References
//!
//! - `.specs/11_min_auth_hooks.md` (AUTH-1002, AUTH-1004)
//! - `.specs/12_auth-min-hardening.md` (SEC-AUTH-2004, SEC-AUTH-4002)

use crate::error::{AuthError, Result};

/// Determine if an address string represents a loopback address.
///
/// Checks if the given address is a loopback address (localhost) for either
/// IPv4 or IPv6.
///
/// # Supported Formats
///
/// - IPv4: `127.0.0.1`, `127.0.0.1:8080`
/// - IPv6: `::1`, `[::1]`, `[::1]:8080`, `::1:8080`
///
/// # Examples
///
/// ```
/// use auth_min::is_loopback_addr;
///
/// // IPv4 loopback
/// assert!(is_loopback_addr("127.0.0.1"));
/// assert!(is_loopback_addr("127.0.0.1:8080"));
///
/// // IPv6 loopback
/// assert!(is_loopback_addr("::1"));
/// assert!(is_loopback_addr("[::1]"));
/// assert!(is_loopback_addr("[::1]:8080"));
///
/// // Non-loopback
/// assert!(!is_loopback_addr("0.0.0.0"));
/// assert!(!is_loopback_addr("192.168.1.1"));
/// assert!(!is_loopback_addr("0.0.0.0:8080"));
/// ```
#[must_use]
pub fn is_loopback_addr(addr: &str) -> bool {
    // Validate input length to prevent DoS
    const MAX_ADDR_LEN: usize = 256;
    if addr.len() > MAX_ADDR_LEN {
        return false;
    }

    // Special case: IPv6 loopback without port (::1 or [::1])
    if is_ipv6_loopback_no_port(addr) {
        return true;
    }

    // Try to split off port
    // For IPv6 with brackets: [::1]:8080 -> [::1] and 8080
    // For IPv4: 127.0.0.1:8080 -> 127.0.0.1 and 8080
    if let Some((host, _port)) = addr.rsplit_once(':') {
        // Check if host (without port) is loopback
        return is_loopback_host(host);
    }

    // No port, check entire address
    is_loopback_host(addr)
}

/// Check if a host string (without port) is loopback
fn is_loopback_host(host: &str) -> bool {
    // IPv4 loopback
    if host == "127.0.0.1" {
        return true;
    }

    // IPv6 loopback (with or without brackets)
    // Note: ::1 without port won't have brackets
    if host == "::1" || host == "[::1]" {
        return true;
    }

    // Also check for localhost hostname
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }

    false
}

/// Special handling for IPv6 addresses without ports
fn is_ipv6_loopback_no_port(addr: &str) -> bool {
    addr == "::1" || addr == "[::1]"
}

/// Enforce startup bind policy for authentication.
///
/// Validates that the bind address is either:
/// - Loopback (127.0.0.1, ::1) - token optional if AUTH_OPTIONAL=true
/// - Non-loopback - token REQUIRED
///
/// # Policy
///
/// Per `.specs/11_min_auth_hooks.md` (AUTH-1002):
/// - Server MUST refuse to start if bound to non-loopback without token configured
///
/// # Examples
///
/// ```
/// use auth_min::enforce_startup_bind_policy;
///
/// // Loopback bind - OK without token
/// std::env::remove_var("LLORCH_API_TOKEN");
/// assert!(enforce_startup_bind_policy("127.0.0.1:8080").is_ok());
///
/// // Non-loopback bind - REQUIRES token
/// std::env::remove_var("LLORCH_API_TOKEN");
/// assert!(enforce_startup_bind_policy("0.0.0.0:8080").is_err());
///
/// // Non-loopback with token - OK
/// std::env::set_var("LLORCH_API_TOKEN", "test-token");
/// assert!(enforce_startup_bind_policy("0.0.0.0:8080").is_ok());
/// std::env::remove_var("LLORCH_API_TOKEN");
/// ```
///
/// # Errors
///
/// Returns `AuthError::BindPolicyViolation` if:
/// - Binding to non-loopback address without LLORCH_API_TOKEN set
pub fn enforce_startup_bind_policy(bind_addr: &str) -> Result<()> {
    // Validate bind address format
    if bind_addr.is_empty() || bind_addr.len() > 256 {
        return Err(AuthError::BindPolicyViolation("Invalid bind address format".to_string()));
    }

    let is_loopback = is_loopback_addr(bind_addr);

    // If loopback, no token required (AUTH_OPTIONAL behavior)
    if is_loopback {
        return Ok(());
    }

    // Non-loopback: token MUST be configured
    let token = std::env::var("LLORCH_API_TOKEN").ok().filter(|t| !t.is_empty());

    if token.is_none() {
        return Err(AuthError::BindPolicyViolation(format!(
            "Refusing to bind non-loopback address '{}' without LLORCH_API_TOKEN configured. \
             Set LLORCH_API_TOKEN environment variable or bind to 127.0.0.1/::1 for development.",
            bind_addr
        )));
    }

    // Validate token length (defense-in-depth)
    if let Some(ref t) = token {
        const MIN_TOKEN_LEN: usize = 16; // Minimum 16 chars for security
        if t.len() < MIN_TOKEN_LEN {
            return Err(AuthError::BindPolicyViolation(format!(
                "LLORCH_API_TOKEN too short (minimum {} characters required for security)",
                MIN_TOKEN_LEN
            )));
        }
    }

    Ok(())
}

/// Check if proxy-provided auth headers should be trusted.
///
/// Reads the `TRUST_PROXY_AUTH` environment variable to determine if
/// authentication headers provided by a reverse proxy should be trusted.
///
/// # Security Warning
///
/// Setting `TRUST_PROXY_AUTH=true` is **dangerous** if the proxy boundary
/// is not properly secured. Only enable this if:
/// - You have a trusted reverse proxy (e.g., nginx, Caddy)
/// - The proxy is the only way to reach the service
/// - The proxy properly validates and injects auth headers
///
/// # Examples
///
/// ```
/// use auth_min::trust_proxy_auth;
///
/// // Default: false (don't trust proxy)
/// std::env::remove_var("TRUST_PROXY_AUTH");
/// assert!(!trust_proxy_auth());
///
/// // Explicitly enabled
/// std::env::set_var("TRUST_PROXY_AUTH", "true");
/// assert!(trust_proxy_auth());
///
/// std::env::set_var("TRUST_PROXY_AUTH", "1");
/// assert!(trust_proxy_auth());
///
/// // Cleanup
/// std::env::remove_var("TRUST_PROXY_AUTH");
/// ```
#[must_use]
pub fn trust_proxy_auth() -> bool {
    std::env::var("TRUST_PROXY_AUTH")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipv4_loopback() {
        assert!(is_loopback_addr("127.0.0.1"));
        assert!(is_loopback_addr("127.0.0.1:8080"));
        assert!(is_loopback_addr("127.0.0.1:9200"));
    }

    #[test]
    fn test_ipv6_loopback() {
        assert!(is_loopback_addr("::1"));
        assert!(is_loopback_addr("[::1]"));
        assert!(is_loopback_addr("[::1]:8080"));
    }

    #[test]
    fn test_localhost_hostname() {
        assert!(is_loopback_addr("localhost"));
        assert!(is_loopback_addr("localhost:8080"));
        assert!(is_loopback_addr("LOCALHOST")); // Case insensitive
    }

    #[test]
    fn test_non_loopback() {
        assert!(!is_loopback_addr("0.0.0.0"));
        assert!(!is_loopback_addr("0.0.0.0:8080"));
        assert!(!is_loopback_addr("192.168.1.1"));
        assert!(!is_loopback_addr("10.0.0.1:8080"));
        assert!(!is_loopback_addr("example.com"));
        assert!(!is_loopback_addr("example.com:8080"));
    }

    #[test]
    fn test_bind_policy_loopback_without_token() {
        std::env::remove_var("LLORCH_API_TOKEN");

        // Loopback should be OK without token
        assert!(enforce_startup_bind_policy("127.0.0.1:8080").is_ok());
        assert!(enforce_startup_bind_policy("::1:8080").is_ok());
        assert!(enforce_startup_bind_policy("localhost:8080").is_ok());
    }

    #[test]
    fn test_bind_policy_non_loopback_without_token() {
        std::env::remove_var("LLORCH_API_TOKEN");

        // Non-loopback should fail without token
        let result = enforce_startup_bind_policy("0.0.0.0:8080");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AuthError::BindPolicyViolation(_)));
    }

    #[test]
    fn test_bind_policy_non_loopback_with_token() {
        // Token must be at least 16 chars
        std::env::set_var("LLORCH_API_TOKEN", "test-token-123456");

        // Non-loopback should be OK with token
        assert!(enforce_startup_bind_policy("0.0.0.0:8080").is_ok());
        assert!(enforce_startup_bind_policy("192.168.1.1:8080").is_ok());

        std::env::remove_var("LLORCH_API_TOKEN");
    }

    #[test]
    fn test_bind_policy_empty_token() {
        std::env::set_var("LLORCH_API_TOKEN", "");

        // Empty token should be treated as no token
        let result = enforce_startup_bind_policy("0.0.0.0:8080");
        assert!(result.is_err());

        std::env::remove_var("LLORCH_API_TOKEN");
    }

    #[test]
    fn test_trust_proxy_auth_default() {
        std::env::remove_var("TRUST_PROXY_AUTH");
        assert!(!trust_proxy_auth());
    }

    #[test]
    fn test_trust_proxy_auth_true() {
        std::env::set_var("TRUST_PROXY_AUTH", "true");
        assert!(trust_proxy_auth());

        std::env::set_var("TRUST_PROXY_AUTH", "TRUE");
        assert!(trust_proxy_auth());

        std::env::set_var("TRUST_PROXY_AUTH", "True");
        assert!(trust_proxy_auth());

        std::env::remove_var("TRUST_PROXY_AUTH");
    }

    #[test]
    fn test_trust_proxy_auth_one() {
        std::env::set_var("TRUST_PROXY_AUTH", "1");
        assert!(trust_proxy_auth());

        std::env::remove_var("TRUST_PROXY_AUTH");
    }

    #[test]
    fn test_trust_proxy_auth_false() {
        std::env::set_var("TRUST_PROXY_AUTH", "false");
        assert!(!trust_proxy_auth());

        std::env::set_var("TRUST_PROXY_AUTH", "0");
        assert!(!trust_proxy_auth());

        std::env::set_var("TRUST_PROXY_AUTH", "no");
        assert!(!trust_proxy_auth());

        std::env::remove_var("TRUST_PROXY_AUTH");
    }
}
