// redaction.rs — Secret and PII redaction helpers
// Implements ORCH-3302: Narration MUST NOT include secrets or PII

use regex::Regex;
use std::sync::OnceLock;

/// Redaction policy for secrets and PII.
#[derive(Debug, Clone)]
pub struct RedactionPolicy {
    /// Mask bearer tokens
    pub mask_bearer_tokens: bool,
    /// Mask API keys
    pub mask_api_keys: bool,
    /// Mask UUIDs (potential session/correlation IDs in human text)
    pub mask_uuids: bool,
    /// Replacement string for redacted content
    pub replacement: String,
}

impl Default for RedactionPolicy {
    fn default() -> Self {
        Self {
            mask_bearer_tokens: true,
            mask_api_keys: true,
            mask_uuids: false, // UUIDs are usually safe in narration
            replacement: "[REDACTED]".to_string(),
        }
    }
}

/// Regex patterns for secret detection (compiled once, reused)
static BEARER_TOKEN_PATTERN: OnceLock<Regex> = OnceLock::new();
static API_KEY_PATTERN: OnceLock<Regex> = OnceLock::new();
static UUID_PATTERN: OnceLock<Regex> = OnceLock::new();

fn bearer_token_regex() -> &'static Regex {
    BEARER_TOKEN_PATTERN.get_or_init(|| {
        // Match "Bearer <token>" or "bearer <token>" with various token formats
        Regex::new(r"(?i)bearer\s+[a-zA-Z0-9_\-\.=]+").unwrap()
    })
}

fn api_key_regex() -> &'static Regex {
    API_KEY_PATTERN.get_or_init(|| {
        // Match common API key patterns: "api_key=...", "apikey=...", "key=..."
        Regex::new(r"(?i)(api_?key|key|token|secret|password)\s*[=:]\s*[a-zA-Z0-9_\-\.]+").unwrap()
    })
}

fn uuid_regex() -> &'static Regex {
    UUID_PATTERN.get_or_init(|| {
        // Match UUIDs (8-4-4-4-12 hex format)
        Regex::new(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
            .unwrap()
    })
}

/// Redact secrets and PII from a string according to policy.
/// Implements ORCH-3302.
///
/// # Example
/// ```rust
/// use observability_narration_core::{redact_secrets, RedactionPolicy};
///
/// let text = "Authorization: Bearer abc123xyz";
/// let redacted = redact_secrets(text, RedactionPolicy::default());
/// assert_eq!(redacted, "Authorization: [REDACTED]");
/// ```
pub fn redact_secrets(text: &str, policy: RedactionPolicy) -> String {
    let mut result = text.to_string();

    if policy.mask_bearer_tokens {
        result = bearer_token_regex().replace_all(&result, &policy.replacement).to_string();
    }

    if policy.mask_api_keys {
        result = api_key_regex().replace_all(&result, &policy.replacement).to_string();
    }

    if policy.mask_uuids {
        result = uuid_regex().replace_all(&result, &policy.replacement).to_string();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redact_bearer_token() {
        let text = "Authorization: Bearer abc123xyz";
        let redacted = redact_secrets(text, RedactionPolicy::default());
        assert_eq!(redacted, "Authorization: [REDACTED]");
    }

    #[test]
    fn test_redact_api_key() {
        let text = "api_key=secret123";
        let redacted = redact_secrets(text, RedactionPolicy::default());
        assert_eq!(redacted, "[REDACTED]");
    }

    #[test]
    fn test_redact_multiple_secrets() {
        let text = "Bearer token123 and api_key=secret456";
        let redacted = redact_secrets(text, RedactionPolicy::default());
        assert!(redacted.contains("[REDACTED]"));
        assert!(!redacted.contains("token123"));
        assert!(!redacted.contains("secret456"));
    }

    #[test]
    fn test_no_redaction_when_no_secrets() {
        let text = "Accepted request; queued at position 3";
        let redacted = redact_secrets(text, RedactionPolicy::default());
        assert_eq!(redacted, text);
    }

    #[test]
    fn test_custom_replacement() {
        let text = "Bearer abc123";
        let policy = RedactionPolicy { replacement: "***".to_string(), ..Default::default() };
        let redacted = redact_secrets(text, policy);
        assert_eq!(redacted, "***");
    }

    #[test]
    fn test_case_insensitive_bearer() {
        let text = "BEARER abc123";
        let redacted = redact_secrets(text, RedactionPolicy::default());
        assert_eq!(redacted, "[REDACTED]");
    }

    #[test]
    fn test_uuid_redaction_when_enabled() {
        let text = "session_id: 550e8400-e29b-41d4-a716-446655440000";
        let mut policy = RedactionPolicy::default();
        policy.mask_uuids = true;
        let redacted = redact_secrets(text, policy);
        assert_eq!(redacted, "session_id: [REDACTED]");
    }

    #[test]
    fn test_uuid_not_redacted_by_default() {
        let text = "session_id: 550e8400-e29b-41d4-a716-446655440000";
        let redacted = redact_secrets(text, RedactionPolicy::default());
        assert_eq!(redacted, text); // UUIDs not redacted by default
    }
}
