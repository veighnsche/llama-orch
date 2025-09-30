use once_cell::sync::Lazy;
use regex::Regex;

/// Redact secrets in a log string, masking bearer tokens, X-API-Key, and common token/api_key patterns to fp6.
pub fn redact_secrets(s: &str) -> String {
    // Case-insensitive header patterns and common key/value token appearances.
    // Replace captures with masked fp6 suffix.
    static AUTH_BEARER_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?i)Authorization\s*:\s*Bearer\s+(?P<t>[A-Za-z0-9._\-]+)").unwrap()
    });
    static X_API_KEY_RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?i)X-API-Key\s*:\s*(?P<t>[A-Za-z0-9._\-]+)").unwrap());
    static KV_TOKEN_RE: Lazy<Regex> = Lazy::new(|| {
        // Matches token or api_key in simple JSON/text: token:"..." or api_key=...
        // Use a raw string with # so we can include double quotes without escaping.
        Regex::new(r#"(?i)(token|api[_-]?key)\s*[:=]\s*"?(?P<t>[A-Za-z0-9._\-]{8,})"?"#).unwrap()
    });

    let mut out = AUTH_BEARER_RE
        .replace_all(s, |caps: &regex::Captures| {
            let t = &caps["t"];
            let fp6 = auth_min::token_fp6(t);
            format!("Authorization: Bearer ****{}", fp6)
        })
        .into_owned();

    out = X_API_KEY_RE
        .replace_all(&out, |caps: &regex::Captures| {
            let t = &caps["t"];
            let fp6 = auth_min::token_fp6(t);
            format!("X-API-Key: ****{}", fp6)
        })
        .into_owned();

    out = KV_TOKEN_RE
        .replace_all(&out, |caps: &regex::Captures| {
            let t = &caps["t"];
            let fp6 = auth_min::token_fp6(t);
            // Preserve the key name as matched in group 1, normalize formatting
            format!("{}: \"****{}\"", &caps[1], fp6)
        })
        .into_owned();

    out
}
