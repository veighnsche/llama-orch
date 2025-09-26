//! worker-adapters/http-util — shared HTTP client and streaming helpers for adapters.

use http::header::AUTHORIZATION;
use once_cell::sync::Lazy;
use reqwest::Client;
use std::time::Duration;

static DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

static CLIENT: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .pool_idle_timeout(Duration::from_secs(90))
        .pool_max_idle_per_host(8)
        .tcp_keepalive(Duration::from_secs(60))
        .timeout(DEFAULT_TIMEOUT)
        .build()
        .expect("http client")
});

pub fn client() -> &'static Client {
    &CLIENT
}

/// Return an Authorization header value if AUTH_TOKEN is configured.
pub fn bearer_header_from_env() -> Option<String> {
    std::env::var("AUTH_TOKEN").ok().map(|t| format!("Bearer {}", t))
}

/// Apply Authorization header if available from env to a RequestBuilder.
pub fn with_bearer_if_configured(rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    if let Some(v) = bearer_header_from_env() {
        rb.header(AUTHORIZATION, v)
    } else {
        rb
    }
}

/// Redact secrets in a log string, masking bearer tokens down to fp6.
pub fn redact_secrets(s: &str) -> String {
    // Very simple: replace `Authorization: Bearer <token>` patterns with masked fp6.
    let mut out = String::with_capacity(s.len());
    let mut last = 0usize;
    let pat = "Authorization: Bearer ";
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i + pat.len() <= bytes.len() {
        if &s[i..i + pat.len()] == pat {
            // Emit up to pattern
            out.push_str(&s[last..i]);
            // Extract token until whitespace or end
            let mut j = i + pat.len();
            while j < s.len() && !s.as_bytes()[j].is_ascii_whitespace() {
                j += 1;
            }
            let token = &s[i + pat.len()..j];
            let fp6 = auth_min::token_fp6(token);
            out.push_str("Authorization: Bearer ****");
            out.push_str(&fp6);
            last = j;
            i = j;
            continue;
        }
        i += 1;
    }
    out.push_str(&s[last..]);
    out
}
