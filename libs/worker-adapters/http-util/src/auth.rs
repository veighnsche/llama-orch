use reqwest::header::AUTHORIZATION;

/// Returns an Authorization token from environment variable AUTH_TOKEN if set.
pub fn bearer_header_from_env() -> Option<String> {
    std::env::var("AUTH_TOKEN").ok()
}

/// Apply Authorization header if available from env to a RequestBuilder.
pub fn with_bearer_if_configured(rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    if let Some(token) = bearer_header_from_env() {
        let rb = rb.bearer_auth(&token);
        let val = format!("Bearer {}", token);
        rb.header(AUTHORIZATION, val)
    } else {
        rb
    }
}

/// Inject an Authorization: Bearer <token> header without reading environment variables.
/// Preferred helper for adapters.
pub fn with_bearer(rb: reqwest::RequestBuilder, token: impl AsRef<str>) -> reqwest::RequestBuilder {
    let t = token.as_ref();
    let rb = rb.bearer_auth(t);
    let val = format!("Bearer {}", t);
    rb.header(AUTHORIZATION, val)
}
