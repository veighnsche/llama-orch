//! Backpressure helpers (planning-only).

use http::HeaderMap;
use serde_json::json;

/// Policy label attached to 429 bodies (for advisory debugging, not a contract to clients).
#[derive(Debug, Clone)]
pub enum PolicyLabel {
    Reject,
    DropLru,
    ShedLowPriority,
}

/// Computed backoff to surface via headers.
#[derive(Debug, Clone, Copy)]
pub struct Backoff {
    pub retry_after_seconds: u64,
    pub x_backoff_ms: u64,
}

/// Compute advisory policy label for a given condition.
/// Planning-only: to be implemented when admission policies are fully wired.
pub fn compute_policy_label(_ctx: ()) -> PolicyLabel {
    // TODO(impl): read from admission decision
    PolicyLabel::Reject
}

/// Build standard 429 headers (Retry-After, X-Backoff-Ms) for backpressure.
/// Planning-only: returns empty headers for now.
pub fn build_429_headers(_backoff: Backoff) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        "Retry-After",
        _backoff.retry_after_seconds.to_string().parse().unwrap(),
    );
    headers.insert(
        "X-Backoff-Ms",
        _backoff.x_backoff_ms.to_string().parse().unwrap(),
    );
    headers
}

/// Build a minimal 429 body with advisory policy label.
/// Planning-only: returns a JSON object shape but with default values.
pub fn build_429_body(_policy: PolicyLabel) -> serde_json::Value {
    // TODO(impl): include policy_label and optional advisory fields
    let label = match _policy {
        PolicyLabel::Reject => "reject",
        PolicyLabel::DropLru => "drop-lru",
        PolicyLabel::ShedLowPriority => "shed-low-priority",
    };
    json!({
        "policy_label": label,
        "retriable": true,
        "retry_after_ms": 1000,
    })
}
