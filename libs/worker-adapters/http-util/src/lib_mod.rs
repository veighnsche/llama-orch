//! worker-adapters/http-util — modular root

pub mod client;
pub mod auth;
pub mod redact;
pub mod retry;
pub mod streaming;
pub mod error;

// Re-exports for stable public API at crate root
pub use crate::auth::{bearer_header_from_env, with_bearer, with_bearer_if_configured};
pub use crate::client::{client, default_config, h2_preference, make_client, HttpClientConfig};
pub use crate::error::{is_non_retriable_status, is_retriable_status, parse_retry_after};
pub use crate::redact::redact_secrets;
pub use crate::retry::{get_and_clear_retry_timeline, with_retries, RetryError, RetryPolicy};
pub use crate::streaming::{stream_decode, StreamEvent};
