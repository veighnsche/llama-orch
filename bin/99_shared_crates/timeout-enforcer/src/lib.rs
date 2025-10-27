//! Timeout Enforcer - Hard timeout enforcement with visual countdown
//!
//! Created by: TEAM-163
//! Updated by: TEAM-197 (narration-core v0.5.0 migration)
//! Updated by: TEAM-312 (narration-core v0.7.0 migration - n!() macro)
//! Updated by: TEAM-330 (Universal context propagation - works everywhere)
//!
//! # Purpose
//! Prevents hanging operations by enforcing hard timeouts with visual feedback.
//! Every operation that could hang MUST use this crate.
//!
//! # Features
//! - Hard timeout enforcement (operation WILL fail after timeout)
//! - Visual countdown in terminal (shows remaining time)
//! - Clear error messages when timeout occurs
//! - Zero tolerance for hanging operations
//! - **Universal**: Works in client, server, and WASM contexts
//! - **Context-aware**: Automatically includes job_id from narration context
//!
//! # Usage
//!
//! ## Simple (Client-side)
//! ```no_run
//! use timeout_enforcer::TimeoutEnforcer;
//! use std::time::Duration;
//!
//! async fn my_operation() -> anyhow::Result<String> {
//!     // Your operation here
//!     Ok("done".to_string())
//! }
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let result = TimeoutEnforcer::new(Duration::from_secs(30))
//!         .with_label("Starting queen-rbee")
//!         .enforce(my_operation())
//!         .await?;
//!     Ok(())
//! }
//! ```
//!
//! ## With Context (Server-side - SSE routing)
//! ```no_run
//! use timeout_enforcer::TimeoutEnforcer;
//! use observability_narration_core::{NarrationContext, with_narration_context};
//! use std::time::Duration;
//!
//! async fn my_operation() -> anyhow::Result<String> {
//!     Ok("done".to_string())
//! }
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let ctx = NarrationContext::new()
//!         .with_job_id("job-123");
//!     
//!     with_narration_context(ctx, async {
//!         // Timeout narration automatically includes job_id for SSE routing!
//!         let result = TimeoutEnforcer::new(Duration::from_secs(30))
//!             .with_label("Starting queen-rbee")
//!             .enforce(my_operation())
//!             .await?;
//!         Ok(())
//!     }).await
//! }
//! ```

// TEAM-330: Modular structure for better readability
mod enforcer;
mod enforcement;

#[cfg(test)]
mod tests;

// Re-export main struct
pub use enforcer::TimeoutEnforcer;

// TEAM-330: Re-export attribute macro for ergonomic usage
/// Attribute macro to enforce timeouts on async functions
///
/// # Example
/// ```rust,ignore
/// use timeout_enforcer::with_timeout;
///
/// #[with_timeout(secs = 30, label = "Slow operation")]
/// async fn slow_operation() -> anyhow::Result<String> {
///     // ... operation ...
///     Ok("done".into())
/// }
/// ```
pub use timeout_enforcer_macros::with_timeout;

// TEAM-330: Core implementation moved to separate modules
// See enforcer.rs for struct definition
// See enforcement.rs for timeout logic
// See tests.rs for unit tests
