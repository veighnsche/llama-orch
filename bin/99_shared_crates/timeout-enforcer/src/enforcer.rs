//! Core TimeoutEnforcer struct and builder methods
//!
//! Created by: TEAM-163
//! Updated by: TEAM-330 (Universal context propagation)

use std::time::Duration;

/// Timeout enforcer with visual countdown feedback
///
/// TEAM-330: Simplified API - context propagates automatically!
///
/// # Example (Client)
/// ```no_run
/// use timeout_enforcer::TimeoutEnforcer;
/// use std::time::Duration;
///
/// async fn slow_operation() -> anyhow::Result<()> {
///     tokio::time::sleep(Duration::from_secs(5)).await;
///     Ok(())
/// }
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     TimeoutEnforcer::new(Duration::from_secs(10))
///         .with_label("Slow operation")
///         .enforce(slow_operation())
///         .await?;
///     Ok(())
/// }
/// ```
///
/// # Example (Server with SSE)
/// ```no_run
/// use timeout_enforcer::TimeoutEnforcer;
/// use observability_narration_core::{NarrationContext, with_narration_context};
/// use std::time::Duration;
///
/// async fn slow_operation() -> anyhow::Result<()> {
///     tokio::time::sleep(Duration::from_secs(5)).await;
///     Ok(())
/// }
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let ctx = NarrationContext::new().with_job_id("job-123");
///     
///     with_narration_context(ctx, async {
///         // job_id automatically included in timeout narration!
///         TimeoutEnforcer::new(Duration::from_secs(10))
///             .with_label("Slow operation")
///             .enforce(slow_operation())
///             .await?;
///         Ok(())
///     }).await
/// }
/// ```
pub struct TimeoutEnforcer {
    pub(crate) duration: Duration,
    pub(crate) label: Option<String>,
    pub(crate) show_countdown: bool,
    // TEAM-330: Removed job_id field - use NarrationContext instead!
    // Context propagates automatically via tokio::task_local
}

impl TimeoutEnforcer {
    /// Create a new timeout enforcer
    ///
    /// TEAM-330: Context (job_id, correlation_id) propagates automatically!
    ///
    /// # Arguments
    /// * `duration` - Maximum time to wait before failing
    ///
    /// # Example
    /// ```
    /// use timeout_enforcer::TimeoutEnforcer;
    /// use std::time::Duration;
    ///
    /// let enforcer = TimeoutEnforcer::new(Duration::from_secs(30));
    /// ```
    pub fn new(duration: Duration) -> Self {
        // TEAM-197: Countdown disabled by default to avoid interfering with narration output
        // TEAM-330: Removed job_id field - context propagates via NarrationContext
        Self { duration, label: None, show_countdown: false }
    }

    /// Set a label for the operation (shown in countdown)
    ///
    /// # Example
    /// ```
    /// use timeout_enforcer::TimeoutEnforcer;
    /// use std::time::Duration;
    ///
    /// let enforcer = TimeoutEnforcer::new(Duration::from_secs(30))
    ///     .with_label("Starting queen-rbee");
    /// ```
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    // TEAM-330: DELETED with_job_id() method (RULE ZERO)
    //
    // ❌ REMOVED: pub fn with_job_id(self, job_id: impl Into<String>) -> Self
    //
    // WHY: RULE ZERO - Breaking changes > backwards compatibility
    //
    // Pre-1.0 software is ALLOWED to break. The compiler will catch breaking changes.
    // Entropy from "backwards compatibility" functions is PERMANENT TECHNICAL DEBT.
    //
    // OLD API (deleted):
    //   TimeoutEnforcer::new(timeout).with_job_id(&job_id).enforce(future).await
    //
    // NEW API (use this):
    //   let ctx = NarrationContext::new().with_job_id(&job_id);
    //   with_narration_context(ctx, async {
    //       TimeoutEnforcer::new(timeout).enforce(future).await
    //   }).await
    //
    // If you get a compilation error about missing with_job_id():
    //   1. Wrap your code in with_narration_context()
    //   2. The compiler found the call site for you (30 seconds)
    //   3. Fix it (2 minutes)
    //   4. Done - no permanent entropy!
    //
    // See: .windsurf/rules/engineering-rules.md (RULE ZERO)

    /// Enable countdown display (visual feedback)
    ///
    /// Note: Countdown is disabled by default to avoid interfering with narration output.
    /// Only enable if you need the visual countdown feedback.
    ///
    /// # Example
    /// ```
    /// use timeout_enforcer::TimeoutEnforcer;
    /// use std::time::Duration;
    ///
    /// let enforcer = TimeoutEnforcer::new(Duration::from_secs(30))
    ///     .with_countdown();
    /// ```
    pub fn with_countdown(mut self) -> Self {
        self.show_countdown = true;
        self
    }

    /// Disable countdown display (silent mode)
    ///
    /// # Example
    /// ```
    /// use timeout_enforcer::TimeoutEnforcer;
    /// use std::time::Duration;
    ///
    /// let enforcer = TimeoutEnforcer::new(Duration::from_secs(30))
    ///     .silent();
    /// ```
    pub fn silent(mut self) -> Self {
        self.show_countdown = false;
        self
    }
}
