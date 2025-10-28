//! Task-local narration context
//!
//! Eliminates the need to pass job_id and correlation_id to every narration.
//! Set once at task start, automatically included in all narrations.

use std::cell::RefCell;

tokio::task_local! {
    static NARRATION_CONTEXT: RefCell<NarrationContext>;
}

#[derive(Debug, Clone, Default)]
pub struct NarrationContext {
    pub job_id: Option<String>,
    pub correlation_id: Option<String>,
    // TEAM-300: Phase 2 - Add actor support
    // Actor identifies which component is narrating (e.g., "qn-router", "hive-mgr")
    pub actor: Option<&'static str>,
}

impl NarrationContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.job_id = Some(job_id.into());
        self
    }

    pub fn with_correlation_id(mut self, correlation_id: impl Into<String>) -> Self {
        self.correlation_id = Some(correlation_id.into());
        self
    }

    // TEAM-300: Phase 2 - Add actor builder
    /// ⚠️ **DEPRECATED** - Actor is now auto-detected from crate name via env!("CARGO_CRATE_NAME").
    ///
    /// Use `n!()` macro directly - it will automatically use the crate name as actor.
    #[deprecated(
        since = "0.5.0",
        note = "Actor is now auto-detected from crate name. Just use n!() macro directly."
    )]
    pub fn with_actor(mut self, actor: &'static str) -> Self {
        self.actor = Some(actor);
        self
    }
}

/// Run a task with narration context
///
/// ⚠️ **DEPRECATED for actor setting** - Actor is now auto-detected from crate name.
/// Still useful for job_id and correlation_id.
///
/// # Example (NEW - actor auto-detected)
///
/// ```rust,ignore
/// use observability_narration_core::n;
///
/// async fn my_handler() -> Result<()> {
///     n!("start", "Starting job");  // actor = crate name (auto)
///     // ... do work ...
///     n!("complete", "Job complete");
///     Ok(())
/// }
/// ```
///
/// # Example (OLD - still works for job_id/correlation_id)
///
/// ```rust,ignore
/// use observability_narration_core::{n, NarrationContext, with_narration_context};
///
/// async fn my_handler(job_id: String) -> Result<()> {
///     let ctx = NarrationContext::new()
///         .with_job_id(&job_id);  // ← Still useful for job_id
///     
///     with_narration_context(ctx, async {
///         n!("start", "Starting job");  // actor auto-detected, job_id from context
///         n!("complete", "Job complete");
///     }).await
/// }
/// ```
pub async fn with_narration_context<F>(ctx: NarrationContext, f: F) -> F::Output
where
    F: std::future::Future,
{
    NARRATION_CONTEXT.scope(RefCell::new(ctx), f).await
}

/// Get the current narration context (internal use)
pub(crate) fn get_context() -> Option<NarrationContext> {
    NARRATION_CONTEXT.try_with(|ctx| ctx.borrow().clone()).ok()
}
