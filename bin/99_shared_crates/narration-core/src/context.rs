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
}

/// Run a task with narration context
///
/// Sets job_id and correlation_id once, then all narrations inside
/// automatically include them.
///
/// # Example
/// ```rust,ignore
/// use observability_narration_core::{with_narration_context, NarrationContext, NarrationFactory};
///
/// const NARRATE: NarrationFactory = NarrationFactory::new("qn-router");
///
/// let ctx = NarrationContext::new()
///     .with_job_id("job-abc123")
///     .with_correlation_id("corr-xyz789");
///
/// with_narration_context(ctx, async move {
///     // No need to add job_id or correlation_id!
///     NARRATE.action("hive_start").human("Starting hive").emit();
///     NARRATE.action("hive_check").human("Checking status").emit();
///     // Both automatically include job_id and correlation_id
/// }).await;
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
