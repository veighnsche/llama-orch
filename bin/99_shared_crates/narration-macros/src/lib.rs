extern crate proc_macro;

mod actor_inference;
mod narrate;
mod template;
mod trace_fn;
mod with_actor; // TEAM-309: Actor injection macro

use proc_macro::TokenStream;

/// Attribute macro for automatic function tracing with entry/exit narration.
///
/// Automatically generates narration events at function entry and exit, with timing.
/// Actor is auto-inferred from module path.
///
/// # Example
/// ```ignore
/// #[trace_fn]
/// async fn dispatch_job(job_id: &str) -> Result<WorkerId> {
///     // Function body
/// }
/// ```
///
/// Generates:
/// - Entry trace with function name and arguments
/// - Exit trace with result and elapsed time
/// - Automatic error handling for Result types
#[proc_macro_attribute]
pub fn trace_fn(attr: TokenStream, item: TokenStream) -> TokenStream {
    trace_fn::trace_fn_impl(attr, item)
}

/// Attribute macro for narration with template interpolation.
///
/// Supports compile-time template expansion with automatic actor inference.
///
/// # Example
/// ```ignore
/// #[narrate(
///     action: "dispatch",
///     human: "Dispatched job {job_id} to worker {worker_id}",
///     cute: "Sent job {job_id} off to its new friend {worker_id}! 🎫"
/// )]
/// fn dispatch_job(job_id: &str, worker_id: &str) -> Result<()> {
///     // Function body
/// }
/// ```
#[proc_macro_attribute]
pub fn narrate(attr: TokenStream, item: TokenStream) -> TokenStream {
    narrate::narrate_impl(attr, item)
}

/// Attribute macro to set narration actor for entire function (TEAM-309).
///
/// All n!() calls inside automatically use this actor.
///
/// # Example
/// ```ignore
/// #[with_actor("auto-update")]
/// fn needs_rebuild() -> Result<bool> {
///     n!("check", "Checking..."); // Uses actor="auto-update"
///     Ok(true)
/// }
/// ```
#[proc_macro_attribute]
pub fn with_actor(attr: TokenStream, item: TokenStream) -> TokenStream {
    with_actor::with_actor_impl(attr, item)
}
