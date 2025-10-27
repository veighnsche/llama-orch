// TEAM-309: Proc macros for narration system
// TEAM-328: Added #[with_job_id] macro to eliminate job_id context boilerplate
//! Procedural macros for the narration system

use proc_macro::TokenStream;

mod with_job_id;

/// Attribute macro to automatically handle job_id context wrapping (TEAM-328).
///
/// Eliminates the repetitive pattern of extracting job_id from config,
/// creating NarrationContext, and wrapping with with_narration_context.
///
/// # Usage
/// ```ignore
/// use observability_narration_macros::with_job_id;
/// use observability_narration_core::n;
///
/// #[with_job_id(config_param = "rebuild_config")]
/// pub async fn rebuild_with_hot_reload(
///     rebuild_config: RebuildConfig,
///     daemon_config: HttpDaemonConfig,
/// ) -> Result<bool> {
///     // Your implementation here
///     // n!() calls automatically use job_id from rebuild_config
///     n!("start", "Starting rebuild...");
///     Ok(true)
/// }
/// ```
///
/// # Auto-detection
/// If you don't specify `config_param`, the macro will automatically find
/// the first parameter with "config" in its name:
/// ```ignore
/// #[with_job_id]  // Auto-detects "rebuild_config"
/// pub async fn rebuild(rebuild_config: RebuildConfig) -> Result<()> {
///     n!("action", "Doing thing");
///     Ok(())
/// }
/// ```
///
/// # Requirements
/// - Function must be `async`
/// - Config parameter must have a `job_id: Option<String>` field
/// - Must import `observability_narration_core::{with_narration_context, NarrationContext}`
#[proc_macro_attribute]
pub fn with_job_id(attr: TokenStream, item: TokenStream) -> TokenStream {
    with_job_id::with_job_id_impl(attr, item)
}
