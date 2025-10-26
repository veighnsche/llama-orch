// TEAM-309: Proc macros for narration system
//! Procedural macros for the narration system
//!
//! **Current:** Only `#[narrate_fn]` is supported - adds function name as narration target.
//! **Removed:** `#[narrate]` and `#[trace_fn]` - use `n!()` macro instead.

use proc_macro::TokenStream;

mod with_actor;

/// Attribute macro to set function name as narration target (TEAM-309).
///
/// **OPTIONAL** - Only use this if you want function-scoped narration.
/// Most functions don't need this - just use `n!()` directly.
///
/// All n!() calls inside automatically use function name as target.
/// Actor is auto-detected from crate name.
///
/// # When to use
/// - Use `#[narrate_fn]` for important functions where you want to see the function name in logs
/// - Skip it for simple helper functions - just use `n!()` directly
///
/// # Example
/// ```ignore
/// #[narrate_fn]  // ← OPTIONAL - only if you want [crate/function] format
/// fn rebuild() -> Result<()> {
///     n!("start", "Starting...");   // Output: [auto-update/rebuild] start : Starting...
///     n!("success", "Done!");        // Output: [auto-update/rebuild] success : Done!
///     Ok(())
/// }
///
/// // Without #[narrate_fn]:
/// fn simple_helper() -> Result<()> {
///     n!("action", "Doing thing");  // Output: [auto-update] action : Doing thing
///     Ok(())
/// }
/// ```
#[proc_macro_attribute]
pub fn narrate_fn(attr: TokenStream, item: TokenStream) -> TokenStream {
    with_actor::narrate_fn_impl(attr, item)
}
