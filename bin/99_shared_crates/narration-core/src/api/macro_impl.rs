// TEAM-297: Phase 0 - Macro implementation
//! Internal implementation for the n!() macro

use crate::{mode::get_narration_mode, NarrationFields, NarrationLevel, NarrationMode};

/// Clean up closure names by removing `::{{closure}}` suffixes
///
/// TEAM-312: Rust's function_name!() includes `::{{closure}}` for anonymous closures,
/// which is not user-friendly. This removes all such suffixes.
///
/// # Examples
/// ```
/// # use observability_narration_core::api::macro_impl::clean_closure_name;
/// assert_eq!(clean_closure_name("my_fn::{{closure}}"), "my_fn");
/// assert_eq!(clean_closure_name("my_fn::{{closure}}::{{closure}}"), "my_fn");
/// assert_eq!(clean_closure_name("my_fn"), "my_fn");
/// ```
fn clean_closure_name(name: &str) -> String {
    // Remove all ::{{closure}} suffixes (can appear multiple times for nested closures)
    let mut cleaned = name.to_string();
    while cleaned.ends_with("::{{closure}}") {
        cleaned.truncate(cleaned.len() - "::{{closure}}".len());
    }
    cleaned
}

// ============================================================================
// TEAM-312: DELETED BACKWARDS COMPATIBILITY WRAPPERS (RULE ZERO)
// ============================================================================
//
// This file previously had 6 wrapper functions all calling the same implementation:
//   1. macro_emit() → macro_emit_with_actor()
//   2. macro_emit_auto() → macro_emit_with_actor_and_level()
//   3. macro_emit_auto_with_level() → macro_emit_with_actor_and_level()
//   4. macro_emit_with_actor() → macro_emit_with_actor_and_level()
//   5. macro_emit_with_actor_and_level() → macro_emit_with_actor_fn_and_level()
//   6. macro_emit_auto_with_fn() → macro_emit_with_actor_fn_and_level()
//
// ❌ PROBLEM: This is a backwards compatibility pyramid creating permanent debt.
//    Every bug fix or feature needs to be tested through 6 different entry points.
//
// ✅ SOLUTION: Keep only the main implementation. The n!() macro calls it directly.
//    Breaking change: 0 minutes (nothing external uses these #[doc(hidden)] functions)
//    Maintenance saved: 6 functions → 1 function (83% reduction)
//
// See: .windsurf/rules/engineering-rules.md (RULE ZERO)
// ============================================================================

/// Emit narration from n!() macro (internal implementation)
///
/// TEAM-297: This is called by n!() macro, not by users directly.
/// TEAM-312: Renamed from macro_emit_with_actor_fn_and_level, deleted 6 wrapper functions
///
/// # Arguments
/// - `action`: Action name (e.g., "worker_spawn")
/// - `human`: Human-readable message (always required)
/// - `cute`: Optional cute message (if None, falls back to human)
/// - `story`: Optional story message (if None, falls back to human)
/// - `crate_name`: Crate name from env!("CARGO_CRATE_NAME")
/// - `fn_name`: Function name from stdext::function_name!()
#[doc(hidden)]
pub fn macro_emit(
    action: &'static str,
    human: &str,
    cute: Option<&str>,
    story: Option<&str>,
    crate_name: &'static str,
    fn_name: &'static str,
) {
    // TEAM-312: Default to Info level (was a parameter in old function)
    let level = NarrationLevel::Info;
    let explicit_actor = Some(crate_name);
    let explicit_fn_name = Some(fn_name);
    // TEAM-297: Get narration mode from global config
    let mode = get_narration_mode();

    // TEAM-297: Select which message to display based on mode
    let selected_message = match mode {
        NarrationMode::Human => human,
        NarrationMode::Cute => cute.unwrap_or(human),
        NarrationMode::Story => story.unwrap_or(human),
    };

    // TEAM-297: Get context if available (for job_id and correlation_id)
    // TEAM-300: Phase 2 - Now also gets actor from context
    let ctx = crate::context::get_context();
    let job_id = ctx.as_ref().and_then(|c| c.job_id.clone());
    let correlation_id = ctx.as_ref().and_then(|c| c.correlation_id.clone());

    // TEAM-309: Actor priority: explicit_actor > context > "unknown"
    let actor = explicit_actor.or_else(|| ctx.as_ref().and_then(|c| c.actor)).unwrap_or("unknown");

    // TEAM-312: Function name priority: explicit_fn_name > thread-local > None
    // Thread-local is set by #[narrate_fn], explicit_fn_name is from function_name!()
    let fn_name = explicit_fn_name
        .map(|s| clean_closure_name(s))
        .or_else(|| crate::thread_actor::get_target().map(|s| clean_closure_name(&s)));

    // TEAM-309: Target defaults to action
    let target = action.to_string();

    // TEAM-297: Build fields with selected message
    // TEAM-311: Include level and fn_name
    let fields = NarrationFields {
        actor,
        action,
        target,
        human: selected_message.to_string(),
        level,   // TEAM-311: Include level
        fn_name, // TEAM-312: Function name from function_name!() or #[narrate_fn]
        cute: cute.map(|s| s.to_string()),
        story: story.map(|s| s.to_string()),
        job_id,
        correlation_id,
        ..Default::default()
    };

    // TEAM-311: CRITICAL FIX - Use narrate_at_level() not narrate()
    // narrate() always uses Info level, ignoring fields.level!
    crate::narrate(fields, level);
}
