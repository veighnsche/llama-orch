// TEAM-297: Phase 0 - Macro implementation
//! Internal implementation for the n!() macro

use crate::{mode::get_narration_mode, NarrationFields, NarrationMode, NarrationLevel};

/// Emit narration from macro (internal use)
///
/// TEAM-297: This is called by n!() macro, not by users directly.
/// Selects the appropriate message based on current narration mode.
///
/// # Arguments
/// - `action`: Action name (e.g., "worker_spawn")
/// - `human`: Human-readable message (always required)
/// - `cute`: Optional cute message (if None, falls back to human)
/// - `story`: Optional story message (if None, falls back to human)
///
/// TEAM-309: Added optional actor parameter for crates that can't use async context
#[doc(hidden)]
pub fn macro_emit(
    action: &'static str,
    human: &str,
    cute: Option<&str>,
    story: Option<&str>,
) {
    macro_emit_with_actor(action, human, cute, story, None)
}

/// TEAM-309: Emit narration with auto-detected crate name as actor
#[doc(hidden)]
pub fn macro_emit_auto(
    action: &'static str,
    human: &str,
    cute: Option<&str>,
    story: Option<&str>,
    crate_name: &'static str,
) {
    macro_emit_with_actor_and_level(action, human, cute, story, Some(crate_name), NarrationLevel::Info)
}

/// TEAM-311: Emit narration with auto-detected crate name and explicit level
#[doc(hidden)]
pub fn macro_emit_auto_with_level(
    action: &'static str,
    human: &str,
    cute: Option<&str>,
    story: Option<&str>,
    crate_name: &'static str,
    level: NarrationLevel,
) {
    macro_emit_with_actor_and_level(action, human, cute, story, Some(crate_name), level)
}

/// TEAM-309: Emit narration with explicit actor (for sync code that can't use context)
#[doc(hidden)]
pub fn macro_emit_with_actor(
    action: &'static str,
    human: &str,
    cute: Option<&str>,
    story: Option<&str>,
    explicit_actor: Option<&'static str>,
) {
    macro_emit_with_actor_and_level(action, human, cute, story, explicit_actor, NarrationLevel::Info)
}

/// TEAM-311: Emit narration with explicit actor and level
#[doc(hidden)]
pub fn macro_emit_with_actor_and_level(
    action: &'static str,
    human: &str,
    cute: Option<&str>,
    story: Option<&str>,
    explicit_actor: Option<&'static str>,
    level: NarrationLevel,
) {
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
    let actor = explicit_actor
        .or_else(|| ctx.as_ref().and_then(|c| c.actor))
        .unwrap_or("unknown");
    
    // TEAM-311: Function name from thread-local (set by #[narrate_fn])
    let fn_name = crate::thread_actor::get_target();
    
    // TEAM-309: Target defaults to action
    let target = action.to_string();
    
    // TEAM-297: Build fields with selected message
    // TEAM-311: Include level and fn_name
    let fields = NarrationFields {
        actor,
        action,
        target,
        human: selected_message.to_string(),
        level, // TEAM-311: Include level
        fn_name, // TEAM-311: Function name from #[narrate_fn]
        cute: cute.map(|s| s.to_string()),
        story: story.map(|s| s.to_string()),
        job_id,
        correlation_id,
        ..Default::default()
    };
    
    // TEAM-311: CRITICAL FIX - Use narrate_at_level() not narrate()
    // narrate() always uses Info level, ignoring fields.level!
    crate::narrate_at_level(fields, level);
}
