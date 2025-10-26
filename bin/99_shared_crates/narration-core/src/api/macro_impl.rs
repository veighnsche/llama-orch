// TEAM-297: Phase 0 - Macro implementation
//! Internal implementation for the n!() macro

use crate::{mode::get_narration_mode, NarrationFields, NarrationMode};

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
#[doc(hidden)]
pub fn macro_emit(
    action: &'static str,
    human: &str,
    cute: Option<&str>,
    story: Option<&str>,
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
    
    // TEAM-300: Phase 2 - Actor from context, defaults to "unknown"
    let actor = ctx.as_ref().and_then(|c| c.actor).unwrap_or("unknown");
    
    // TEAM-297: Build fields with selected message
    let fields = NarrationFields {
        actor,
        action,
        target: action.to_string(), // Use action as default target
        human: selected_message.to_string(),
        cute: cute.map(|s| s.to_string()),
        story: story.map(|s| s.to_string()),
        job_id,
        correlation_id,
        ..Default::default()
    };
    
    // TEAM-297: Emit using existing narrate() function
    crate::narrate(fields);
}
