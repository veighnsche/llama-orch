//! Narration setup for rbee-keeper
//!
//! TEAM-185: Renamed from actions.rs to operations.rs
//! TEAM-186: Removed OP_* constants - now using typed Operation enum from rbee-operations crate
//! TEAM-191: Added narrate! macro for ergonomic narration

// Re-export narration_macro from narration-core
pub use observability_narration_core::narration_macro;

// Actor
pub const ACTOR_RBEE_KEEPER: &str = "🧑‍🌾 rbee-keeper";

// Job lifecycle actions (for narration)
pub const ACTION_QUEEN_START: &str = "queen_start";
pub const ACTION_QUEEN_STOP: &str = "queen_stop";
pub const ACTION_QUEEN_STATUS: &str = "queen_status"; // TEAM-186: Added for status checks
pub const ACTION_JOB_SUBMIT: &str = "job_submit";
pub const ACTION_JOB_STREAM: &str = "job_stream";
pub const ACTION_JOB_COMPLETE: &str = "job_complete";

// ============================================================================
// TEAM-191: Narration Macro Setup
// ============================================================================

/// Create the narrate! macro with rbee-keeper's actor baked in.
///
/// This allows ergonomic narration throughout rbee-keeper without repeating the actor.
///
/// # Usage
/// ```rust,ignore
/// use crate::narration::narrate;
///
/// narrate!(ACTION_QUEEN_START, "queen-rbee")
///     .human("🚀 Starting queen-rbee")
///     .emit();
/// ```
///
/// # TEAM-191: Ultimate Ergonomics!
/// This pattern is inspired by `println!` - define once, use everywhere.
narration_macro!(ACTOR_RBEE_KEEPER);
