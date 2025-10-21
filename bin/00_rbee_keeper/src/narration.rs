//! Operation constants for rbee-keeper
//!
//! TEAM-185: Renamed from actions.rs to operations.rs
//! TEAM-186: Removed OP_* constants - now using typed Operation enum from rbee-operations crate

// Actor
pub const ACTOR_RBEE_KEEPER: &str = "🧑‍🌾 rbee-keeper";

// Job lifecycle actions (for narration)
pub const ACTION_QUEEN_START: &str = "queen_start";
pub const ACTION_QUEEN_STOP: &str = "queen_stop";
pub const ACTION_QUEEN_STATUS: &str = "queen_status"; // TEAM-186: Added for status checks
pub const ACTION_JOB_SUBMIT: &str = "job_submit";
pub const ACTION_JOB_STREAM: &str = "job_stream";
pub const ACTION_JOB_COMPLETE: &str = "job_complete";
