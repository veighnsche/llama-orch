//! Hive narration configuration
//!
//! TEAM-202: Narration for rbee-hive using job-scoped SSE
//! 
//! This module provides narration constants for the hive daemon.
//! Narration flows through job-scoped SSE channels (from TEAM-200)
//! and uses centralized formatting (from TEAM-201).

use observability_narration_core::NarrationFactory;

// TEAM-202: Narration factory for hive
// Use "hive" as actor to match other components (queen, keeper, worker)
pub const NARRATE: NarrationFactory = NarrationFactory::new("hive");

// Hive-specific action constants
pub const ACTION_STARTUP: &str = "startup";
pub const ACTION_HEARTBEAT: &str = "heartbeat";
pub const ACTION_WORKER_SPAWN: &str = "worker_spawn";
pub const ACTION_WORKER_STOP: &str = "worker_stop";
pub const ACTION_LISTEN: &str = "listen";
pub const ACTION_READY: &str = "ready";
