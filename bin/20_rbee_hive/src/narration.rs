//! Hive narration configuration
//!
//! TEAM-202: Narration for rbee-hive using job-scoped SSE
//! TEAM-218: Investigated Oct 22, 2025 - Narration constants documented
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
// TEAM-261: Removed ACTION_HEARTBEAT (workers send heartbeats to queen directly)
pub const ACTION_WORKER_SPAWN: &str = "worker_spawn";
pub const ACTION_WORKER_STOP: &str = "worker_stop";
pub const ACTION_LISTEN: &str = "listen";
pub const ACTION_READY: &str = "ready";

// TEAM-206: Capabilities endpoint actions
pub const ACTION_CAPS_REQUEST: &str = "caps_request";
pub const ACTION_CAPS_GPU_CHECK: &str = "caps_gpu_check";
pub const ACTION_CAPS_GPU_FOUND: &str = "caps_gpu_found";
pub const ACTION_CAPS_CPU_ADD: &str = "caps_cpu_add";
pub const ACTION_CAPS_RESPONSE: &str = "caps_response";
