// TEAM-300: Modular reorganization - Action constants
//! Action taxonomy for narration system
//!
//! Actions represent WHAT operation was performed in the system.

// ============================================================================
// Admission queue operations
// ============================================================================

/// Admission queue operations
pub const ACTION_ADMISSION: &str = "admission";
pub const ACTION_ENQUEUE: &str = "enqueue";
pub const ACTION_DISPATCH: &str = "dispatch";

// ============================================================================
// Worker lifecycle
// ============================================================================

pub const ACTION_SPAWN: &str = "spawn";
pub const ACTION_READY_CALLBACK: &str = "ready_callback";
pub const ACTION_HEARTBEAT_SEND: &str = "heartbeat_send";
pub const ACTION_HEARTBEAT_RECEIVE: &str = "heartbeat_receive";
pub const ACTION_SHUTDOWN: &str = "shutdown";

// ============================================================================
// Inference operations
// ============================================================================

pub const ACTION_INFERENCE_START: &str = "inference_start";
pub const ACTION_INFERENCE_COMPLETE: &str = "inference_complete";
pub const ACTION_INFERENCE_ERROR: &str = "inference_error";
pub const ACTION_CANCEL: &str = "cancel";

// ============================================================================
// VRAM operations
// ============================================================================

pub const ACTION_VRAM_ALLOCATE: &str = "vram_allocate";
pub const ACTION_VRAM_DEALLOCATE: &str = "vram_deallocate";
pub const ACTION_SEAL: &str = "seal";
pub const ACTION_VERIFY: &str = "verify";

// ============================================================================
// Pool management
// ============================================================================

pub const ACTION_REGISTER: &str = "register";
pub const ACTION_DEREGISTER: &str = "deregister";
pub const ACTION_PROVISION: &str = "provision";

// ============================================================================
// TEAM-191: Job routing actions (used by queen-rbee)
// ============================================================================

/// Route job to appropriate handler
pub const ACTION_ROUTE_JOB: &str = "route_job";
/// Parse operation payload
pub const ACTION_PARSE_OPERATION: &str = "parse_operation";
/// Create new job
pub const ACTION_JOB_CREATE: &str = "job_create";

// ============================================================================
// TEAM-191: Hive management actions (used by queen-rbee)
// ============================================================================

/// Install hive
pub const ACTION_HIVE_INSTALL: &str = "hive_install";
/// Uninstall hive
pub const ACTION_HIVE_UNINSTALL: &str = "hive_uninstall";
/// Start hive daemon
pub const ACTION_HIVE_START: &str = "hive_start";
/// Stop hive daemon
pub const ACTION_HIVE_STOP: &str = "hive_stop";
/// Check hive status
pub const ACTION_HIVE_STATUS: &str = "hive_status";
/// List all hives
pub const ACTION_HIVE_LIST: &str = "hive_list";

// ============================================================================
// TEAM-191: System actions (used by queen-rbee)
// ============================================================================

/// Get system status
pub const ACTION_STATUS: &str = "status";
/// Start service
pub const ACTION_START: &str = "start";
/// Listen for connections
pub const ACTION_LISTEN: &str = "listen";
/// Service ready
pub const ACTION_READY: &str = "ready";
/// Error occurred
pub const ACTION_ERROR: &str = "error";
