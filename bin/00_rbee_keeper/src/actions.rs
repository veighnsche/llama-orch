//! Action constants for rbee-keeper narration
//!
//! Centralized action identifiers used in narration events.

// Actor
pub const ACTOR_RBEE_KEEPER: &str = "🧑‍🌾 rbee-keeper";

// Queen actions
pub const ACTION_QUEEN_START: &str = "queen_start";
pub const ACTION_QUEEN_STOP: &str = "queen_stop";

// Hive actions
pub const ACTION_HIVE_START: &str = "hive_start";
pub const ACTION_HIVE_STOP: &str = "hive_stop";
pub const ACTION_HIVE_LIST: &str = "hive_list";
pub const ACTION_HIVE_GET: &str = "hive_get";
pub const ACTION_HIVE_CREATE: &str = "hive_create";
pub const ACTION_HIVE_UPDATE: &str = "hive_update";
pub const ACTION_HIVE_DELETE: &str = "hive_delete";

// Worker actions
pub const ACTION_WORKER_SPAWN: &str = "worker_spawn";
pub const ACTION_WORKER_LIST: &str = "worker_list";
pub const ACTION_WORKER_GET: &str = "worker_get";
pub const ACTION_WORKER_DELETE: &str = "worker_delete";

// Model actions
pub const ACTION_MODEL_DOWNLOAD: &str = "model_download";
pub const ACTION_MODEL_LIST: &str = "model_list";
pub const ACTION_MODEL_GET: &str = "model_get";
pub const ACTION_MODEL_DELETE: &str = "model_delete";

// Job actions
pub const ACTION_INFER: &str = "infer";
pub const ACTION_STREAM: &str = "stream_sse";

// Legacy actions (kept for compatibility)
pub const ACTION_ADD_HIVE: &str = "add_hive";
pub const ACTION_HEALTH_CHECK: &str = "health_check";
