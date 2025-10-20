//! Operation constants for rbee-keeper
//!
//! Centralized operation identifiers used in job payloads and narration.

// Actor
pub const ACTOR_RBEE_KEEPER: &str = "🧑‍🌾 rbee-keeper";

// Hive operations
pub const OP_HIVE_START: &str = "hive_start";
pub const OP_HIVE_STOP: &str = "hive_stop";
pub const OP_HIVE_LIST: &str = "hive_list";
pub const OP_HIVE_GET: &str = "hive_get";
pub const OP_HIVE_CREATE: &str = "hive_create";
pub const OP_HIVE_UPDATE: &str = "hive_update";
pub const OP_HIVE_DELETE: &str = "hive_delete";

// Worker operations
pub const OP_WORKER_SPAWN: &str = "worker_spawn";
pub const OP_WORKER_LIST: &str = "worker_list";
pub const OP_WORKER_GET: &str = "worker_get";
pub const OP_WORKER_DELETE: &str = "worker_delete";

// Model operations
pub const OP_MODEL_DOWNLOAD: &str = "model_download";
pub const OP_MODEL_LIST: &str = "model_list";
pub const OP_MODEL_GET: &str = "model_get";
pub const OP_MODEL_DELETE: &str = "model_delete";

// Inference operation
pub const OP_INFER: &str = "infer";

// Job lifecycle actions (for narration)
pub const ACTION_QUEEN_START: &str = "queen_start";
pub const ACTION_QUEEN_STOP: &str = "queen_stop";
pub const ACTION_JOB_SUBMIT: &str = "job_submit";
pub const ACTION_JOB_STREAM: &str = "job_stream";
pub const ACTION_JOB_COMPLETE: &str = "job_complete";
