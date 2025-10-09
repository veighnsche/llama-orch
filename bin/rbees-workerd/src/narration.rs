//! Narration constants for rbees-workerd
//!
//! Defines all actor and action constants for triple-narration observability.
//! Following the Narration Core Team's editorial standards.
//!
//! Created by: Narration Core Team 🎀

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ACTORS — Who's doing the work
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Main worker daemon
pub const ACTOR_RBEES_WORKERD: &str = "rbees-workerd";

/// Inference backend (Candle)
pub const ACTOR_CANDLE_BACKEND: &str = "candle-backend";

/// HTTP server
pub const ACTOR_HTTP_SERVER: &str = "http-server";

/// Device initialization
pub const ACTOR_DEVICE_MANAGER: &str = "device-manager";

/// Model loading
pub const ACTOR_MODEL_LOADER: &str = "model-loader";

/// Tokenization
pub const ACTOR_TOKENIZER: &str = "tokenizer";

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ACTIONS — What's happening
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Worker starting
pub const ACTION_STARTUP: &str = "startup";

/// Loading model
pub const ACTION_MODEL_LOAD: &str = "model_load";

/// Initializing device
pub const ACTION_DEVICE_INIT: &str = "device_init";

/// GPU warmup
pub const ACTION_WARMUP: &str = "warmup";

/// HTTP server starting
pub const ACTION_SERVER_START: &str = "server_start";

/// Binding to address
pub const ACTION_SERVER_BIND: &str = "server_bind";

/// Server shutting down
pub const ACTION_SERVER_SHUTDOWN: &str = "server_shutdown";

/// Health endpoint called
pub const ACTION_HEALTH_CHECK: &str = "health_check";

/// Execute endpoint called
pub const ACTION_EXECUTE_REQUEST: &str = "execute_request";

/// Inference starting
pub const ACTION_INFERENCE_START: &str = "inference_start";

/// Inference completed
pub const ACTION_INFERENCE_COMPLETE: &str = "inference_complete";

/// Token generated
pub const ACTION_TOKEN_GENERATE: &str = "token_generate";

/// Cache reset
pub const ACTION_CACHE_RESET: &str = "cache_reset";

/// Pool manager callback
pub const ACTION_CALLBACK_READY: &str = "callback_ready";

/// Error occurred
pub const ACTION_ERROR: &str = "error";

/// Tokenization
pub const ACTION_TOKENIZE: &str = "tokenize";
