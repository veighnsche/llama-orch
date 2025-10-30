// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Dual-output narration (stdout + SSE)

//! Narration constants and dual-output wrapper for llm-worker-rbee
//!
//! Defines all actor and action constants for triple-narration observability.
//! Following the Narration Core Team's editorial standards.
//!
//! TEAM-039: Enhanced with dual-output narration (stdout + SSE)
//!
//! Created by: Narration Core Team 🎀
//! Modified by: TEAM-039 (added SSE channel support)

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ACTORS — Who's doing the work
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Main worker daemon
// TEAM-155: Added emoji prefix for visual identification
pub const ACTOR_LLM_WORKER_RBEE: &str = "🐝 llm-worker-rbee";

/// Inference backend (Candle)
pub const ACTOR_CANDLE_BACKEND: &str = "🐝 candle-backend";

/// HTTP server
pub const ACTOR_HTTP_SERVER: &str = "🐝 http-server";

/// Device initialization
pub const ACTOR_DEVICE_MANAGER: &str = "🐝 device-manager";

/// Model loading
pub const ACTOR_MODEL_LOADER: &str = "🐝 model-loader";

/// Tokenization
pub const ACTOR_TOKENIZER: &str = "🐝 tokenizer";

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

/// Error occurred
pub const ACTION_ERROR: &str = "error";

/// Tokenization
pub const ACTION_TOKENIZE: &str = "tokenize";

// TEAM-088: Additional actions for comprehensive debugging
/// Model load failed
pub const ACTION_MODEL_LOAD_FAILED: &str = "model_load_failed";

/// GGUF file operations
pub const ACTION_GGUF_LOAD_START: &str = "gguf_load_start";
pub const ACTION_GGUF_OPEN_FAILED: &str = "gguf_open_failed";
pub const ACTION_GGUF_FILE_OPENED: &str = "gguf_file_opened";
pub const ACTION_GGUF_PARSE_FAILED: &str = "gguf_parse_failed";
pub const ACTION_GGUF_INSPECT_METADATA: &str = "gguf_inspect_metadata";
pub const ACTION_GGUF_METADATA_KEYS: &str = "gguf_metadata_keys";
pub const ACTION_GGUF_METADATA_MISSING: &str = "gguf_metadata_missing";
pub const ACTION_GGUF_METADATA_LOADED: &str = "gguf_metadata_loaded";
pub const ACTION_GGUF_VOCAB_SIZE_DERIVED: &str = "gguf_vocab_size_derived"; // TEAM-089
pub const ACTION_GGUF_LOAD_WEIGHTS: &str = "gguf_load_weights";
pub const ACTION_GGUF_WEIGHTS_FAILED: &str = "gguf_weights_failed";
pub const ACTION_GGUF_LOAD_COMPLETE: &str = "gguf_load_complete";

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-039: Dual-output narration wrapper
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

use crate::http::{narration_channel, sse::InferenceEvent};
use observability_narration_core::NarrationFields;

/// Emit narration to BOTH stdout (tracing) AND SSE stream (if in request context)
///
/// This is the worker-specific wrapper that implements dual-output narration:
/// 1. Always emits to tracing (stdout → logs) for operators/debugging
/// 2. If in HTTP request context, also emits to SSE stream for users
///
/// TEAM-039: This enables real-time visibility in rbee-keeper shell
pub fn narrate_dual(fields: NarrationFields) {
    // 1. ALWAYS emit to tracing (for operators/developers)
    observability_narration_core::narrate(fields.clone(), observability_narration_core::NarrationLevel::Info);

    // 2. IF in HTTP request context, ALSO emit to SSE (for users)
    let sse_event = InferenceEvent::Narration {
        actor: fields.actor.to_string(),
        action: fields.action.to_string(),
        target: fields.target.clone(),
        human: fields.human.clone(),
        cute: fields.cute.clone(),
        story: fields.story.clone(),
        correlation_id: fields.correlation_id.clone(),
        job_id: fields.job_id.clone(),
    };

    // Send to SSE channel (returns false if no active request)
    let _ = narration_channel::send_narration(sse_event);
}
