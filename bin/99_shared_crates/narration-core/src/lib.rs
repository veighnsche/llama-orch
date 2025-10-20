//! observability-narration-core — shared, lightweight narration helper.
//!
//! Provides structured, human-readable narration for debugging and observability.
//! Implements ORCH-3300..3312 from the narration logging proposal.
//!
//! # Features
//! - Human-readable narration with structured fields
//! - Automatic secret redaction
//! - Test capture adapter for BDD assertions
//! - Correlation ID propagation
//! - Story snapshot generation
//! - **Cloud Profile**: OpenTelemetry integration, HTTP header propagation, auto-injection
//!
//! # Example (Basic)
//! ```rust
//! use observability_narration_core::{narrate, NarrationFields};
//!
//! narrate(NarrationFields {
//!     actor: "orchestratord",
//!     action: "admission",
//!     target: "session-abc123".to_string(),
//!     human: "Accepted request; queued at position 3 (ETA 420 ms) on pool 'default'".to_string(),
//!     correlation_id: Some("req-xyz".into()),
//!     session_id: Some("session-abc123".into()),
//!     pool_id: Some("default".into()),
//!     ..Default::default()
//! });
//! ```
//!
//! # Example (Cloud Profile - Auto-injection)
//! ```rust
//! use observability_narration_core::{narrate_auto, NarrationFields};
//!
//! // Automatically injects service identity and timestamp
//! narrate_auto(NarrationFields {
//!     actor: "pool-managerd",
//!     action: "spawn",
//!     target: "GPU0".to_string(),
//!     human: "Spawning engine llamacpp-v1".to_string(),
//!     pool_id: Some("default".into()),
//!     ..Default::default()
//! });
//! ```

pub mod auto;
#[cfg(feature = "axum")]
pub mod axum;
mod builder;
mod capture;
pub mod correlation;
pub mod http;
pub mod otel;
mod redaction;
pub mod trace;
pub mod unicode;

pub use auto::{current_timestamp_ms, narrate_auto, narrate_full, service_identity};
pub use builder::Narration;
pub use capture::{CaptureAdapter, CapturedNarration};

/// Macro to emit narration with automatic caller crate provenance.
///
/// This macro captures the caller's crate name and version at compile time,
/// providing accurate provenance information for debugging.
///
/// # Example
/// ```rust,ignore
/// use observability_narration_core::narrate;
///
/// narrate!(
///     Narration::new("rbee-keeper", "start", "queen")
///         .human("Starting queen-rbee")
/// );
/// ```
#[macro_export]
macro_rules! narrate {
    ($narration:expr) => {{
        $narration.emit_with_provenance(
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION")
        )
    }};
}
pub use correlation::{
    from_header as correlation_from_header, generate_correlation_id,
    propagate as correlation_propagate, validate_correlation_id,
};
pub use otel::narrate_with_otel_context;
pub use redaction::{redact_secrets, RedactionPolicy};
pub use unicode::{sanitize_crlf, sanitize_for_json, validate_action, validate_actor};

// Trace macros are exported via #[macro_export] in trace.rs

// ============================================================================
// Taxonomy: Actors
// ============================================================================

/// Core orchestration service
pub const ACTOR_ORCHESTRATORD: &str = "orchestratord";
/// GPU pool manager service
pub const ACTOR_POOL_MANAGERD: &str = "pool-managerd";
/// Worker daemon (inference service)
pub const ACTOR_WORKER_ORCD: &str = "worker-orcd";
/// Inference engine (llama.cpp, vLLM, etc.)
pub const ACTOR_INFERENCE_ENGINE: &str = "inference-engine";
/// VRAM residency manager
pub const ACTOR_VRAM_RESIDENCY: &str = "vram-residency";

/// Extract service name from a module path string.
///
/// Used by the `#[narrate(...)]` macro to infer actor from module path.
///
/// # Examples
/// ```
/// use observability_narration_core::extract_service_name;
///
/// assert_eq!(extract_service_name("llama_orch::orchestratord::admission"), "orchestratord");
/// assert_eq!(extract_service_name("llama_orch::pool_managerd::spawn"), "pool-managerd");
/// assert_eq!(extract_service_name("llama_orch::worker_orcd::inference"), "worker-orcd");
/// assert_eq!(extract_service_name("unknown::path"), "unknown");
/// ```
pub fn extract_service_name(module_path: &str) -> &'static str {
    let parts: Vec<&str> = module_path.split("::").collect();

    // Look for known service names
    for part in &parts {
        match *part {
            "orchestratord" => return ACTOR_ORCHESTRATORD,
            "pool_managerd" => return ACTOR_POOL_MANAGERD,
            "worker_orcd" => return ACTOR_WORKER_ORCD,
            "vram_residency" => return ACTOR_VRAM_RESIDENCY,
            "inference_engine" => return ACTOR_INFERENCE_ENGINE,
            _ => continue,
        }
    }

    // Fallback: return "unknown"
    "unknown"
}

#[cfg(test)]
mod extract_service_name_tests {
    use super::*;

    #[test]
    fn test_extract_service_name() {
        assert_eq!(
            extract_service_name("llama_orch::orchestratord::admission"),
            ACTOR_ORCHESTRATORD
        );
        assert_eq!(extract_service_name("llama_orch::pool_managerd::spawn"), ACTOR_POOL_MANAGERD);
        assert_eq!(extract_service_name("llama_orch::worker_orcd::inference"), ACTOR_WORKER_ORCD);
        assert_eq!(extract_service_name("llama_orch::vram_residency::seal"), ACTOR_VRAM_RESIDENCY);
        assert_eq!(extract_service_name("unknown::path"), "unknown");
    }
}

// ============================================================================
// Taxonomy: Actions
// ============================================================================

/// Admission queue operations
pub const ACTION_ADMISSION: &str = "admission";
pub const ACTION_ENQUEUE: &str = "enqueue";
pub const ACTION_DISPATCH: &str = "dispatch";

/// Worker lifecycle
pub const ACTION_SPAWN: &str = "spawn";
pub const ACTION_READY_CALLBACK: &str = "ready_callback";
pub const ACTION_HEARTBEAT_SEND: &str = "heartbeat_send";
pub const ACTION_HEARTBEAT_RECEIVE: &str = "heartbeat_receive";
pub const ACTION_SHUTDOWN: &str = "shutdown";

/// Inference operations
pub const ACTION_INFERENCE_START: &str = "inference_start";
pub const ACTION_INFERENCE_COMPLETE: &str = "inference_complete";
pub const ACTION_INFERENCE_ERROR: &str = "inference_error";
pub const ACTION_CANCEL: &str = "cancel";

/// VRAM operations
pub const ACTION_VRAM_ALLOCATE: &str = "vram_allocate";
pub const ACTION_VRAM_DEALLOCATE: &str = "vram_deallocate";
pub const ACTION_SEAL: &str = "seal";
pub const ACTION_VERIFY: &str = "verify";

/// Pool management
pub const ACTION_REGISTER: &str = "register";
pub const ACTION_DEREGISTER: &str = "deregister";
pub const ACTION_PROVISION: &str = "provision";

use serde::{Deserialize, Serialize};
use tracing::{event, Level};

/// Narration logging level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NarrationLevel {
    Mute,  // No output
    Trace, // Ultra-fine detail
    Debug, // Developer diagnostics
    Info,  // Narration backbone (default)
    Warn,  // Anomalies & degradations
    Error, // Operational failures
    Fatal, // Unrecoverable errors
}

impl NarrationLevel {
    fn to_tracing_level(self) -> Option<Level> {
        match self {
            NarrationLevel::Mute => None,
            NarrationLevel::Trace => Some(Level::TRACE),
            NarrationLevel::Debug => Some(Level::DEBUG),
            NarrationLevel::Info => Some(Level::INFO),
            NarrationLevel::Warn => Some(Level::WARN),
            NarrationLevel::Error => Some(Level::ERROR),
            NarrationLevel::Fatal => Some(Level::ERROR), // tracing doesn't have FATAL
        }
    }
}

/// Structured fields for narration events.
/// Implements ORCH-3304 field taxonomy.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NarrationFields {
    /// Who performed the action (e.g., "orchestratord", "pool-managerd")
    pub actor: &'static str,

    /// What action was performed (e.g., "admission", "spawn", "build")
    pub action: &'static str,

    /// What was acted upon (e.g., session_id, pool_id, replica_id)
    pub target: String,

    /// Human-readable description (ORCH-3305: ≤100 chars, present tense, SVO)
    pub human: String,

    /// Cute children's book narration (optional, whimsical storytelling)
    /// Example: "Tucked the model safely into GPU0's cozy VRAM blanket! 🛏️✨"
    pub cute: Option<String>,

    /// Story-mode dialogue narration (optional, conversation-focused)
    /// Only use when components are actually communicating!
    /// Example: "'Do you have 2GB VRAM?' asked orchestratord. 'No,' replied pool-managerd-3, 'only 512MB free.'"
    pub story: Option<String>,

    // Correlation and identity fields
    pub correlation_id: Option<String>,
    pub session_id: Option<String>,
    pub job_id: Option<String>,
    pub task_id: Option<String>,
    pub pool_id: Option<String>,
    pub replica_id: Option<String>,
    pub worker_id: Option<String>,

    // Contextual fields (ORCH-3304)
    pub error_kind: Option<String>,
    pub retry_after_ms: Option<u64>,
    pub backoff_ms: Option<u64>,
    pub duration_ms: Option<u64>,
    pub queue_position: Option<usize>,
    pub predicted_start_ms: Option<u64>,

    // Engine/model context
    pub engine: Option<String>,
    pub engine_version: Option<String>,
    pub model_ref: Option<String>,
    pub device: Option<String>,

    // Performance metrics
    pub tokens_in: Option<u64>,
    pub tokens_out: Option<u64>,
    pub decode_time_ms: Option<u64>,

    // Provenance (audit trail and debugging)
    /// Service name and version (e.g., "orchestratord@0.1.0")
    pub emitted_by: Option<String>,
    /// Unix timestamp in milliseconds
    pub emitted_at_ms: Option<u64>,
    /// Distributed trace ID (OpenTelemetry compatible)
    pub trace_id: Option<String>,
    /// Span ID within the trace
    pub span_id: Option<String>,
    /// Parent span ID (for span hierarchy)
    pub parent_span_id: Option<String>,
    /// Source location for dev builds (e.g., "data.rs:155")
    pub source_location: Option<String>,
}

/// Internal macro to emit a narration event at a specific level.
/// Reduces duplication across TRACE/DEBUG/INFO/WARN/ERROR levels.
macro_rules! emit_event {
    ($level:expr, $fields:expr, $human:expr, $cute:expr, $story:expr) => {
        event!(
            $level,
            actor = $fields.actor,
            action = $fields.action,
            target = %$fields.target,
            human = %$human,
            cute = $cute.as_deref(),
            story = $story.as_deref(),
            correlation_id = $fields.correlation_id.as_deref(),
            session_id = $fields.session_id.as_deref(),
            job_id = $fields.job_id.as_deref(),
            task_id = $fields.task_id.as_deref(),
            pool_id = $fields.pool_id.as_deref(),
            replica_id = $fields.replica_id.as_deref(),
            worker_id = $fields.worker_id.as_deref(),
            error_kind = $fields.error_kind.as_deref(),
            retry_after_ms = $fields.retry_after_ms,
            backoff_ms = $fields.backoff_ms,
            duration_ms = $fields.duration_ms,
            queue_position = $fields.queue_position,
            predicted_start_ms = $fields.predicted_start_ms,
            engine = $fields.engine.as_deref(),
            engine_version = $fields.engine_version.as_deref(),
            model_ref = $fields.model_ref.as_deref(),
            device = $fields.device.as_deref(),
            tokens_in = $fields.tokens_in,
            tokens_out = $fields.tokens_out,
            decode_time_ms = $fields.decode_time_ms,
            emitted_by = $fields.emitted_by.as_deref(),
            emitted_at_ms = $fields.emitted_at_ms,
            trace_id = $fields.trace_id.as_deref(),
            span_id = $fields.span_id.as_deref(),
            parent_span_id = $fields.parent_span_id.as_deref(),
            source_location = $fields.source_location.as_deref(),
        )
    };
}

/// Emit a narration event at a specific level.
/// Implements ORCH-3300, ORCH-3301, ORCH-3303.
///
/// TEAM-153: Outputs to both tracing (if configured) AND stderr for guaranteed visibility
pub fn narrate_at_level(fields: NarrationFields, level: NarrationLevel) {
    let Some(tracing_level) = level.to_tracing_level() else {
        return; // MUTE - no output
    };

    // Apply redaction to human text (ORCH-3302)
    let human = redact_secrets(&fields.human, RedactionPolicy::default());

    // Apply redaction to cute text if present
    let cute = fields.cute.as_ref().map(|c| redact_secrets(c, RedactionPolicy::default()));

    // Apply redaction to story text if present
    let story = fields.story.as_ref().map(|s| redact_secrets(s, RedactionPolicy::default()));

    // TEAM-153: Always output to stderr for guaranteed shell visibility
    // This works whether or not tracing subscriber is initialized
    // TEAM-155: Multi-line format for readability - [actor] on first line, message indented
    eprintln!("[{}]\n  {}", fields.actor, human);

    // Emit structured event at appropriate level using macro (for tracing subscribers if configured)
    match tracing_level {
        Level::TRACE => emit_event!(Level::TRACE, fields, human, cute, story),
        Level::DEBUG => emit_event!(Level::DEBUG, fields, human, cute, story),
        Level::INFO => emit_event!(Level::INFO, fields, human, cute, story),
        Level::WARN => emit_event!(Level::WARN, fields, human, cute, story),
        Level::ERROR => emit_event!(Level::ERROR, fields, human, cute, story),
    }

    // Notify capture adapter if active (ORCH-3306)
    #[cfg(any(test, feature = "test-support"))]
    {
        // Create redacted fields for capture
        let mut redacted_fields = fields;
        redacted_fields.human = human.to_string();
        redacted_fields.cute = cute.map(|c| c.to_string());
        redacted_fields.story = story.map(|s| s.to_string());
        capture::notify(redacted_fields);
    }
}

/// Emit INFO-level narration (default)
pub fn narrate(fields: NarrationFields) {
    narrate_at_level(fields, NarrationLevel::Info)
}

/// Emit WARN-level narration
pub fn narrate_warn(fields: NarrationFields) {
    narrate_at_level(fields, NarrationLevel::Warn)
}

/// Emit ERROR-level narration
pub fn narrate_error(fields: NarrationFields) {
    narrate_at_level(fields, NarrationLevel::Error)
}

/// Emit FATAL-level narration
pub fn narrate_fatal(fields: NarrationFields) {
    narrate_at_level(fields, NarrationLevel::Fatal)
}

/// Emit DEBUG-level narration (requires `debug-enabled` feature)
#[cfg(feature = "debug-enabled")]
pub fn narrate_debug(fields: NarrationFields) {
    narrate_at_level(fields, NarrationLevel::Debug)
}

/// Emit TRACE-level narration (requires `trace-enabled` feature)
#[cfg(feature = "trace-enabled")]
pub fn narrate_trace(fields: NarrationFields) {
    narrate_at_level(fields, NarrationLevel::Trace)
}

/// Legacy compatibility function for existing callers.
/// Prefer `narrate()` with full `NarrationFields` for new code.
#[deprecated(since = "0.1.0", note = "Use narrate() with NarrationFields instead")]
pub fn human<S: AsRef<str>>(actor: &'static str, action: &'static str, target: &str, msg: S) {
    narrate(NarrationFields {
        actor,
        action,
        target: target.to_string(),
        human: msg.as_ref().to_string(),
        ..Default::default()
    });
}
