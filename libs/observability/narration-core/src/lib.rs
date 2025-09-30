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
//!
//! # Example
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

mod capture;
mod redaction;

pub use capture::{CaptureAdapter, CapturedNarration};
pub use redaction::{redact_secrets, RedactionPolicy};

use serde::{Deserialize, Serialize};
use tracing::{event, Level};

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
    /// Source location for dev builds (e.g., "data.rs:155")
    pub source_location: Option<String>,
}

/// Emit a narration event with structured fields.
/// Implements ORCH-3300, ORCH-3301, ORCH-3303.
///
/// # Example
/// ```rust
/// use observability_narration_core::{narrate, NarrationFields};
///
/// narrate(NarrationFields {
///     actor: "pool-managerd",
///     action: "spawn",
///     target: "GPU0".to_string(),
///     human: "Spawning engine llamacpp-v1 for pool 'default' on GPU0".to_string(),
///     correlation_id: Some("req-xyz".into()),
///     pool_id: Some("default".into()),
///     replica_id: Some("r0".into()),
///     engine: Some("llamacpp-v1".into()),
///     device: Some("GPU0".into()),
///     ..Default::default()
/// });
/// ```
pub fn narrate(fields: NarrationFields) {
    // Apply redaction to human text (ORCH-3302)
    let human = redact_secrets(&fields.human, RedactionPolicy::default());
    
    // Emit structured event
    event!(
        Level::INFO,
        actor = fields.actor,
        action = fields.action,
        target = %fields.target,
        human = %human,
        correlation_id = fields.correlation_id.as_deref(),
        session_id = fields.session_id.as_deref(),
        job_id = fields.job_id.as_deref(),
        task_id = fields.task_id.as_deref(),
        pool_id = fields.pool_id.as_deref(),
        replica_id = fields.replica_id.as_deref(),
        worker_id = fields.worker_id.as_deref(),
        error_kind = fields.error_kind.as_deref(),
        retry_after_ms = fields.retry_after_ms,
        backoff_ms = fields.backoff_ms,
        duration_ms = fields.duration_ms,
        queue_position = fields.queue_position,
        predicted_start_ms = fields.predicted_start_ms,
        engine = fields.engine.as_deref(),
        engine_version = fields.engine_version.as_deref(),
        model_ref = fields.model_ref.as_deref(),
        device = fields.device.as_deref(),
        tokens_in = fields.tokens_in,
        tokens_out = fields.tokens_out,
        decode_time_ms = fields.decode_time_ms,
        emitted_by = fields.emitted_by.as_deref(),
        emitted_at_ms = fields.emitted_at_ms,
        trace_id = fields.trace_id.as_deref(),
        span_id = fields.span_id.as_deref(),
        source_location = fields.source_location.as_deref(),
    );
    
    // Notify capture adapter if active (ORCH-3306)
    capture::notify(fields);
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
