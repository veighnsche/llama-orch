// TEAM-300: Modular reorganization - Core types
//! Core types for narration system
//!
//! This module contains the fundamental types used throughout the narration system:
//! - NarrationFields: The main data structure for narration events
//! - NarrationLevel: Logging levels for narration

use serde::{Deserialize, Serialize};
use tracing::Level;

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
    pub(crate) fn to_tracing_level(self) -> Option<Level> {
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
    /// TEAM-185: Added hive_id for multi-hive rbee operations
    pub hive_id: Option<String>,

    // Operation context (for job-based systems)
    /// The specific operation being performed (e.g., "worker_spawn", "infer", "model_download")
    /// Unlike action (which is static), this can be dynamic and operation-specific
    /// TEAM-185: Added operation field for job-based systems to track dynamic operation names
    pub operation: Option<String>,

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
