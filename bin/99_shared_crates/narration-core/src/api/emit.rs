// TEAM-300: Modular reorganization - Emit functions
//! Core narration emit functions
//!
//! This module contains the main functions for emitting narration events.

use crate::core::{NarrationFields, NarrationLevel};
use crate::mode;
use crate::output::{capture, sse_sink}; // TEAM-306: Added capture for notify
use tracing::{event, Level};

/// Internal macro to emit a narration event at a specific level.
/// Reduces duplication across TRACE/DEBUG/INFO/WARN/ERROR levels.
/// TEAM-204: No redaction - narration is for debugging, not compliance
macro_rules! emit_event {
    ($level:expr, $fields:expr) => {
        event!(
            $level,
            actor = $fields.actor,
            action = $fields.action,
            target = %$fields.target,
            human = %$fields.human,
            fn_name = $fields.fn_name.as_deref(), // TEAM-311: Function name from #[narrate_fn]
            cute = $fields.cute.as_deref(),
            story = $fields.story.as_deref(),
            correlation_id = $fields.correlation_id.as_deref(),
            session_id = $fields.session_id.as_deref(),
            job_id = $fields.job_id.as_deref(),
            task_id = $fields.task_id.as_deref(),
            pool_id = $fields.pool_id.as_deref(),
            replica_id = $fields.replica_id.as_deref(),
            worker_id = $fields.worker_id.as_deref(),
            hive_id = $fields.hive_id.as_deref(),
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
///
/// # ⚠️ NOT FOR COMPLIANCE
///
/// Narration is for USERS to see what's happening!
/// - NO redaction (users need full context)
/// - Visible in UI/CLI (not hidden)
/// - NOT for legal/security audit trails
///
/// **For compliance/audit logging, see:** `bin/99_shared_crates/audit-logging/`
pub fn narrate(fields: NarrationFields, level: NarrationLevel) {
    let Some(tracing_level) = level.to_tracing_level() else {
        return; // MUTE - no output
    };

    // TEAM-204: No redaction - users need to see what's happening
    // For audit logging (hidden, redacted), see: bin/99_shared_crates/audit-logging/

    // TEAM-297: Select message based on current narration mode
    // TEAM-299: Message selection still happens for tracing and SSE formatting,
    // even though we removed stderr output
    let mode = mode::get_narration_mode();
    let _message = match mode {
        mode::NarrationMode::Human => &fields.human,
        mode::NarrationMode::Cute => fields.cute.as_ref().unwrap_or(&fields.human),
        mode::NarrationMode::Story => fields.story.as_ref().unwrap_or(&fields.human),
    };

    // TEAM-299: Phase 1 Privacy Fix - CRITICAL SECURITY CHANGE
    // TEAM-310: Formatting now centralized in format.rs module
    //
    // REMOVED: Global stderr output (was line 559)
    //
    // WHY REMOVED:
    // Previous implementation printed ALL narration to global stderr:
    //   ⚠️ DEPRECATED FORMAT (pre-TEAM-310):
    //   eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, message);
    //
    //   This format is DEPRECATED. Use observability_narration_core::format::format_message() instead.
    //   New format: Bold first line + message on newline (20-char fields)
    //
    // This caused CRITICAL privacy violation in multi-tenant environments:
    //   - User A's narration visible to User B (data leak)
    //   - Sensitive data (job_id, inference prompts) exposed globally
    //   - No isolation between jobs (security violation)
    //
    // SECURITY BY DESIGN:
    // Code that doesn't exist cannot be exploited. Environment variables
    // and feature flags are bypassable. Complete removal is the ONLY
    // secure solution.
    //
    // NEW ARCHITECTURE:
    //   - SSE is PRIMARY output (job-scoped, secure, isolated)
    //   - No global stderr in narration-core (not even conditional)
    //   - Keeper displays via separate SSE subscription (Phase 4)
    //   - Tests use capture adapter (no stderr dependency)
    //   - Formatting: Centralized in format.rs (TEAM-310)
    //
    // See: .plan/PRIVACY_FIX_FINAL_APPROACH.md
    // See: .plan/PRIVACY_FIX_REQUIRED.md
    // See: TEAM_310_FORMAT_MODULE.md (formatting centralization)

    // TEAM-299: SSE is PRIMARY and ONLY output in narration-core
    // Job-scoped, secure, no privacy leaks.
    if sse_sink::is_enabled() {
        let _sse_sent = sse_sink::try_send(&fields);
        // Opportunistic delivery - failure is OK for cases like:
        //   - No job_id in fields (narration dropped, secure)
        //   - Channel doesn't exist yet (narration before channel creation)
        //   - Channel is full (backpressure)
        //   - Channel closed (job completed)
    }

    // Emit structured event at appropriate level using macro (for tracing subscribers if configured)
    match tracing_level {
        Level::TRACE => emit_event!(Level::TRACE, fields),
        Level::DEBUG => emit_event!(Level::DEBUG, fields),
        Level::INFO => emit_event!(Level::INFO, fields),
        Level::WARN => emit_event!(Level::WARN, fields),
        Level::ERROR => emit_event!(Level::ERROR, fields),
    }

    // Notify capture adapter if active (ORCH-3306)
    // TEAM-306: Always enabled - integration tests need this
    capture::notify(fields);
}

// TEAM-380: DELETED human() function (RULE ZERO - deprecated code removed)
// Use narrate() with NarrationFields or n!() macro instead
