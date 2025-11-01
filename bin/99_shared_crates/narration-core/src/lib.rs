//! observability-narration-core — human-readable narration for users.
//!
//! **Narration shows users what's happening in the system.** 🐝
//!
//! Users see narration via:
//! - SSE streams (web UI)
//! - stderr (CLI tools)
//! - Logs (for operators)
//!
//! Narration is NOT redacted - users need full context to understand what's happening.
//!
//! # ⚠️ NOT FOR COMPLIANCE/AUDIT LOGGING
//!
//! **Audit logging is completely separate!**
//!
//! - ❌ Don't use narration for GDPR/PCI-DSS/SOC 2
//! - ❌ Don't use narration for security audit trails
//! - ❌ Don't use narration for legal evidence
//!
//! **For compliance/audit logging, see:** `bin/99_shared_crates/audit-logging/`
//!
//! Audit logs are:
//! - Hidden from users (file-only, never in UI)
//! - Redacted for compliance
//! - Tamper-evident
//! - For legal/security purposes
//!
//! # Features
//! - Human-readable narration with structured fields
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
//!
//! # Modular Structure (TEAM-300)
//!
//! The crate is organized into logical modules:
//!
//! - **`core/`** - Fundamental types (NarrationFields, NarrationLevel)
//! - **`api/`** - Public APIs (emit functions, builder, macros)
//! - **`taxonomy/`** - Constants (actors, actions)
//! - **`output/`** - Output mechanisms (SSE, capture adapter)
//! - **`context`** - Thread-local context for auto-injection
//! - **`mode`** - Narration mode selection (human/cute/story)
//! - **`format`** - Formatting utilities (messages, tables, job IDs) - TEAM-310
//! - **`correlation`** - Correlation ID utilities
//! - **`unicode`** - Unicode validation and sanitization
//!
//! # Existing modules (kept in place)
pub mod api;
pub mod core;
pub mod output;
pub mod taxonomy;

// Existing modules (kept in place)
pub mod context;
pub mod correlation;
pub mod format; // TEAM-310: Centralized formatting logic
pub mod mode;
pub mod unicode;

// TEAM-300: Process stdout capture for workers! 🎉💯✨
pub mod process_capture;

// TEAM-309: Thread-local actor storage for proc macros
mod thread_actor;

// ============================================================================
// Re-exports for backward compatibility
// ============================================================================

// Core types
pub use core::{NarrationFields, NarrationLevel};

// API functions
pub use api::narrate;

// TEAM-380: DELETED deprecated human() function (RULE ZERO)
// TEAM-380: DELETED Narration and NarrationFactory (RULE ZERO - use n!() macro)

#[doc(hidden)]
pub use api::macro_emit; // TEAM-312: Only function needed by n!() macro

// Context
pub use context::{with_narration_context, NarrationContext};

// Mode
pub use mode::{get_narration_mode, set_narration_mode, NarrationMode};

// Output
pub use output::{CaptureAdapter, CapturedNarration, NarrationEvent};

// Taxonomy
pub use taxonomy::*;

// Correlation utilities
pub use correlation::{
    from_header as correlation_from_header, generate_correlation_id,
    propagate as correlation_propagate, validate_correlation_id,
};

// Unicode utilities
pub use unicode::{sanitize_crlf, sanitize_for_json, validate_action, validate_actor};

// TEAM-310: Formatting utilities
// TEAM-312: Removed deprecated format_message and interpolate_context
pub use format::{
    format_array_table, format_message, format_object_table, format_value_compact, short_job_id,
    ACTION_WIDTH, ACTOR_WIDTH, FN_NAME_WIDTH, SHORT_JOB_ID_SUFFIX,
};

// TEAM-300: Process capture (re-export for convenience! 🎀)
pub use process_capture::ProcessNarrationCapture;

// TEAM-309: Thread-local target (for proc macros)
#[doc(hidden)]
pub use thread_actor::{
    clear_target as __internal_clear_target, set_target as __internal_set_target,
};

// SSE sink (for external use)
pub mod sse_sink {
    pub use crate::output::sse_sink::*;
}

// ============================================================================
// TEAM-297: Phase 0 - Ultra-concise narration macro
// ============================================================================

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
        $narration.emit_with_provenance(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
    }};
}

/// Create a module-level narration macro with a default actor.
///
/// This is the MOST ergonomic way to use narration! The macro captures the actor
/// at the call site, so you never have to repeat it.
///
/// # Example
/// ```rust,ignore
/// use observability_narration_core::{narration_macro, ACTOR_QUEEN_ROUTER, ACTION_STATUS};
///
/// // Create the macro with the actor baked in!
/// narration_macro!(ACTOR_QUEEN_ROUTER);
///
/// // Now use it - ONLY ACTION NEEDED!
/// narrate!(ACTION_STATUS)
///     .context("http://localhost:8080")
///     .context("8080")
///     .human("Found 2 hives on {0}, port {1}")
///     .emit();
/// ```
///
/// # TEAM-191: Ultimate Ergonomics with .context()!
/// This pattern is inspired by `println!` - define once, use everywhere.
/// The actor is captured at macro definition time, not at use time.
/// Use `.context()` to add values that can be referenced with `{0}`, `{1}`, etc.
#[macro_export]
macro_rules! narration_macro {
    ($actor:expr) => {
        /// Create a narration with the default actor.
        ///
        /// # Arguments
        /// - `action`: Action constant (e.g., `ACTION_STATUS`)
        ///
        /// # Example
        /// ```ignore
        /// narrate!(ACTION_STATUS)
        ///     .context("value1")
        ///     .context("value2")
        ///     .human("Status: {0}, {1}")
        ///     .emit();
        /// ```
        macro_rules! narrate {
            ($action:expr) => {
                $crate::Narration::new($actor, $action, stringify!($action))
            };
        }
    };
}

/// Ultra-concise narration macro using Rust's format!()
///
/// TEAM-297: This is the NEW API that reduces narration from 5 lines to 1 line!
/// Uses standard Rust format!() instead of custom {0}, {1} replacement.
///
/// # Simple usage (1 line instead of 5):
/// ```rust,ignore
/// use observability_narration_core::n;
///
/// n!("action", "message");
/// n!("action", "message {}", var);
/// n!("action", "msg {} and {}", var1, var2);
/// ```
///
/// # With narration mode (explicit):
/// ```rust,ignore
/// n!(human: "action", "Technical message");
/// n!(cute: "action", "🐝 Fun message");
/// n!(story: "action", "'Hello,' said the system");
/// ```
///
/// # All three modes (runtime selectable):
/// ```rust,ignore
/// n!("action",
///     human: "Technical message {}",
///     cute: "🐝 Fun message {}",
///     story: "'Message,' said {}",
///     var
/// );
/// ```
#[macro_export]
macro_rules! n {
    // Simple: n!("action", "message")
    ($action:expr, $msg:expr) => {{
        $crate::macro_emit($action, $msg, None, None, env!("CARGO_CRATE_NAME"), stdext::function_name!());
    }};

    // With format: n!("action", "msg {}", arg)
    ($action:expr, $fmt:expr, $($arg:expr),+ $(,)?) => {{
        $crate::macro_emit($action, &format!($fmt, $($arg),+), None, None, env!("CARGO_CRATE_NAME"), stdext::function_name!());
    }};

    // Explicit human: n!(human: "action", "msg {}", arg)
    (human: $action:expr, $fmt:expr $(, $arg:expr)* $(,)?) => {{
        $crate::macro_emit($action, &format!($fmt $(, $arg)*), None, None, env!("CARGO_CRATE_NAME"), stdext::function_name!());
    }};

    // Explicit cute: n!(cute: "action", "msg {}", arg)
    // TEAM-309: Use cute message as fallback for human (was empty string)
    (cute: $action:expr, $fmt:expr $(, $arg:expr)* $(,)?) => {{
        let msg = format!($fmt $(, $arg)*);
        $crate::macro_emit($action, &msg, Some(&msg), None, env!("CARGO_CRATE_NAME"), stdext::function_name!());
    }};

    // Explicit story: n!(story: "action", "msg {}", arg)
    // TEAM-309: Use story message as fallback for human (was empty string)
    (story: $action:expr, $fmt:expr $(, $arg:expr)* $(,)?) => {{
        let msg = format!($fmt $(, $arg)*);
        $crate::macro_emit($action, &msg, None, Some(&msg), env!("CARGO_CRATE_NAME"), stdext::function_name!());
    }};

    // Human + Cute: n!("action", human: "msg", cute: "msg", args...)
    ($action:expr,
     human: $human_fmt:expr,
     cute: $cute_fmt:expr
     $(, $arg:expr)* $(,)?
    ) => {{
        $crate::macro_emit(
            $action,
            &format!($human_fmt $(, $arg)*),
            Some(&format!($cute_fmt $(, $arg)*)),
            None,
            env!("CARGO_CRATE_NAME"),
            stdext::function_name!()
        );
    }};

    // Human + Story: n!("action", human: "msg", story: "msg", args...)
    ($action:expr,
     human: $human_fmt:expr,
     story: $story_fmt:expr
     $(, $arg:expr)* $(,)?
    ) => {{
        $crate::macro_emit(
            $action,
            &format!($human_fmt $(, $arg)*),
            None,
            Some(&format!($story_fmt $(, $arg)*)),
            env!("CARGO_CRATE_NAME"),
            stdext::function_name!()
        );
    }};

    // All three: n!("action", human: "msg", cute: "msg", story: "msg", args...)
    ($action:expr,
     human: $human_fmt:expr,
     cute: $cute_fmt:expr,
     story: $story_fmt:expr
     $(, $arg:expr)* $(,)?
    ) => {{
        $crate::macro_emit(
            $action,
            &format!($human_fmt $(, $arg)*),
            Some(&format!($cute_fmt $(, $arg)*)),
            Some(&format!($story_fmt $(, $arg)*)),
            env!("CARGO_CRATE_NAME"),
            stdext::function_name!()
        );
    }};
}

// TEAM-312: ENTROPY REMOVED
// - Deleted nd!() macro - use n!() with NarrationLevel::Debug instead
// - Deleted narrate_concise!() alias - just use n!()
//
// If you need debug-level narration, use the builder API:
//   NarrationFields { level: NarrationLevel::Debug, .. }.emit()
//
// The n!() macro is for Info level (99% of cases).
// Debug narration should be rare and explicit.
