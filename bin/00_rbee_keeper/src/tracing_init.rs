//! Consolidated tracing initialization for rbee-keeper
//!
//! TEAM-336: Extracted repeated tracing setup from main.rs
//! TEAM-336: Different behavior for CLI vs GUI modes
//! TEAM-337: Use existing format_message_with_fn() for consistent formatting

use observability_narration_core::format_message;
use serde::{Deserialize, Serialize};
use tauri::Emitter; // TEAM-336: Required for app_handle.emit() in Tauri v2
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer}; // TEAM-337: Use existing formatting

/// Narration event payload for Tauri frontend
#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
pub struct NarrationEvent {
    pub level: String,
    pub message: String,
    pub timestamp: String,
}

/// Initialize tracing for CLI mode (stderr only)
pub fn init_cli_tracing() {
    fmt()
        .with_writer(std::io::stderr)
        .with_ansi(true)
        .with_line_number(false)
        .with_file(false)
        .with_target(false)
        // TEAM-335: NO .compact() - it buffers! Use default formatter for immediate output
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
}

/// Initialize tracing for GUI mode (stderr + Tauri events)
///
/// TEAM-336: Dual output - stderr for debugging + Tauri events for React sidebar
/// TEAM-337: Use existing format_message_with_fn() for consistent formatting
pub fn init_gui_tracing(app_handle: tauri::AppHandle) {
    // Layer 1: stderr output - use format_message_with_fn() for consistent formatting
    let stderr_layer = StderrNarrationLayer;

    // Layer 2: Tauri event emitter for React sidebar
    let tauri_layer = TauriNarrationLayer::new(app_handle);

    // Combine layers
    // TEAM-337: Check if tracing is already initialized
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(stderr_layer)
        .with(tauri_layer)
        .try_init()
        .expect("Failed to initialize GUI tracing - already initialized?");
}

/// Custom tracing layer that outputs to stderr using format_message_with_fn()
struct StderrNarrationLayer;

impl<S> Layer<S> for StderrNarrationLayer
where
    S: tracing::Subscriber,
    S: for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        // Extract event fields using same logic as TauriNarrationLayer
        let mut visitor = EventVisitor::default();
        event.record(&mut visitor);

        // Extract values before consuming visitor
        let action = visitor.action.clone().unwrap_or_else(|| "unknown".to_string());
        let fn_name = visitor.fn_name.clone().unwrap_or_else(|| "unknown".to_string());
        let message = visitor.extract_message();

        // TEAM-337: Use EXISTING format_message_with_fn() from narration-core
        eprint!("{}", format_message(&action, &message, &fn_name));
    }
}

/// Custom tracing layer that emits narration events to Tauri frontend
struct TauriNarrationLayer {
    app_handle: tauri::AppHandle,
}

impl TauriNarrationLayer {
    fn new(app_handle: tauri::AppHandle) -> Self {
        Self { app_handle }
    }
}

impl<S> Layer<S> for TauriNarrationLayer
where
    S: tracing::Subscriber,
    S: for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        // Extract event message and metadata
        let mut visitor = EventVisitor::default();
        event.record(&mut visitor);

        // TEAM-337: Use extract_message() to properly handle narration events
        let payload = NarrationEvent {
            level: event.metadata().level().to_string(),
            message: visitor.extract_message(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        // Emit to Tauri frontend (non-blocking, ignore errors)
        // TEAM-336: Tauri v2 uses emit() instead of emit_all()
        // TEAM-337: Requires core:event:allow-listen permission in tauri.conf.json
        let _ = self.app_handle.emit("narration", &payload);
    }
}

// ============================================================
// BUG FIX: TEAM-337 | EventVisitor not extracting narration messages
// ============================================================
// SUSPICION:
// - TEAM-336 thought EventVisitor was correctly extracting messages
// - Events weren't appearing in NarrationPanel
//
// INVESTIGATION:
// - Verified TauriNarrationLayer is registered and on_event() is called
// - Verified n!() macro emits tracing events via narrate_at_level()
// - Found that emit_event!() macro emits structured fields:
//   actor, action, target, human, fn_name, cute, story, etc.
// - EventVisitor was grabbing first field (actor), not the message
//
// ROOT CAUSE:
// - Narration events have structured fields, with message in "human" field
// - EventVisitor::record_str() just grabbed first field value
// - For n!("action", "message"), fields are: actor="rbee_keeper", action="action", human="message"
// - EventVisitor extracted "rbee_keeper" instead of "message"
//
// FIX:
// - Check field name == "human" (that's the actual message)
// - If "human" field not found, check for "message" field (standard tracing)
// - Fallback to building from actor/action if neither exists
//
// TESTING:
// - cargo run --bin rbee-keeper (GUI mode)
// - Click "Test" button in Narration panel
// - Verify 4 events appear with correct messages
// - Check browser console shows [NarrationPanel] Received event
// ============================================================

/// Visitor to extract message from tracing event
#[derive(Default)]
struct EventVisitor {
    message: String,
    human: Option<String>,
    actor: Option<String>,
    action: Option<String>,
    fn_name: Option<String>, // TEAM-337: Extract fn_name for format_message_with_fn()
}

impl tracing::field::Visit for EventVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        // TEAM-337: Extract specific fields from narration events
        match field.name() {
            "human" => {
                let msg = format!("{:?}", value);
                // Remove debug quotes
                self.human = Some(if msg.starts_with('"') && msg.ends_with('"') {
                    msg[1..msg.len() - 1].to_string()
                } else {
                    msg
                });
            }
            "actor" => {
                let actor_str = format!("{:?}", value);
                self.actor = Some(if actor_str.starts_with('"') && actor_str.ends_with('"') {
                    actor_str[1..actor_str.len() - 1].to_string()
                } else {
                    actor_str
                });
            }
            "action" => {
                let action_str = format!("{:?}", value);
                self.action = Some(if action_str.starts_with('"') && action_str.ends_with('"') {
                    action_str[1..action_str.len() - 1].to_string()
                } else {
                    action_str
                });
            }
            "fn_name" => {
                let fn_str = format!("{:?}", value);
                self.fn_name = Some(if fn_str.starts_with('"') && fn_str.ends_with('"') {
                    fn_str[1..fn_str.len() - 1].to_string()
                } else {
                    fn_str
                });
            }
            "message" => {
                // Standard tracing message field
                if self.message.is_empty() {
                    let msg = format!("{:?}", value);
                    self.message = if msg.starts_with('"') && msg.ends_with('"') {
                        msg[1..msg.len() - 1].to_string()
                    } else {
                        msg
                    };
                }
            }
            _ => {
                // Ignore other fields
            }
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        // TEAM-337: Extract specific string fields
        match field.name() {
            "human" => {
                self.human = Some(value.to_string());
            }
            "actor" => {
                self.actor = Some(value.to_string());
            }
            "action" => {
                self.action = Some(value.to_string());
            }
            "fn_name" => {
                self.fn_name = Some(value.to_string());
            }
            "message" => {
                // Standard tracing message field
                if self.message.is_empty() {
                    self.message = value.to_string();
                }
            }
            _ => {
                // Ignore other fields
            }
        }
    }
}

impl EventVisitor {
    /// Extract the final message, prioritizing "human" field from narration events
    fn extract_message(self) -> String {
        // Priority 1: "human" field from narration events
        if let Some(human) = self.human {
            return human;
        }

        // Priority 2: "message" field from standard tracing
        if !self.message.is_empty() {
            return self.message;
        }

        // Priority 3: Build from actor/action if available
        match (self.actor, self.action) {
            (Some(actor), Some(action)) => format!("[{}] {}", actor, action),
            (Some(actor), None) => actor,
            (None, Some(action)) => action,
            (None, None) => String::from("(no message)"),
        }
    }
}
