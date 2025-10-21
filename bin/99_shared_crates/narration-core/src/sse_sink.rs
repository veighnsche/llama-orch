//! SSE sink for distributed narration transport.
//!
//! Allows narration events to be sent over Server-Sent Events (SSE) channels
//! for remote observability in distributed systems.

use crate::NarrationFields;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

/// Global SSE broadcaster for narration events.
static SSE_BROADCASTER: once_cell::sync::Lazy<SseBroadcaster> =
    once_cell::sync::Lazy::new(|| SseBroadcaster::new());

/// Broadcaster for SSE narration events.
pub struct SseBroadcaster {
    sender: Arc<Mutex<Option<broadcast::Sender<NarrationEvent>>>>,
}

/// Narration event formatted for SSE transport.
#[derive(Debug, Clone, serde::Serialize)]
pub struct NarrationEvent {
    pub actor: String,
    pub action: String,
    pub target: String,
    pub human: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cute: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub story: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emitted_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emitted_at_ms: Option<u64>,
}

impl From<NarrationFields> for NarrationEvent {
    fn from(fields: NarrationFields) -> Self {
        Self {
            actor: fields.actor.to_string(),
            action: fields.action.to_string(),
            target: fields.target,
            human: fields.human,
            cute: fields.cute,
            story: fields.story,
            correlation_id: fields.correlation_id,
            job_id: fields.job_id,
            emitted_by: fields.emitted_by,
            emitted_at_ms: fields.emitted_at_ms,
        }
    }
}

impl SseBroadcaster {
    fn new() -> Self {
        Self { sender: Arc::new(Mutex::new(None)) }
    }

    /// Initialize the SSE broadcaster with a channel capacity.
    pub fn init(&self, capacity: usize) {
        let (tx, _) = broadcast::channel(capacity);
        *self.sender.lock().unwrap() = Some(tx);
    }

    /// Send a narration event to all SSE subscribers.
    pub fn send(&self, event: NarrationEvent) {
        if let Some(tx) = self.sender.lock().unwrap().as_ref() {
            // Ignore send errors (no subscribers is OK)
            let _ = tx.send(event);
        }
    }

    /// Subscribe to narration events.
    pub fn subscribe(&self) -> Option<broadcast::Receiver<NarrationEvent>> {
        self.sender.lock().unwrap().as_ref().map(|tx| tx.subscribe())
    }
}

/// Initialize the global SSE broadcaster.
///
/// Call this once at application startup if you want narration to be
/// transported over SSE in addition to stderr/tracing.
///
/// # Example
/// ```rust,ignore
/// use observability_narration_core::sse_sink;
///
/// #[tokio::main]
/// async fn main() {
///     sse_sink::init(1000); // Buffer up to 1000 events
///     // Now all narration will be sent to SSE subscribers
/// }
/// ```
pub fn init(capacity: usize) {
    SSE_BROADCASTER.init(capacity);
}

/// Send a narration event to SSE subscribers.
///
/// This is called automatically by `narrate_at_level` if SSE is initialized.
pub fn send(fields: &NarrationFields) {
    SSE_BROADCASTER.send(fields.clone().into());
}

/// Subscribe to narration events over SSE.
///
/// Returns None if SSE broadcaster hasn't been initialized.
///
/// # Example
/// ```rust,ignore
/// use observability_narration_core::sse_sink;
///
/// let mut rx = sse_sink::subscribe().expect("SSE not initialized");
/// while let Ok(event) = rx.recv().await {
///     println!("Narration: {}", event.human);
/// }
/// ```
pub fn subscribe() -> Option<broadcast::Receiver<NarrationEvent>> {
    SSE_BROADCASTER.subscribe()
}

/// Check if SSE broadcasting is enabled.
pub fn is_enabled() -> bool {
    SSE_BROADCASTER.sender.lock().unwrap().is_some()
}
