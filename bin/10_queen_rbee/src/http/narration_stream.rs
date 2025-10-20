//! SSE endpoint for streaming narration events.
//!
//! TEAM-164: All queen-rbee operations emit narration over SSE for distributed observability.

use axum::{
    response::sse::{Event, KeepAlive, Sse},
    response::IntoResponse,
};
use observability_narration_core::sse_sink;
use std::convert::Infallible;
use std::time::Duration;

/// GET /narration/stream - Stream all narration events from queen-rbee
///
/// This endpoint allows rbee-keeper and web UIs to observe all queen-rbee
/// operations in real-time, regardless of where the queen is running.
///
/// # Example
/// ```bash
/// curl -N http://localhost:8500/narration/stream
/// ```
pub async fn handle_narration_stream() -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let rx = sse_sink::subscribe();

    let event_stream = async_stream::stream! {
        if let Some(mut rx) = rx {
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        // Format as SSE event
                        let json = serde_json::to_string(&event).unwrap_or_default();
                        yield Ok::<_, Infallible>(Event::default().data(json));
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                        // Client too slow, some events were dropped
                        let msg = format!("{{\"warning\": \"Lagged by {} events\"}}", skipped);
                        yield Ok::<_, Infallible>(Event::default().data(msg));
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        // Channel closed, end stream
                        break;
                    }
                }
            }
        }
    };

    Sse::new(event_stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keepalive"),
    )
}
