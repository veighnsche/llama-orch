// TEAM-303: Fake queen-rbee binary for E2E testing
//!
//! Simulates queen-rbee behavior for integration tests:
//! - Receives job submissions from keeper
//! - Creates jobs and SSE channels
//! - Forwards worker operations to hive
//! - Streams narration events back to keeper

use observability_narration_core::{n, with_narration_context, NarrationContext};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::Mutex;
use axum::{
    Router,
    routing::{get, post},
    Json,
    extract::{Path, State},
    http::HeaderMap,
};
use tokio::net::TcpListener;

/// Queen state
struct QueenState {
    jobs: Arc<Mutex<HashMap<String, serde_json::Value>>>,
    hive_url: Option<String>,
}

/// Job submission handler
async fn create_job_handler(
    State(state): State<Arc<QueenState>>,
    headers: HeaderMap,
    Json(payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    // TEAM-303: Extract correlation ID from headers
    let correlation_id = headers
        .get("x-correlation-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    
    // Create job
    let job_id = uuid::Uuid::new_v4().to_string();
    state.jobs.lock().await.insert(job_id.clone(), payload.clone());
    
    // Create SSE channel
    observability_narration_core::output::sse_sink::create_job_channel(
        job_id.clone(),
        1000
    );
    
    // Emit queen narration
    let mut ctx = NarrationContext::new().with_job_id(&job_id);
    if let Some(corr_id) = correlation_id.as_ref() {
        ctx = ctx.with_correlation_id(corr_id);
    }
    
    with_narration_context(ctx, async {
        n!("queen_receive", "Queen received job submission");
        
        // Check if this should be forwarded to hive
        if let Some(operation) = payload.get("operation").and_then(|v| v.as_str()) {
            if operation.starts_with("worker_") || operation.starts_with("model_") {
                if let Some(hive_url) = state.hive_url.as_ref() {
                    n!("queen_forward", "Forwarding to hive at {}", hive_url);
                    
                    // Forward to hive
                    let client = reqwest::Client::new();
                    let mut req = client
                        .post(format!("{}/v1/jobs", hive_url))
                        .json(&payload);
                    
                    // Forward correlation ID
                    if let Some(corr_id) = correlation_id.as_ref() {
                        req = req.header("x-correlation-id", corr_id);
                    }
                    
                    match req.send().await {
                        Ok(resp) => {
                            n!("queen_forward_success", "Hive accepted job");
                            
                            // Get hive's SSE stream and forward events
                            if let Ok(hive_response) = resp.json::<serde_json::Value>().await {
                                if let Some(hive_job_id) = hive_response.get("job_id").and_then(|v| v.as_str()) {
                                    // In a real implementation, we'd stream from hive
                                    // For testing, we just emit a marker
                                    n!("queen_hive_streaming", "Streaming from hive job {}", hive_job_id);
                                }
                            }
                        }
                        Err(e) => {
                            n!("queen_forward_error", "Failed to forward to hive: {}", e);
                        }
                    }
                }
            }
        }
        
        n!("queen_complete", "Queen job processing complete");
        // TEAM-304: Removed [DONE] emission - job-server sends it when channel closes
    }).await;
    
    Json(serde_json::json!({
        "job_id": job_id,
        "sse_url": format!("/v1/jobs/{}/stream", job_id),
        "correlation_id": correlation_id
    }))
}

/// SSE stream handler
async fn stream_job_handler(
    State(_state): State<Arc<QueenState>>,
    Path(job_id): Path<String>,
) -> axum::response::Response {
    use axum::response::sse::{Event, Sse};
    use futures::stream;
    use axum::response::IntoResponse;
    
    // Get receiver from SSE channel
    let receiver = observability_narration_core::output::sse_sink::take_job_receiver(&job_id);
    
    if let Some(rx) = receiver {
        // TEAM-304: Stream events and send [DONE] when channel closes
        let event_stream = stream::unfold((rx, false), |(mut rx, done_sent)| async move {
            if done_sent {
                return None;
            }
            
            match rx.recv().await {
                Some(event) => {
                    let data = event.formatted.clone();
                    Some((Ok::<_, std::io::Error>(Event::default().data(data)), (rx, false)))
                }
                None => {
                    // TEAM-304: Channel closed - send [DONE] signal
                    Some((Ok::<_, std::io::Error>(Event::default().data("[DONE]")), (rx, true)))
                }
            }
        });
        
        Sse::new(event_stream).into_response()
    } else {
        let empty_stream = stream::iter(vec![
            Ok::<_, std::io::Error>(Event::default().data("error: job not found"))
        ]);
        Sse::new(empty_stream).into_response()
    }
}

/// Health check endpoint
async fn health_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "fake-queen-rbee"
    }))
}

#[tokio::main]
async fn main() {
    // Get port from environment or use default
    let port: u16 = std::env::var("QUEEN_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8500);
    
    // Get hive URL if configured
    let hive_url = std::env::var("HIVE_URL").ok();
    
    let state = Arc::new(QueenState {
        jobs: Arc::new(Mutex::new(HashMap::new())),
        hive_url,
    });
    
    let app = Router::new()
        .route("/v1/jobs", post(create_job_handler))
        .route("/v1/jobs/{job_id}/stream", get(stream_job_handler))
        .route("/health", get(health_handler))
        .with_state(state);
    
    let listener = TcpListener::bind(format!("127.0.0.1:{}", port))
        .await
        .expect("Failed to bind");
    
    let actual_port = listener.local_addr().unwrap().port();
    
    eprintln!("fake-queen-rbee listening on port {}", actual_port);
    
    axum::serve(listener, app)
        .await
        .expect("Server failed");
}
