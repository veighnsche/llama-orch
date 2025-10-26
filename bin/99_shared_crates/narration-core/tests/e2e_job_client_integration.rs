// TEAM-303: Job-client integration tests (simplified E2E)
//!
//! Tests for job-client → job-server integration with narration flow.
//!
//! # Approach
//!
//! Instead of building fake binaries, we use the existing job-server
//! infrastructure with a real HTTP server to test realistic scenarios.
//!
//! # Coverage
//!
//! - HTTP job submission
//! - SSE streaming
//! - Narration propagation through HTTP
//! - Job-client error handling

use observability_narration_core::{n, with_narration_context, NarrationContext};
use job_server::JobRegistry;
use std::sync::Arc;
use axum::{Router, routing::{get, post}, Json, extract::{Path, State}};
use tokio::net::TcpListener;

// TEAM-303: Import test harness (available for future use)
mod harness;

/// Test state for HTTP server
struct TestServerState {
    registry: Arc<JobRegistry<String>>,
}

/// Create job endpoint handler
async fn create_job_handler(
    State(state): State<Arc<TestServerState>>,
    Json(payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    // TEAM-303: Create job and SSE channel
    let job_id = state.registry.create_job();
    state.registry.set_payload(&job_id, payload);
    
    // Create SSE channel
    observability_narration_core::output::sse_sink::create_job_channel(
        job_id.clone(),
        1000
    );
    
    // Emit narration (simulating queen-rbee)
    let ctx = NarrationContext::new().with_job_id(&job_id);
    with_narration_context(ctx, async {
        n!("job_created", "Job created by server");
    }).await;
    
    Json(serde_json::json!({
        "job_id": job_id,
        "sse_url": format!("/v1/jobs/{}/stream", job_id)
    }))
}

/// SSE stream endpoint handler
async fn stream_job_handler(
    State(state): State<Arc<TestServerState>>,
    Path(job_id): Path<String>,
) -> axum::response::Response {
    use axum::response::sse::{Event, Sse};
    use futures::stream;
    use axum::response::IntoResponse;
    
    // TEAM-303: Get receiver from SSE channel
    let receiver = observability_narration_core::output::sse_sink::take_job_receiver(&job_id);
    
    if let Some(rx) = receiver {
        // Emit some test narration in background
        let job_id_clone = job_id.clone();
        tokio::spawn(async move {
            let ctx = NarrationContext::new().with_job_id(&job_id_clone);
            with_narration_context(ctx, async {
                n!("stream_start", "Stream started");
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                n!("stream_processing", "Processing request");
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                n!("stream_complete", "Request complete");
                // TEAM-304: Removed [DONE] emission - job-server sends it when channel closes
            }).await;
        });
        
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
        // No receiver available
        let empty_stream = stream::iter(vec![
            Ok::<_, std::io::Error>(Event::default().data("error: job not found"))
        ]);
        Sse::new(empty_stream).into_response()
    }
}

/// Start test HTTP server
/// Returns (state, actual_port)
async fn start_test_server() -> (Arc<TestServerState>, u16) {
    let state = Arc::new(TestServerState {
        registry: Arc::new(JobRegistry::new()),
    });
    
    let app = Router::new()
        .route("/v1/jobs", post(create_job_handler))
        .route("/v1/jobs/{job_id}/stream", get(stream_job_handler))
        .with_state(state.clone());
    
    // TEAM-303: Use port 0 for automatic port assignment
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap();
    
    let actual_port = listener.local_addr().unwrap().port();
    
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    
    // Give server time to start
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    
    (state, actual_port)
}

#[tokio::test]
async fn test_job_client_http_submission() {
    // TEAM-303: Test job-client submitting to real HTTP server
    
    let (_state, port) = start_test_server().await;
    
    // Use job-client to submit
    let client = job_client::JobClient::new(format!("http://localhost:{}", port));
    let operation = operations_contract::Operation::HiveList;
    
    let mut events = Vec::new();
    let result = client.submit_and_stream(operation, |line| {
        events.push(line.to_string());
        Ok(())
    }).await;
    
    assert!(result.is_ok(), "Job submission failed: {:?}", result.err());
    
    // Verify we received narration events
    assert!(!events.is_empty(), "No events received");
    assert!(events.iter().any(|e| e.contains("job_created")), "Missing job_created event");
    assert!(events.iter().any(|e| e.contains("stream_start")), "Missing stream_start event");
}

#[tokio::test]
async fn test_job_client_narration_sequence() {
    // TEAM-303: Test that narration events arrive in correct sequence
    
    let (_state, port) = start_test_server().await;
    
    let client = job_client::JobClient::new(format!("http://localhost:{}", port));
    let operation = operations_contract::Operation::HiveList;
    
    let mut events = Vec::new();
    client.submit_and_stream(operation, |line| {
        events.push(line.to_string());
        Ok(())
    }).await.unwrap();
    
    // Verify sequence
    let actions: Vec<&str> = events.iter()
        .filter_map(|e| {
            if e.contains("job_created") { Some("job_created") }
            else if e.contains("stream_start") { Some("stream_start") }
            else if e.contains("stream_processing") { Some("stream_processing") }
            else if e.contains("stream_complete") { Some("stream_complete") }
            else { None }
        })
        .collect();
    
    assert_eq!(actions, vec!["job_created", "stream_start", "stream_processing", "stream_complete"]);
}

#[tokio::test]
async fn test_job_client_concurrent_requests() {
    // TEAM-303: Test multiple concurrent job submissions
    
    let (_state, port) = start_test_server().await;
    
    let client = job_client::JobClient::new(format!("http://localhost:{}", port));
    
    // Submit 5 jobs concurrently
    let mut handles = Vec::new();
    for i in 0..5 {
        let client_clone = client.clone();
        let handle = tokio::spawn(async move {
            let operation = operations_contract::Operation::HiveList;
            let mut events = Vec::new();
            
            client_clone.submit_and_stream(operation, |line| {
                events.push(line.to_string());
                Ok(())
            }).await.unwrap();
            
            (i, events)
        });
        handles.push(handle);
    }
    
    // Wait for all
    let mut results = Vec::new();
    for handle in handles {
        let (i, events) = handle.await.unwrap();
        results.push((i, events));
    }
    
    // Verify all succeeded
    assert_eq!(results.len(), 5);
    for (i, events) in results {
        assert!(!events.is_empty(), "Job {} received no events", i);
        assert!(events.iter().any(|e| e.contains("job_created")), "Job {} missing job_created", i);
    }
}

#[tokio::test]
async fn test_job_client_with_different_operations() {
    // TEAM-303: Test different operation types
    
    let (_state, port) = start_test_server().await;
    
    let client = job_client::JobClient::new(format!("http://localhost:{}", port));
    
    // Test HiveList
    let mut events1 = Vec::new();
    client.submit_and_stream(operations_contract::Operation::HiveList, |line| {
        events1.push(line.to_string());
        Ok(())
    }).await.unwrap();
    assert!(!events1.is_empty());
    
    // Test HiveGet
    let mut events2 = Vec::new();
    client.submit_and_stream(
        operations_contract::Operation::HiveGet { alias: "localhost".to_string() },
        |line| {
            events2.push(line.to_string());
            Ok(())
        }
    ).await.unwrap();
    assert!(!events2.is_empty());
    
    // Test Status
    let mut events3 = Vec::new();
    client.submit_and_stream(operations_contract::Operation::Status, |line| {
        events3.push(line.to_string());
        Ok(())
    }).await.unwrap();
    assert!(!events3.is_empty());
}

#[tokio::test]
async fn test_job_client_error_handling() {
    // TEAM-303: Test error handling when server is unavailable
    
    let client = job_client::JobClient::new("http://localhost:19999"); // Non-existent server
    let operation = operations_contract::Operation::HiveList;
    
    let result = client.submit_and_stream(operation, |_line| Ok(())).await;
    
    assert!(result.is_err(), "Expected error for unavailable server");
}
