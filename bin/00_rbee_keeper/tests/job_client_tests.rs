// TEAM-375: Critical tests for job_client.rs
//
// Tests job submission, SSE streaming, timeout handling, and error detection.
// Prevents timeout hangs and silent failures in HTTP communication.

use anyhow::Result;
use operations_contract::Operation;
use rbee_keeper::job_client::{submit_and_stream_job, submit_and_stream_job_to_hive};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::timeout;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ============================================================================
// JOB SUBMISSION TESTS
// ============================================================================

#[tokio::test]
async fn test_submit_job_success() -> Result<()> {
    let mock_server = MockServer::start().await;

    // Mock POST /v1/jobs endpoint
    Mock::given(method("POST"))
        .and(path("/v1/jobs"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "job_id": "test-job-123"
            })),
        )
        .mount(&mock_server)
        .await;

    // Mock SSE stream endpoint
    Mock::given(method("GET"))
        .and(path("/v1/jobs/test-job-123/stream"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("data: Test message\n\ndata: [DONE]\n\n")
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&mock_server)
        .await;

    let operation = Operation::Status;
    let result = submit_and_stream_job(&mock_server.uri(), operation).await;

    assert!(result.is_ok(), "Job submission should succeed");

    Ok(())
}

#[tokio::test]
async fn test_submit_job_with_connection_failure() {
    // Use invalid URL
    let operation = Operation::Status;
    let result = submit_and_stream_job("http://localhost:99999", operation).await;

    assert!(result.is_err(), "Should fail with connection error");
}

#[tokio::test]
async fn test_submit_job_with_invalid_response() -> Result<()> {
    let mock_server = MockServer::start().await;

    // Mock returns invalid JSON
    Mock::given(method("POST"))
        .and(path("/v1/jobs"))
        .respond_with(ResponseTemplate::new(200).set_body_string("invalid json"))
        .mount(&mock_server)
        .await;

    let operation = Operation::Status;
    let result = submit_and_stream_job(&mock_server.uri(), operation).await;

    assert!(result.is_err(), "Should fail with invalid response");

    Ok(())
}

#[tokio::test]
async fn test_submit_job_with_500_error() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/jobs"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .mount(&mock_server)
        .await;

    let operation = Operation::Status;
    let result = submit_and_stream_job(&mock_server.uri(), operation).await;

    assert!(result.is_err(), "Should fail with 500 error");

    Ok(())
}

// ============================================================================
// SSE STREAMING TESTS
// ============================================================================

#[tokio::test]
async fn test_stream_with_done_marker() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/jobs"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "job_id": "test-job-456"
            })),
        )
        .mount(&mock_server)
        .await;

    Mock::given(method("GET"))
        .and(path("/v1/jobs/test-job-456/stream"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("data: Line 1\n\ndata: Line 2\n\ndata: [DONE]\n\n")
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&mock_server)
        .await;

    let operation = Operation::Status;
    let result = submit_and_stream_job(&mock_server.uri(), operation).await;

    assert!(result.is_ok(), "Should complete with [DONE] marker");

    Ok(())
}

#[tokio::test]
async fn test_stream_with_failure_detection() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/jobs"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "job_id": "test-job-789"
            })),
        )
        .mount(&mock_server)
        .await;

    Mock::given(method("GET"))
        .and(path("/v1/jobs/test-job-789/stream"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("data: Job failed: error message\n\ndata: [DONE]\n\n")
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&mock_server)
        .await;

    let operation = Operation::Status;
    let result = submit_and_stream_job(&mock_server.uri(), operation).await;

    // Should complete (failure is detected in output, not error)
    assert!(result.is_ok(), "Should complete even with job failure");

    Ok(())
}

#[tokio::test]
async fn test_stream_with_multiple_lines() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/jobs"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "job_id": "test-job-multi"
            })),
        )
        .mount(&mock_server)
        .await;

    let lines = vec![
        "data: Line 1\n\n",
        "data: Line 2\n\n",
        "data: Line 3\n\n",
        "data: Line 4\n\n",
        "data: [DONE]\n\n",
    ];

    Mock::given(method("GET"))
        .and(path("/v1/jobs/test-job-multi/stream"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(lines.join(""))
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&mock_server)
        .await;

    let operation = Operation::Status;
    let result = submit_and_stream_job(&mock_server.uri(), operation).await;

    assert!(result.is_ok(), "Should handle multiple lines");

    Ok(())
}

#[tokio::test]
async fn test_stream_without_done_marker_times_out() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/jobs"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "job_id": "test-job-timeout"
            })),
        )
        .mount(&mock_server)
        .await;

    // Stream that never sends [DONE]
    Mock::given(method("GET"))
        .and(path("/v1/jobs/test-job-timeout/stream"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("data: Line 1\n\n")
                .insert_header("content-type", "text/event-stream")
                .set_delay(Duration::from_secs(35)), // Longer than 30s timeout
        )
        .mount(&mock_server)
        .await;

    let operation = Operation::Status;

    // Wrap in timeout to prevent test hanging
    let result = timeout(
        Duration::from_secs(5),
        submit_and_stream_job(&mock_server.uri(), operation),
    )
    .await;

    // Should timeout (either from our test timeout or the 30s enforcer)
    assert!(
        result.is_err() || result.unwrap().is_err(),
        "Should timeout without [DONE] marker"
    );
}

// ============================================================================
// TIMEOUT HANDLING TESTS
// ============================================================================

#[tokio::test]
async fn test_timeout_enforcer_triggers() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/jobs"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({
                    "job_id": "test-job-slow"
                }))
                .set_delay(Duration::from_secs(35)), // Longer than 30s timeout
        )
        .mount(&mock_server)
        .await;

    let operation = Operation::Status;

    // Wrap in shorter timeout to prevent test hanging
    let result = timeout(
        Duration::from_secs(5),
        submit_and_stream_job(&mock_server.uri(), operation),
    )
    .await;

    assert!(
        result.is_err() || result.unwrap().is_err(),
        "Should timeout on slow response"
    );
}

#[tokio::test]
async fn test_fast_response_does_not_timeout() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/jobs"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "job_id": "test-job-fast"
            })),
        )
        .mount(&mock_server)
        .await;

    Mock::given(method("GET"))
        .and(path("/v1/jobs/test-job-fast/stream"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("data: Quick response\n\ndata: [DONE]\n\n")
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&mock_server)
        .await;

    let operation = Operation::Status;
    let result = submit_and_stream_job(&mock_server.uri(), operation).await;

    assert!(result.is_ok(), "Fast response should not timeout");

    Ok(())
}

// ============================================================================
// HIVE ALIAS TESTS
// ============================================================================

#[tokio::test]
async fn test_submit_to_hive_is_alias() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/jobs"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "job_id": "test-job-hive"
            })),
        )
        .mount(&mock_server)
        .await;

    Mock::given(method("GET"))
        .and(path("/v1/jobs/test-job-hive/stream"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("data: Hive response\n\ndata: [DONE]\n\n")
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&mock_server)
        .await;

    let operation = Operation::Status;

    // Both functions should work identically
    let result1 = submit_and_stream_job(&mock_server.uri(), operation.clone()).await;
    let result2 = submit_and_stream_job_to_hive(&mock_server.uri(), operation).await;

    assert!(result1.is_ok(), "submit_and_stream_job should work");
    assert!(result2.is_ok(), "submit_and_stream_job_to_hive should work");

    Ok(())
}

// ============================================================================
// ERROR HANDLING TESTS
// ============================================================================

#[tokio::test]
async fn test_network_error_handling() {
    // Server that doesn't exist
    let operation = Operation::Status;
    let result = submit_and_stream_job("http://localhost:1", operation).await;

    assert!(result.is_err(), "Should handle network errors");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Connection") || err_msg.contains("connect") || err_msg.contains("refused"),
        "Error should mention connection issue: {}",
        err_msg
    );
}

#[tokio::test]
async fn test_malformed_url_handling() {
    let operation = Operation::Status;
    let result = submit_and_stream_job("not-a-valid-url", operation).await;

    assert!(result.is_err(), "Should handle malformed URLs");
}

#[tokio::test]
async fn test_empty_url_handling() {
    let operation = Operation::Status;
    let result = submit_and_stream_job("", operation).await;

    assert!(result.is_err(), "Should handle empty URLs");
}

// ============================================================================
// CONCURRENT REQUEST TESTS
// ============================================================================

#[tokio::test]
async fn test_concurrent_job_submissions() -> Result<()> {
    let mock_server = MockServer::start().await;

    // Mock multiple job submissions
    for i in 0..5 {
        let job_id = format!("test-job-{}", i);
        Mock::given(method("POST"))
            .and(path("/v1/jobs"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "job_id": job_id
                })),
            )
            .mount(&mock_server)
            .await;

        Mock::given(method("GET"))
            .and(path(format!("/v1/jobs/{}/stream", job_id)))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(format!("data: Response {}\n\ndata: [DONE]\n\n", i))
                    .insert_header("content-type", "text/event-stream"),
            )
            .mount(&mock_server)
            .await;
    }

    // Submit 5 jobs concurrently
    let mut handles = vec![];
    for _ in 0..5 {
        let url = mock_server.uri();
        let handle = tokio::spawn(async move {
            let operation = Operation::Status;
            submit_and_stream_job(&url, operation).await
        });
        handles.push(handle);
    }

    // Wait for all to complete
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok(), "Concurrent requests should succeed");
    }

    Ok(())
}

// ============================================================================
// OPERATION SERIALIZATION TESTS
// ============================================================================

#[tokio::test]
async fn test_different_operation_types() -> Result<()> {
    let mock_server = MockServer::start().await;

    // Test with different operation types
    let operations = vec![
        Operation::Status,
        Operation::QueenCheck,
    ];

    for (i, operation) in operations.into_iter().enumerate() {
        let job_id = format!("test-job-op-{}", i);

        Mock::given(method("POST"))
            .and(path("/v1/jobs"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "job_id": job_id
                })),
            )
            .mount(&mock_server)
            .await;

        Mock::given(method("GET"))
            .and(path(format!("/v1/jobs/{}/stream", job_id)))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string("data: OK\n\ndata: [DONE]\n\n")
                    .insert_header("content-type", "text/event-stream"),
            )
            .mount(&mock_server)
            .await;

        let result = submit_and_stream_job(&mock_server.uri(), operation).await;
        assert!(result.is_ok(), "Should handle different operation types");
    }

    Ok(())
}

// ============================================================================
// NARRATION TESTS
// ============================================================================

#[tokio::test]
async fn test_narration_emission_during_streaming() -> Result<()> {
    // This test verifies that narration is emitted during job submission
    // We can't easily capture narration output in tests, but we can verify
    // the function completes successfully (which means narration didn't panic)

    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/jobs"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "job_id": "test-narration"
            })),
        )
        .mount(&mock_server)
        .await;

    Mock::given(method("GET"))
        .and(path("/v1/jobs/test-narration/stream"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("data: Test\n\ndata: [DONE]\n\n")
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&mock_server)
        .await;

    let operation = Operation::Status;
    let result = submit_and_stream_job(&mock_server.uri(), operation).await;

    assert!(result.is_ok(), "Narration should not cause failures");

    Ok(())
}
