//! E2E Integration Test: Full Axum request flow with narration.
//!
//! This test simulates a complete request lifecycle:
//! 1. Client sends request with correlation ID
//! 2. Axum middleware extracts/validates correlation ID
//! 3. Handler emits narration events
//! 4. Response includes correlation ID
//! 5. All narration events are captured and verified

#[cfg(feature = "axum")]
mod axum_e2e {
    use axum::{
        body::Body,
        extract::{Extension, Json},
        http::{Request, StatusCode},
        middleware,
        response::IntoResponse,
        routing::post,
        Router,
    };
    use observability_narration_core::{
        axum::correlation_middleware,
        generate_correlation_id, validate_correlation_id,
        Narration, CaptureAdapter,
        ACTOR_WORKER_ORCD, ACTION_INFERENCE_START, ACTION_INFERENCE_COMPLETE,
    };
    use serde::{Deserialize, Serialize};
    use serial_test::serial;
    use tower::util::ServiceExt;

    #[derive(Deserialize)]
    struct ExecuteRequest {
        job_id: String,
        prompt: String,
    }

    #[derive(Serialize)]
    struct ExecuteResponse {
        job_id: String,
        result: String,
    }

    /// Simulates a foundation engineer's handler
    async fn execute_handler(
        Extension(correlation_id): Extension<String>,
        Json(payload): Json<ExecuteRequest>,
    ) -> Result<impl IntoResponse, StatusCode> {
        // Narrate request received
        Narration::new(ACTOR_WORKER_ORCD, ACTION_INFERENCE_START, &payload.job_id)
            .human(format!("Received execute request for job {}", payload.job_id))
            .correlation_id(&correlation_id)
            .job_id(&payload.job_id)
            .emit();

        // Simulate processing
        let result = format!("Processed: {}", payload.prompt);

        // Narrate completion
        Narration::new(ACTOR_WORKER_ORCD, ACTION_INFERENCE_COMPLETE, &payload.job_id)
            .human(format!("Completed inference for job {}", payload.job_id))
            .correlation_id(&correlation_id)
            .job_id(&payload.job_id)
            .duration_ms(150)
            .tokens_in(100)
            .tokens_out(50)
            .emit();

        Ok(Json(ExecuteResponse {
            job_id: payload.job_id,
            result,
        }))
    }

    #[tokio::test]
    #[serial(capture_adapter)]
    async fn e2e_full_request_lifecycle_with_correlation_id() {
        let adapter = CaptureAdapter::install();

        // Foundation engineer sets up their Axum app
        let app = Router::new()
            .route("/execute", post(execute_handler))
            .layer(middleware::from_fn(correlation_middleware));

        // Client sends request with correlation ID
        let correlation_id = "550e8400-e29b-41d4-a716-446655440000";
        let request = Request::builder()
            .uri("/execute")
            .method("POST")
            .header("content-type", "application/json")
            .header("X-Correlation-ID", correlation_id)
            .body(Body::from(
                r#"{"job_id": "job-123", "prompt": "Hello world"}"#,
            ))
            .unwrap();

        // Process request
        let response = app.oneshot(request).await.unwrap();

        // Verify response
        assert_eq!(response.status(), StatusCode::OK);

        // Verify correlation ID in response
        let response_correlation_id = response
            .headers()
            .get("X-Correlation-ID")
            .and_then(|v| v.to_str().ok())
            .unwrap();
        assert_eq!(response_correlation_id, correlation_id);

        // Verify narration events were captured
        let captured = adapter.captured();
        assert_eq!(captured.len(), 2, "Expected 2 narration events (start + complete)");

        // Verify first event (inference start)
        assert_eq!(captured[0].actor, ACTOR_WORKER_ORCD);
        assert_eq!(captured[0].action, ACTION_INFERENCE_START);
        assert_eq!(captured[0].target, "job-123");
        assert!(captured[0].human.contains("Received execute request"));
        assert_eq!(captured[0].correlation_id, Some(correlation_id.to_string()));
        assert_eq!(captured[0].job_id, Some("job-123".to_string()));

        // Verify second event (inference complete)
        assert_eq!(captured[1].actor, ACTOR_WORKER_ORCD);
        assert_eq!(captured[1].action, ACTION_INFERENCE_COMPLETE);
        assert_eq!(captured[1].target, "job-123");
        assert!(captured[1].human.contains("Completed inference"));
        assert_eq!(captured[1].correlation_id, Some(correlation_id.to_string()));
        assert_eq!(captured[1].duration_ms, Some(150));
        assert_eq!(captured[1].tokens_in, Some(100));
        assert_eq!(captured[1].tokens_out, Some(50));
    }

    #[tokio::test]
    #[serial(capture_adapter)]
    async fn e2e_middleware_generates_correlation_id_when_missing() {
        let adapter = CaptureAdapter::install();

        let app = Router::new()
            .route("/execute", post(execute_handler))
            .layer(middleware::from_fn(correlation_middleware));

        // Client sends request WITHOUT correlation ID
        let request = Request::builder()
            .uri("/execute")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"job_id": "job-456", "prompt": "Test"}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        // Middleware should generate a correlation ID
        let response_correlation_id = response
            .headers()
            .get("X-Correlation-ID")
            .and_then(|v| v.to_str().ok())
            .unwrap();

        // Should be a valid UUID
        assert!(validate_correlation_id(response_correlation_id).is_some());

        // Narration events should have the generated correlation ID
        let captured = adapter.captured();
        assert_eq!(captured.len(), 2);
        assert!(captured[0].correlation_id.is_some());
        assert!(captured[1].correlation_id.is_some());
        assert_eq!(captured[0].correlation_id, captured[1].correlation_id);
    }

    #[tokio::test]
    #[serial(capture_adapter)]
    async fn e2e_error_handling_with_narration() {
        let adapter = CaptureAdapter::install();

        async fn error_handler(
            Extension(correlation_id): Extension<String>,
        ) -> impl IntoResponse {
            // Narrate error
            Narration::new(ACTOR_WORKER_ORCD, "error", "job-999")
                .human("Failed to process request: resource exhausted")
                .correlation_id(&correlation_id)
                .error_kind("ResourceExhausted")
                .emit_error();

            StatusCode::SERVICE_UNAVAILABLE
        }

        let app = Router::new()
            .route("/error", post(error_handler))
            .layer(middleware::from_fn(correlation_middleware));

        let correlation_id = generate_correlation_id();
        let request = Request::builder()
            .uri("/error")
            .method("POST")
            .header("X-Correlation-ID", &correlation_id)
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        // Verify error response
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

        // Verify correlation ID preserved
        let response_correlation_id = response
            .headers()
            .get("X-Correlation-ID")
            .and_then(|v| v.to_str().ok())
            .unwrap();
        assert_eq!(response_correlation_id, correlation_id);

        // Verify error narration captured
        let captured = adapter.captured();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].error_kind, Some("ResourceExhausted".to_string()));
        assert!(captured[0].human.contains("Failed"));
    }
}
