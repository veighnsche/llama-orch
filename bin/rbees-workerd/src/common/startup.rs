//! Worker startup and pool manager callback

use crate::narration::*;
use anyhow::Result;
use observability_narration_core::{narrate, NarrationFields};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct ReadyCallback {
    worker_id: String,
    vram_bytes: u64,
    uri: String,
}

/// Call back to pool manager to report worker ready
pub async fn callback_ready(
    callback_url: &str,
    worker_id: &str,
    vram_bytes: u64,
    port: u16,
) -> Result<()> {
    let uri = format!("http://localhost:{port}");

    let payload = ReadyCallback { worker_id: worker_id.to_string(), vram_bytes, uri };

    tracing::info!(
        callback_url = %callback_url,
        worker_id = %worker_id,
        vram_bytes,
        "Calling back to pool manager"
    );

    narrate(NarrationFields {
        actor: ACTOR_LLORCH_CANDLED,
        action: ACTION_CALLBACK_READY,
        target: callback_url.to_string(),
        human: format!(
            "Calling pool-managerd at {} (worker: {}, VRAM: {} MB)",
            callback_url,
            worker_id,
            vram_bytes / 1_000_000
        ),
        cute: Some(format!("Sending ready signal with {} MB of VRAM! 📞", vram_bytes / 1_000_000)),
        story: Some(format!(
            "\"I have {} MB VRAM ready!\" said worker-{}.",
            vram_bytes / 1_000_000,
            worker_id
        )),
        worker_id: Some(worker_id.to_string()),
        ..Default::default()
    });

    let client = reqwest::Client::new();
    let response = client.post(callback_url).json(&payload).send().await?;

    if !response.status().is_success() {
        narrate(NarrationFields {
            actor: ACTOR_LLORCH_CANDLED,
            action: ACTION_ERROR,
            target: callback_url.to_string(),
            human: format!("Pool manager callback failed: {}", response.status()),
            cute: Some(format!("Oh no! Pool-managerd didn't answer ({}). 😟", response.status())),
            error_kind: Some("callback_failed".to_string()),
            worker_id: Some(worker_id.to_string()),
            ..Default::default()
        });

        anyhow::bail!("Pool manager callback failed: {}", response.status());
    }

    tracing::info!("Pool manager callback successful");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_callback_ready_success() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/ready"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&mock_server)
            .await;

        let callback_url = format!("{}/ready", mock_server.uri());
        let result = callback_ready(&callback_url, "worker-1", 8_000_000_000, 8080).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_callback_ready_failure_status() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/ready"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&mock_server)
            .await;

        let callback_url = format!("{}/ready", mock_server.uri());
        let result = callback_ready(&callback_url, "worker-1", 8_000_000_000, 8080).await;

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Pool manager callback failed"));
        assert!(err_msg.contains("500"));
    }

    #[tokio::test]
    async fn test_callback_ready_network_error() {
        // Use invalid URL to trigger network error
        let callback_url = "http://localhost:1/invalid";
        let result = callback_ready(callback_url, "worker-1", 8_000_000_000, 8080).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_callback_ready_payload_structure() {
        let mock_server = MockServer::start().await;

        let expected_payload = serde_json::json!({
            "worker_id": "worker-123",
            "vram_bytes": 16_000_000_000u64,
            "uri": "http://localhost:9090"
        });

        Mock::given(method("POST"))
            .and(path("/ready"))
            .and(body_json(&expected_payload))
            .respond_with(ResponseTemplate::new(200))
            .mount(&mock_server)
            .await;

        let callback_url = format!("{}/ready", mock_server.uri());
        let result = callback_ready(&callback_url, "worker-123", 16_000_000_000, 9090).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_callback_ready_uri_formatting() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/ready"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&mock_server)
            .await;

        let callback_url = format!("{}/ready", mock_server.uri());

        // Test various port numbers
        let result1 = callback_ready(&callback_url, "worker-1", 8_000_000_000, 8080).await;
        assert!(result1.is_ok());

        let result2 = callback_ready(&callback_url, "worker-2", 8_000_000_000, 3000).await;
        assert!(result2.is_ok());

        let result3 = callback_ready(&callback_url, "worker-3", 8_000_000_000, 65535).await;
        assert!(result3.is_ok());
    }

    #[tokio::test]
    async fn test_callback_ready_various_vram_sizes() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/ready"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&mock_server)
            .await;

        let callback_url = format!("{}/ready", mock_server.uri());

        // 8GB
        let result1 = callback_ready(&callback_url, "worker-1", 8_000_000_000, 8080).await;
        assert!(result1.is_ok());

        // 16GB
        let result2 = callback_ready(&callback_url, "worker-2", 16_000_000_000, 8080).await;
        assert!(result2.is_ok());

        // 24GB
        let result3 = callback_ready(&callback_url, "worker-3", 24_000_000_000, 8080).await;
        assert!(result3.is_ok());

        // 80GB
        let result4 = callback_ready(&callback_url, "worker-4", 80_000_000_000, 8080).await;
        assert!(result4.is_ok());
    }

    #[tokio::test]
    async fn test_callback_ready_worker_id_formats() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/ready"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&mock_server)
            .await;

        let callback_url = format!("{}/ready", mock_server.uri());

        // Simple ID
        let result1 = callback_ready(&callback_url, "worker-1", 8_000_000_000, 8080).await;
        assert!(result1.is_ok());

        // UUID-like ID
        let result2 = callback_ready(
            &callback_url,
            "550e8400-e29b-41d4-a716-446655440000",
            8_000_000_000,
            8080,
        )
        .await;
        assert!(result2.is_ok());

        // Complex ID
        let result3 = callback_ready(&callback_url, "gpu-0-replica-2", 8_000_000_000, 8080).await;
        assert!(result3.is_ok());
    }

    #[tokio::test]
    async fn test_callback_ready_http_method() {
        let mock_server = MockServer::start().await;

        // Only accept POST
        Mock::given(method("POST"))
            .and(path("/ready"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&mock_server)
            .await;

        let callback_url = format!("{}/ready", mock_server.uri());
        let result = callback_ready(&callback_url, "worker-1", 8_000_000_000, 8080).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_callback_ready_retry_on_failure() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/ready"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&mock_server)
            .await;

        let callback_url = format!("{}/ready", mock_server.uri());
        let result = callback_ready(&callback_url, "worker-1", 8_000_000_000, 8080).await;

        // Should fail on first attempt (no built-in retry)
        assert!(result.is_err());
    }

    #[test]
    fn test_ready_callback_serialization() {
        let callback = ReadyCallback {
            worker_id: "worker-1".to_string(),
            vram_bytes: 8_000_000_000,
            uri: "http://localhost:8080".to_string(),
        };

        let json = serde_json::to_string(&callback).unwrap();
        assert!(json.contains("worker-1"));
        assert!(json.contains("8000000000"));
        assert!(json.contains("http://localhost:8080"));
    }

    #[test]
    fn test_ready_callback_deserialization() {
        let json =
            r#"{"worker_id":"worker-1","vram_bytes":8000000000,"uri":"http://localhost:8080"}"#;
        let callback: ReadyCallback = serde_json::from_str(json).unwrap();

        assert_eq!(callback.worker_id, "worker-1");
        assert_eq!(callback.vram_bytes, 8_000_000_000);
        assert_eq!(callback.uri, "http://localhost:8080");
    }
}

// ---
// Verified by Testing Team 🔍
