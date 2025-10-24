// TEAM-270: Worker HTTP API specification

use serde::{Deserialize, Serialize};

/// Worker HTTP API specification
///
/// ALL workers must implement these endpoints, regardless of implementation.
///
/// # Required Endpoints
///
/// 1. **GET /health** - Health check
/// 2. **POST /v1/infer** - Inference request
/// 3. **GET /info** - Worker information
///
/// # Optional Endpoints
///
/// - **POST /v1/cancel** - Cancel ongoing inference (future)
/// - **GET /metrics** - Prometheus metrics (future)
///
/// # Example Implementation
///
/// ```rust,ignore
/// use axum::{routing::{get, post}, Router, Json};
/// use worker_contract::{WorkerInfo, InferRequest, InferResponse};
///
/// async fn health() -> &'static str {
///     "ok"
/// }
///
/// async fn info() -> Json<WorkerInfo> {
///     // Return worker info
///     unimplemented!()
/// }
///
/// async fn infer(Json(req): Json<InferRequest>) -> Json<InferResponse> {
///     // Process inference request
///     unimplemented!()
/// }
///
/// let app = Router::new()
///     .route("/health", get(health))
///     .route("/info", get(info))
///     .route("/v1/infer", post(infer));
/// ```
pub struct WorkerApiSpec;

impl WorkerApiSpec {
    /// Health check endpoint
    ///
    /// **Endpoint:** `GET /health`
    ///
    /// **Returns:** `200 OK` with body `"ok"`
    ///
    /// **Purpose:** Quick liveness check (used by hive/queen)
    pub const HEALTH: &'static str = "/health";

    /// Worker info endpoint
    ///
    /// **Endpoint:** `GET /info`
    ///
    /// **Returns:** `200 OK` with JSON body containing `WorkerInfo`
    ///
    /// **Purpose:** Get complete worker state
    pub const INFO: &'static str = "/info";

    /// Inference endpoint
    ///
    /// **Endpoint:** `POST /v1/infer`
    ///
    /// **Body:** `InferRequest` (JSON)
    ///
    /// **Returns:** `InferResponse` (JSON or SSE stream)
    ///
    /// **Purpose:** Execute inference on loaded model
    pub const INFER: &'static str = "/v1/infer";
}

/// Inference request
///
/// Sent from queen to worker to execute inference.
///
/// # Fields
///
/// - `prompt`: Input text to generate from
/// - `max_tokens`: Maximum tokens to generate
/// - `temperature`: Sampling temperature (0.0-2.0)
/// - `stream`: Whether to stream tokens via SSE
///
/// # Example
///
/// ```
/// use worker_contract::InferRequest;
///
/// let request = InferRequest {
///     prompt: "Hello, world!".to_string(),
///     max_tokens: Some(100),
///     temperature: Some(0.7),
///     stream: Some(true),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferRequest {
    /// Input prompt
    pub prompt: String,

    /// Maximum tokens to generate (optional, worker decides default)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Sampling temperature (optional, default: 0.7)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Stream tokens via SSE (optional, default: false)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// Inference response
///
/// Returned from worker to queen after inference completes.
///
/// For streaming requests, tokens are sent via SSE and this response
/// is sent at the end.
///
/// # Fields
///
/// - `text`: Generated text
/// - `tokens_generated`: Number of tokens generated
/// - `duration_ms`: Inference duration in milliseconds
///
/// # Example
///
/// ```
/// use worker_contract::InferResponse;
///
/// let response = InferResponse {
///     text: "Hello! How can I help you today?".to_string(),
///     tokens_generated: 8,
///     duration_ms: 250,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferResponse {
    /// Generated text
    pub text: String,

    /// Number of tokens generated
    pub tokens_generated: u32,

    /// Inference duration in milliseconds
    pub duration_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_spec_constants() {
        assert_eq!(WorkerApiSpec::HEALTH, "/health");
        assert_eq!(WorkerApiSpec::INFO, "/info");
        assert_eq!(WorkerApiSpec::INFER, "/v1/infer");
    }

    #[test]
    fn test_infer_request_serialization() {
        let request = InferRequest {
            prompt: "Test prompt".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            stream: Some(true),
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: InferRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.prompt, "Test prompt");
        assert_eq!(deserialized.max_tokens, Some(100));
        assert_eq!(deserialized.temperature, Some(0.7));
        assert_eq!(deserialized.stream, Some(true));
    }

    #[test]
    fn test_infer_request_optional_fields() {
        let request = InferRequest {
            prompt: "Test".to_string(),
            max_tokens: None,
            temperature: None,
            stream: None,
        };

        let json = serde_json::to_string(&request).unwrap();

        // Optional fields should be omitted from JSON
        assert!(!json.contains("max_tokens"));
        assert!(!json.contains("temperature"));
        assert!(!json.contains("stream"));
    }

    #[test]
    fn test_infer_response_serialization() {
        let response = InferResponse {
            text: "Generated text".to_string(),
            tokens_generated: 42,
            duration_ms: 1500,
        };

        let json = serde_json::to_string(&response).unwrap();
        let deserialized: InferResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.text, "Generated text");
        assert_eq!(deserialized.tokens_generated, 42);
        assert_eq!(deserialized.duration_ms, 1500);
    }
}
