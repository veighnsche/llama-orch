// TEAM-110: Audited 2025-10-18 - ✅ CLEAN - Readiness check with loading progress URL
// TEAM-113: Verified 2025-10-18 - ✅ NOT KUBERNETES DRIFT
//
//! GET /v1/ready endpoint - Worker readiness check
//!
//! Returns the worker's readiness state for accepting inference requests.
//!
//! ⚠️ NOTE: This is NOT a Kubernetes readiness probe!
//! This endpoint is for rbee-hive to check if the worker is ready.
//! rbee-hive IS the orchestrator that monitors workers.
//! This is orchestrator-to-worker communication, not Kubernetes patterns.
//!
//! Created by: TEAM-045
//!
//! # Spec References
//! - BDD test-001.feature: Worker readiness polling

use crate::http::backend::InferenceBackend;
use crate::narration::{ACTION_HEALTH_CHECK, ACTOR_HTTP_SERVER};
use axum::{extract::State, Json};
use observability_narration_core::{narrate, NarrationFields};
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::debug;

/// Worker readiness response
///
/// Indicates whether the worker is ready to accept inference requests.
/// While loading, provides a `progress_url` for streaming loading status.
#[derive(Serialize)]
pub struct ReadyResponse {
    /// Whether the worker is ready for inference
    ready: bool,
    /// Current worker state: "loading", "idle", "busy"
    state: String,
    /// Optional URL for loading progress (SSE stream)
    #[serde(skip_serializing_if = "Option::is_none")]
    progress_url: Option<String>,
    /// Whether model is loaded (true when ready)
    #[serde(skip_serializing_if = "Option::is_none")]
    model_loaded: Option<bool>,
}

/// Handle GET /v1/ready
///
/// Returns readiness status and current worker state.
/// TEAM-045: Added for BDD test support
pub async fn handle_ready<B: InferenceBackend>(
    State(backend): State<Arc<Mutex<B>>>,
) -> Json<ReadyResponse> {
    debug!("Readiness check requested");

    let backend = backend.lock().await;
    let ready = backend.is_ready();

    // TEAM-045: Determine state based on readiness and health
    let state = if !ready {
        "loading".to_string()
    } else if backend.is_healthy() {
        "idle".to_string()
    } else {
        "error".to_string()
    };

    let progress_url = if ready { None } else { Some("/v1/loading/progress".to_string()) };

    let model_loaded = if ready { Some(true) } else { None };

    narrate(NarrationFields {
        actor: ACTOR_HTTP_SERVER,
        action: ACTION_HEALTH_CHECK,
        target: state.clone(),
        human: format!("Readiness check: ready={ready}, state={state}"),
        cute: if ready {
            Some("Ready to serve! 🚀".to_string())
        } else {
            Some("Still loading... ⏳".to_string())
        },
        ..Default::default()
    });

    Json(ReadyResponse { ready, state, progress_url, model_loaded })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ready_response_when_ready() {
        let response = ReadyResponse {
            ready: true,
            state: "idle".to_string(),
            progress_url: None,
            model_loaded: Some(true),
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["ready"], true);
        assert_eq!(json["state"], "idle");
        assert_eq!(json["model_loaded"], true);
        assert!(json.get("progress_url").is_none());
    }

    #[test]
    fn test_ready_response_when_loading() {
        let response = ReadyResponse {
            ready: false,
            state: "loading".to_string(),
            progress_url: Some("/v1/loading/progress".to_string()),
            model_loaded: None,
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["ready"], false);
        assert_eq!(json["state"], "loading");
        assert_eq!(json["progress_url"], "/v1/loading/progress");
        assert!(json.get("model_loaded").is_none());
    }

    #[test]
    fn test_ready_response_serialization() {
        let response = ReadyResponse {
            ready: true,
            state: "idle".to_string(),
            progress_url: None,
            model_loaded: Some(true),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"ready\":true"));
        assert!(json.contains("\"state\":\"idle\""));
        assert!(json.contains("\"model_loaded\":true"));
    }
}

// ---
// Built by TEAM-045 🐝
