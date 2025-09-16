use axum::body::{to_bytes, Body};
use axum::response::IntoResponse;
use serde_json::json;
use axum::extract::State;
use contracts_api_types as api;

use orchestratord::http::handlers;

#[derive(Debug, cucumber::World)]
pub struct World {
    facts: Vec<serde_json::Value>,
    pub mode_commit: bool,
    pub state: orchestratord::state::AppState,
    pub last_status: Option<http::StatusCode>,
    pub last_headers: Option<http::HeaderMap>,
    pub last_body: Option<String>,
    pub corr_id: Option<String>,
    pub task_id: Option<String>,
    pub api_key: Option<String>,
    pub extra_headers: Vec<(String, String)>,
}

impl Default for World {
    fn default() -> Self {
        Self {
            facts: Vec::new(),
            mode_commit: false,
            state: orchestratord::state::default_state(),
            last_status: None,
            last_headers: None,
            last_body: None,
            corr_id: None,
            task_id: None,
            api_key: Some("valid".to_string()),
            extra_headers: Vec::new(),
        }
    }
}

impl World {
    pub fn push_fact<S: AsRef<str>>(&mut self, stage: S) {
        self.facts.push(json!({
            "stage": stage.as_ref(),
        }));
    }

    pub fn all_facts(&self) -> impl Iterator<Item = &serde_json::Value> {
        self.facts.iter()
    }

    pub async fn http_call(
        &mut self,
        method: http::Method,
        path: &str,
        body_json: Option<serde_json::Value>,
    ) -> anyhow::Result<()> {
        // Build request headers
        let mut headers = http::HeaderMap::new();
        if let Some(key) = &self.api_key {
            headers.insert("X-API-Key", key.parse().unwrap());
        }
        for (k, v) in self.extra_headers.drain(..) {
            headers.insert(k, v.parse().unwrap());
        }

        // Dispatch to handlers by path
        let resp = match (method, path) {
            (http::Method::POST, "/v1/tasks") => {
                let body: api::TaskRequest = serde_json::from_value(
                    body_json.unwrap_or_else(|| json!({}))
                )?;
                handlers::create_task(headers, State(self.state.clone()), axum::Json(body)).await
            }
            (http::Method::GET, p) if p.starts_with("/v1/tasks/") && p.ends_with("/stream") => {
                let id = p.trim_start_matches("/v1/tasks/")
                    .trim_end_matches("/stream")
                    .trim_matches('/').to_string();
                handlers::stream_task(headers, State(self.state.clone()), axum::extract::Path(id)).await
            }
            (http::Method::POST, p) if p.starts_with("/v1/tasks/") && p.ends_with("/cancel") => {
                let id = p.trim_start_matches("/v1/tasks/")
                    .trim_end_matches("/cancel")
                    .trim_matches('/').to_string();
                handlers::cancel_task(headers, State(self.state.clone()), axum::extract::Path(id)).await
            }
            (http::Method::GET, p) if p.starts_with("/v1/sessions/") => {
                let id = p.trim_start_matches("/v1/sessions/").to_string();
                handlers::get_session(headers, State(self.state.clone()), axum::extract::Path(id)).await
            }
            (http::Method::DELETE, p) if p.starts_with("/v1/sessions/") => {
                let id = p.trim_start_matches("/v1/sessions/").to_string();
                handlers::delete_session(headers, State(self.state.clone()), axum::extract::Path(id)).await
            }
            (http::Method::GET, p) if p.starts_with("/v1/pools/") && p.ends_with("/health") => {
                let id = p.trim_start_matches("/v1/pools/")
                    .trim_end_matches("/health")
                    .trim_matches('/').to_string();
                handlers::get_pool_health(headers, State(self.state.clone()), axum::extract::Path(id)).await
            }
            (http::Method::POST, p) if p.starts_with("/v1/pools/") && p.ends_with("/drain") => {
                let _id = p.trim_start_matches("/v1/pools/")
                    .trim_end_matches("/drain")
                    .trim_matches('/').to_string();
                let body: api::control::DrainRequest = serde_json::from_value(
                    body_json.unwrap_or_else(|| json!({"deadline_ms": 0}))
                )?;
                handlers::drain_pool(headers, State(self.state.clone()), axum::extract::Path(_id), axum::Json(body)).await
            }
            (http::Method::POST, p) if p.starts_with("/v1/pools/") && p.ends_with("/reload") => {
                let _id = p.trim_start_matches("/v1/pools/")
                    .trim_end_matches("/reload")
                    .trim_matches('/').to_string();
                let body: api::control::ReloadRequest = serde_json::from_value(
                    body_json.unwrap_or_else(|| json!({"new_model_ref":""}))
                )?;
                handlers::reload_pool(headers, State(self.state.clone()), axum::extract::Path(_id), axum::Json(body)).await
            }
            (http::Method::GET, "/v1/replicasets") => {
                handlers::list_replicasets(headers, State(self.state.clone())).await
            }
            (http::Method::GET, "/metrics") => handlers::metrics_endpoint().await,
            _ => (http::StatusCode::NOT_FOUND, Body::empty()).into_response(),
        };

        let status = resp.status();
        let headers_out = resp.headers().clone();
        let body_bytes = to_bytes(resp.into_body(), 1_048_576).await.unwrap_or_default();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap_or_default();

        self.corr_id = headers_out
            .get("X-Correlation-Id")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        self.last_status = Some(status);
        self.last_headers = Some(headers_out);
        self.last_body = Some(body_str);
        Ok(())
    }
}
