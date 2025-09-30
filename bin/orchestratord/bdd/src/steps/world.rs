use axum::body::{to_bytes, Body};
use serde_json::json;

use http::header::HeaderName;
use orchestratord::state::AppState;
use std::fmt;
use tower::util::ServiceExt; // for Router::oneshot

#[derive(cucumber::World)]
pub struct World {
    facts: Vec<serde_json::Value>,
    pub mode_commit: bool,
    pub state: AppState,
    pub last_status: Option<http::StatusCode>,
    pub last_headers: Option<http::HeaderMap>,
    pub last_body: Option<String>,
    pub corr_id: Option<String>,
    pub task_id: Option<String>,
    pub api_key: Option<String>,
    pub extra_headers: Vec<(String, String)>,
}

impl fmt::Debug for World {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("World")
            .field("facts_len", &self.facts.len())
            .field("mode_commit", &self.mode_commit)
            .field("last_status", &self.last_status)
            .field("corr_id", &self.corr_id)
            .field("task_id", &self.task_id)
            .field("api_key_present", &self.api_key.as_ref().map(|_| true).unwrap_or(false))
            .field("extra_headers_len", &self.extra_headers.len())
            .field("state", &"<AppState redacted>")
            .finish()
    }
}

impl Default for World {
    fn default() -> Self {
        Self {
            facts: Vec::new(),
            mode_commit: false,
            state: orchestratord::state::AppState::new(),
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
    pub fn push_fact<S: Into<serde_json::Value>>(&mut self, fact: S) {
        self.facts.push(fact.into());
    }

    pub fn get_fact_by_prefix(&self, prefix: &str) -> Option<String> {
        self.facts.iter()
            .find(|f| f.as_str().map(|s| s.starts_with(prefix)).unwrap_or(false))
            .and_then(|f| f.as_str().map(|s| s.to_string()))
    }

    pub async fn http_call(
        &mut self,
        method: http::Method,
        path: &str,
        body_json: Option<serde_json::Value>,
    ) -> anyhow::Result<()> {
        // Build the router with the world's state so we exercise real middleware
        let app = orchestratord::app::router::build_router(self.state.clone());

        // Construct request
        let mut req_builder = http::Request::builder().method(method.clone()).uri(path);
        {
            let headers = req_builder.headers_mut().expect("headers mut");
            if let Some(key) = &self.api_key {
                headers.insert("X-API-Key", key.parse().unwrap());
            }
            for (k, v) in self.extra_headers.drain(..) {
                let name = HeaderName::from_bytes(k.as_bytes()).unwrap();
                headers.insert(name, v.parse().unwrap());
            }
            if body_json.is_some() {
                headers.insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
            }
        }

        let body = match body_json {
            Some(v) => Body::from(v.to_string()),
            None => Body::empty(),
        };
        let req: http::Request<Body> = req_builder.body(body).unwrap();

        // Execute request against the app router
        let resp = app.oneshot(req).await?;

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
