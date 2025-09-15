// Deterministic template for a minimal client over OrchQueue v1.
use contracts_api_types as api;

#[derive(Clone)]
pub struct Client {
    pub http: reqwest::Client,
    pub base: String,
}

impl Client {
    pub fn new(base: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base: base.into(),
        }
    }

    pub fn create_task(&self, body: &api::TaskRequest) -> reqwest::RequestBuilder {
        self.http.post(format!("{}/v1/tasks", self.base)).json(body)
    }

    pub fn stream_task(&self, id: &str) -> reqwest::RequestBuilder {
        self.http
            .get(format!("{}/v1/tasks/{}/stream", self.base, id))
            .header("Accept", "text/event-stream")
    }

    pub fn cancel_task(&self, id: &str) -> reqwest::RequestBuilder {
        self.http
            .post(format!("{}/v1/tasks/{}/cancel", self.base, id))
    }

    pub fn get_session(&self, id: &str) -> reqwest::RequestBuilder {
        self.http.get(format!("{}/v1/sessions/{}", self.base, id))
    }

    pub fn delete_session(&self, id: &str) -> reqwest::RequestBuilder {
        self.http
            .delete(format!("{}/v1/sessions/{}", self.base, id))
    }
}
