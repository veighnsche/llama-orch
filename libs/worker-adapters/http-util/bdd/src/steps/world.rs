#[derive(Debug, Default, cucumber::World)]
pub struct BddWorld {
    pub facts: Vec<serde_json::Value>,
    pub mode_commit: bool,
    pub last_status: Option<http::StatusCode>,
    pub last_headers: Option<http::HeaderMap>,
    pub last_body: Option<String>,
    pub corr_id: Option<String>,
    pub task_id: Option<String>,
    pub api_key: Option<String>,
    pub extra_headers: Vec<(String, String)>,
}
