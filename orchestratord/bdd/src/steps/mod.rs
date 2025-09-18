pub mod control_plane;
pub mod data_plane;
pub mod deadlines_preemption;
pub mod error_taxonomy;
pub mod observability;
pub mod security;
pub mod world;

use regex::Regex;

/// Registry of step regex patterns used in local BDD features.
#[cfg_attr(not(test), allow(dead_code))]
pub fn registry() -> Vec<Regex> {
    vec![
        // control_plane
        Regex::new(r"^a Control Plane API endpoint$").unwrap(),
        Regex::new(r"^a pool id$").unwrap(),
        Regex::new(r"^I request pool health$").unwrap(),
        Regex::new(r"^I receive 200 with liveness readiness draining and metrics$").unwrap(),
        Regex::new(r"^I request pool drain with deadline_ms$").unwrap(),
        Regex::new(r"^draining begins$").unwrap(),
        Regex::new(r"^I request pool reload with new model_ref$").unwrap(),
        Regex::new(r"^reload succeeds and is atomic$").unwrap(),
        Regex::new(r"^reload fails and rolls back atomically$").unwrap(),
        Regex::new(r"^I request capabilities$").unwrap(),
        Regex::new(r"^I receive capabilities with engines and API version$").unwrap(),
        // data_plane & SSE & sessions
        Regex::new(r"^an OrchQueue API endpoint$").unwrap(),
        Regex::new(r"^I enqueue a completion task with valid payload$").unwrap(),
        Regex::new(r"^I receive 202 Accepted with correlation id$").unwrap(),
        Regex::new(r"^I stream task events$").unwrap(),
        Regex::new(r"^I receive SSE events started, token, end$").unwrap(),
        Regex::new(r"^I receive SSE metrics frames$").unwrap(),
        Regex::new(r"^started includes queue_position and predicted_start_ms$").unwrap(),
        Regex::new(r"^SSE event ordering is per stream$").unwrap(),
        Regex::new(r"^budget headers are present$").unwrap(),
        Regex::new(r"^SSE transcript artifact exists with events started token metrics end$").unwrap(),
        Regex::new(r"^queue full policy is reject$").unwrap(),
        Regex::new(r"^queue full policy is drop-lru$").unwrap(),
        Regex::new(r"^an OrchQueue API endpoint under load$").unwrap(),
        Regex::new(r"^I enqueue a task beyond capacity$").unwrap(),
        Regex::new(r"^I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id$").unwrap(),
        Regex::new(r"^the error body includes policy_label retriable and retry_after_ms$").unwrap(),
        Regex::new(r"^an existing queued task$").unwrap(),
        Regex::new(r"^I cancel the task$").unwrap(),
        Regex::new(r"^I receive 204 No Content with correlation id$").unwrap(),
        Regex::new(r"^a session id$").unwrap(),
        Regex::new(r"^I query the session$").unwrap(),
        Regex::new(r"^I receive session info with ttl_ms_remaining turns kv_bytes kv_warmth$").unwrap(),
        Regex::new(r"^I delete the session$").unwrap(),
        // deadlines & SSE metrics
        Regex::new(r"^a task with infeasible deadline$").unwrap(),
        Regex::new(r"^I receive error code DEADLINE_UNMET$").unwrap(),
        Regex::new(r"^SSE metrics include on_time_probability$").unwrap(),
        // error taxonomy
        Regex::new(r"^I trigger INVALID_PARAMS$").unwrap(),
        Regex::new(r"^I receive 400 with correlation id and error envelope code INVALID_PARAMS$").unwrap(),
        Regex::new(r"^I trigger POOL_UNAVAILABLE$").unwrap(),
        Regex::new(r"^I receive 503 with correlation id and error envelope code POOL_UNAVAILABLE$").unwrap(),
        Regex::new(r"^I trigger INTERNAL error$").unwrap(),
        Regex::new(r"^I receive 500 with correlation id and error envelope code INTERNAL$").unwrap(),
        Regex::new(r"^error envelope includes engine when applicable$").unwrap(),
        // observability placeholders
        Regex::new(r"^metrics conform to linter names and labels$").unwrap(),
        Regex::new(r"^label cardinality budgets are enforced$").unwrap(),
        Regex::new(r"^started event and admission logs$").unwrap(),
        Regex::new(r"^include queue_position and predicted_start_ms$").unwrap(),
        Regex::new(r"^logs do not contain secrets or API keys$").unwrap(),
        Regex::new(r"^noop$").unwrap(),
        Regex::new(r"^nothing happens$").unwrap(),
        Regex::new(r"^it passes$").unwrap(),
        // security
        Regex::new(r"^no API key is provided$").unwrap(),
        Regex::new(r"^I receive 401 Unauthorized$").unwrap(),
        Regex::new(r"^an invalid API key is provided$").unwrap(),
        Regex::new(r"^I receive 403 Forbidden$").unwrap(),
    ]
}
