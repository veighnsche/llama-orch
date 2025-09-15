use regex::Regex;
use super::World;

pub fn registry() -> Vec<Regex> {
    vec![
        // Enqueue + Stream
        Regex::new(r"^an OrchQueue API endpoint$").unwrap(),
        Regex::new(r"^I enqueue a completion task with valid payload$").unwrap(),
        Regex::new(r"^I receive 202 Accepted with correlation id$").unwrap(),
        Regex::new(r"^I stream task events$").unwrap(),
        Regex::new(r"^I receive SSE events started, token, end$").unwrap(),
        Regex::new(r"^I receive SSE metrics frames$").unwrap(),
        Regex::new(r"^started includes queue_position and predicted_start_ms$").unwrap(),
        Regex::new(r"^SSE event ordering is per stream$").unwrap(),

        // Backpressure 429
        Regex::new(r"^an OrchQueue API endpoint under load$").unwrap(),
        Regex::new(r"^I enqueue a task beyond capacity$").unwrap(),
        Regex::new(r"^I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id$").unwrap(),
        Regex::new(r"^the error body includes policy_label retriable and retry_after_ms$").unwrap(),

        // Cancel
        Regex::new(r"^an existing queued task$").unwrap(),
        Regex::new(r"^I cancel the task$").unwrap(),
        Regex::new(r"^I receive 204 No Content with correlation id$").unwrap(),

        // Sessions
        Regex::new(r"^a session id$").unwrap(),
        Regex::new(r"^I query the session$").unwrap(),
        Regex::new(r"^I receive session info with ttl_ms_remaining turns kv_bytes kv_warmth$").unwrap(),
        Regex::new(r"^I delete the session$").unwrap(),
    ]
}

#[allow(dead_code)]
pub mod stubs {
    use super::World;
    pub fn given_api_endpoint(_w: &mut World) {}
    pub fn when_enqueue_valid_completion(_w: &mut World) {}
    pub fn then_202_with_corr(_w: &mut World) {}
    pub fn when_stream_events(_w: &mut World) {}
    pub fn then_sse_started_token_end(_w: &mut World) {}
    pub fn then_sse_metrics_frames(_w: &mut World) {}
    pub fn then_started_includes_queue_pos_eta(_w: &mut World) {}
    pub fn then_sse_ordering_per_stream(_w: &mut World) {}

    pub fn given_api_under_load(_w: &mut World) {}
    pub fn when_enqueue_beyond_capacity(_w: &mut World) {}
    pub fn then_429_with_headers_and_corr(_w: &mut World) {}
    pub fn then_error_body_policy_label_retry(_w: &mut World) {}

    pub fn given_existing_queued_task(_w: &mut World) {}
    pub fn when_cancel_task(_w: &mut World) {}
    pub fn then_204_with_corr(_w: &mut World) {}

    pub fn given_session_id(_w: &mut World) {}
    pub fn when_query_session(_w: &mut World) {}
    pub fn then_session_info_fields(_w: &mut World) {}
    pub fn when_delete_session(_w: &mut World) {}
}
