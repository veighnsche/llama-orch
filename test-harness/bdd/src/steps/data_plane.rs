use crate::steps::world::World;
use cucumber::{given, then, when};

#[given(regex = r"^an OrchQueue API endpoint$")]
pub async fn given_api_endpoint(_world: &mut World) {}

#[when(regex = r"^I enqueue a completion task with valid payload$")]
pub async fn when_enqueue_valid_completion(world: &mut World) {
    world.push_fact("enqueue.valid");
}

#[then(regex = r"^I receive 202 Accepted with correlation id$")]
pub async fn then_accepted_with_corr(_world: &mut World) {}

#[when(regex = r"^I stream task events$")]
pub async fn when_stream_events(world: &mut World) {
    world.push_fact("sse.start");
}

#[then(regex = r"^I receive SSE events started, token, end$")]
pub async fn then_sse_started_token_end(_world: &mut World) {}

#[then(regex = r"^I receive SSE metrics frames$")]
pub async fn then_sse_metrics_frames(_world: &mut World) {}

#[then(regex = r"^started includes queue_position and predicted_start_ms$")]
pub async fn then_started_includes_queue_eta(_world: &mut World) {}

#[then(regex = r"^SSE event ordering is per stream$")]
pub async fn then_sse_ordering_per_stream(_world: &mut World) {}

#[given(regex = r"^queue full policy is reject$")]
pub async fn given_queue_policy_reject(world: &mut World) {
    world.push_fact("queue.policy.reject");
}

#[given(regex = r"^queue full policy is drop-lru$")]
pub async fn given_queue_policy_drop_lru(world: &mut World) {
    world.push_fact("queue.policy.drop_lru");
}

#[given(regex = r"^queue full policy is shed-low-priority$")]
pub async fn given_queue_policy_shed_low_priority(world: &mut World) {
    world.push_fact("queue.policy.shed_low_priority");
}

#[given(regex = r"^an OrchQueue API endpoint under load$")]
pub async fn given_api_under_load(_world: &mut World) {}

#[when(regex = r"^I enqueue a task beyond capacity$")]
pub async fn when_enqueue_beyond_capacity(world: &mut World) {
    world.push_fact("enqueue.beyond_capacity");
}

#[then(regex = r"^I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id$")]
pub async fn then_backpressure_headers(_world: &mut World) {}

#[then(regex = r"^the error body includes policy_label retriable and retry_after_ms$")]
pub async fn then_error_body_advisory_fields(_world: &mut World) {}

#[given(regex = r"^an existing queued task$")]
pub async fn given_existing_queued_task(_world: &mut World) {}

#[when(regex = r"^I cancel the task$")]
pub async fn when_cancel_task(world: &mut World) {
    world.push_fact("cancel");
}

#[then(regex = r"^I receive 204 No Content with correlation id$")]
pub async fn then_no_content_with_corr(_world: &mut World) {}

#[given(regex = r"^a session id$")]
pub async fn given_session_id(_world: &mut World) {}

#[when(regex = r"^I query the session$")]
pub async fn when_query_session(world: &mut World) {
    world.push_fact("session.get");
}

#[then(regex = r"^I receive session info with ttl_ms_remaining turns kv_bytes kv_warmth$")]
pub async fn then_session_info_fields(_world: &mut World) {}

#[when(regex = r"^I delete the session$")]
pub async fn when_delete_session(world: &mut World) {
    world.push_fact("session.delete");
}
