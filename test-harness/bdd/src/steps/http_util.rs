use crate::steps::world::World;
use cucumber::{given, then, when};
use http::header::AUTHORIZATION;
use worker_adapters_http_util as http_util;

#[given(regex = r"^no special http-util configuration$")]
pub async fn given_no_special_config(_world: &mut World) {}

// Redaction
#[given(regex = r"^a log line with Authorization Bearer token \"([^\"]+)\"$")]
pub async fn given_log_line_with_token(world: &mut World, token: String) {
    let line = format!("INFO: Authorization: Bearer {} used for request", token);
    world.task_id = Some(token); // reuse task_id to store token for assertion
    world.last_body = Some(line);
}

#[when(regex = r"^I apply http-util redaction$")]
pub async fn when_apply_redaction(world: &mut World) {
    let input = world
        .last_body
        .as_ref()
        .cloned()
        .unwrap_or_else(|| "".to_string());
    let out = http_util::redact_secrets(&input);
    world.last_body = Some(out);
}

#[then(regex = r"^the output masks the token and includes its fp6$")]
pub async fn then_output_masks_and_includes_fp6(world: &mut World) {
    let token = world
        .task_id
        .as_ref()
        .expect("token should be set by Given step")
        .clone();
    let out = world.last_body.as_ref().expect("redaction output missing");
    let fp6 = auth_min::token_fp6(&token);
    assert!(out.contains("Authorization: Bearer ****"), "missing masked header: {}", out);
    assert!(out.contains(&fp6), "missing fp6 fingerprint in output: {}", out);
}

#[then(regex = r"^the output does not contain the raw token$")]
pub async fn then_output_not_contains_raw_token(world: &mut World) {
    let token = world.task_id.as_ref().expect("token set");
    let out = world.last_body.as_ref().expect("redaction output missing");
    assert!(!out.contains(token), "raw token leaked in output: {}", out);
}

// Bearer injection
#[given(regex = r"^AUTH_TOKEN is set to \"([^\"]+)\"$")]
pub async fn given_auth_token_set(_world: &mut World, token: String) {
    std::env::set_var("AUTH_TOKEN", token);
}

#[given(regex = r"^AUTH_TOKEN is unset$")]
pub async fn given_auth_token_unset(_world: &mut World) {
    std::env::remove_var("AUTH_TOKEN");
}

#[when(regex = r"^I apply with_bearer_if_configured to a GET request$")]
pub async fn when_apply_with_bearer(world: &mut World) {
    let rb = http_util::client().get("http://localhost/");
    let rb = http_util::with_bearer_if_configured(rb);
    let req = rb.build().expect("build request");
    world.last_headers = Some(req.headers().clone());
}

#[then(regex = r"^the request has Authorization header \"([^\"]+)\"$")]
pub async fn then_request_has_auth_header(world: &mut World, expected: String) {
    let headers = world.last_headers.as_ref().expect("expected headers");
    let got = headers
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(got, expected);
}

#[then(regex = r"^the request has no Authorization header$")]
pub async fn then_request_has_no_auth_header(world: &mut World) {
    let headers = world.last_headers.as_ref().expect("expected headers");
    assert!(headers.get(AUTHORIZATION).is_none(), "unexpected Authorization header present");
}

// Client singleton
#[when(regex = r"^I get the http-util client twice$")]
pub async fn when_get_client_twice(world: &mut World) {
    let c1 = http_util::client() as *const _;
    let c2 = http_util::client() as *const _;
    // Store pointer addresses as strings for assertion reuse
    world.corr_id = Some(format!("{:p}", c1));
    world.task_id = Some(format!("{:p}", c2));
}

#[then(regex = r"^both references point to the same client$")]
pub async fn then_same_client(world: &mut World) {
    let a = world.corr_id.as_ref().expect("ptr1");
    let b = world.task_id.as_ref().expect("ptr2");
    assert_eq!(a, b, "different client instances: {} vs {}", a, b);
}

// Retries (HTU-1002)
#[given(regex = r"^a transient upstream that returns 503 then succeeds$")]
pub async fn given_transient_upstream_503_succeeds(_world: &mut World) {
    // Setup would configure a stub server; placeholder for now.
}

#[when(regex = r"^I invoke with_retries around an idempotent request$")]
pub async fn when_invoke_with_retries(_world: &mut World) {
    // Pending implementation in http-util per spec HTU-1002
    todo!("pending http-util::with_retries implementation");
}

#[then(regex = r"^attempts follow default policy base 100ms multiplier 2\.0 cap 2s max attempts 4$")]
pub async fn then_attempts_follow_default_policy(_world: &mut World) {
    // Pending: verify attempt timing against policy and recorded seed
    todo!("pending http-util::with_retries verification");
}

#[given(regex = r"^an upstream that returns 400 Bad Request$")]
pub async fn given_upstream_400(_world: &mut World) {}

#[then(regex = r"^no retry occurs$")]
pub async fn then_no_retry_occurs(_world: &mut World) {
    todo!("pending http-util::with_retries non-retriable behavior");
}

// Streaming decode (HTU-1003)
#[given(regex = r"^a body stream with started token token metrics end$")]
pub async fn given_body_stream_started_token_metrics_end(world: &mut World) {
    // Provide a minimal SSE-like body transcript to decode later
    let body = "event: started\n\
data: {\"ts\":1}\n\
\n\
event: token\n\
data: {\"i\":0,\"t\":\"Hello\"}\n\
\n\
event: token\n\
data: {\"i\":1,\"t\":\"!\"}\n\
\n\
event: metrics\n\
data: {\"decode_time_ms\":5}\n\
\n\
event: end\n\
data: {}\n".to_string();
    world.last_body = Some(body);
}

#[when(regex = r"^I decode with stream_decode$")]
pub async fn when_decode_with_stream_decode(_world: &mut World) {
    // Pending implementation in http-util per spec HTU-1003
    todo!("pending http-util::stream_decode implementation");
}

#[then(regex = r"^ordering is preserved and token indices are strictly increasing$")]
pub async fn then_ordering_preserved_and_token_indices_increasing(_world: &mut World) {
    // Pending verification once decode is implemented
    todo!("pending verification of ordering and indices");
}
