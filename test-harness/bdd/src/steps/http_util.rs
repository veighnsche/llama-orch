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
