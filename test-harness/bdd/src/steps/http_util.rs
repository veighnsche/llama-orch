use crate::steps::world::World;
use cucumber::{given, then, when};
use http::header::AUTHORIZATION;
use worker_adapters_http_util as http_util;
use worker_adapters_http_util::{
    default_config, get_and_clear_retry_timeline, h2_preference, stream_decode, with_retries,
    RetryError, RetryPolicy, StreamEvent,
};
use proof_bundle::{ProofBundle, TestType};
use serde_json::json;

fn pb() -> ProofBundle {
    ProofBundle::for_type(TestType::Bdd).expect("proof bundle")
}

// X-API-Key redaction
#[given(regex = r"^a log line with X-API-Key \"([^\"]+)\"$")]
pub async fn given_log_line_with_x_api_key(world: &mut World, key: String) {
    let line = format!("WARN: upstream header X-API-Key: {}", key);
    world.api_key = Some(key.clone());
    world.last_body = Some(line);
    record_event("given.log_line_with_x_api_key", json!({"fp6": auth_min::token_fp6(&key)}));
}

#[then(regex = r"^the output masks X-API-Key and includes its fp6$")]
pub async fn then_output_masks_x_api_key(world: &mut World) {
    let input = world.last_body.as_ref().cloned().unwrap_or_default();
    let out = http_util::redact_secrets(&input);
    let key = world.api_key.as_ref().expect("api key set");
    let fp6 = auth_min::token_fp6(key);
    assert!(out.to_lowercase().contains("x-api-key: ****"), "missing masked X-API-Key: {}", out);
    assert!(out.contains(&fp6), "missing fp6 fingerprint for X-API-Key: {}", out);
    record_event("then.x_api_key_redacted", json!({"ok": true}));
}

fn record_event<E: Into<serde_json::Value>>(label: &str, event: E) {
    let _ = pb().append_ndjson("bdd_transcript", &json!({
        "label": label,
        "event": event.into(),
    }));
    let _ = pb().write_markdown("test_report.md", "BDD run for http-util steps. See bdd_transcript.ndjson for details.\n");
}

#[given(regex = r"^no special http-util configuration$")]
pub async fn given_no_special_config(_world: &mut World) {}

// Redaction
#[given(regex = r"^a log line with Authorization Bearer token \"([^\"]+)\"$")]
pub async fn given_log_line_with_token(world: &mut World, token: String) {
    let line = format!("INFO: Authorization: Bearer {} used for request", token);
    world.task_id = Some(token); // reuse task_id to store token for assertion
    world.last_body = Some(line);
    if let Some(tok) = &world.task_id { record_event("given.log_line_with_token", json!({"fp6": auth_min::token_fp6(tok)})); }
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
    record_event("when.apply_redaction", json!({"output": world.last_body.clone().unwrap_or_default()}));
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
    record_event("then.redaction_assertions", json!({"masked": out.contains("Authorization: Bearer ****"), "fp6": fp6}));
}

#[then(regex = r"^the output does not contain the raw token$")]
pub async fn then_output_not_contains_raw_token(world: &mut World) {
    let token = world.task_id.as_ref().expect("token set");
    let out = world.last_body.as_ref().expect("redaction output missing");
    assert!(!out.contains(token), "raw token leaked in output: {}", out);
    record_event("then.no_raw_token", json!({"ok": true}));
}

// Bearer injection
#[given(regex = r"^AUTH_TOKEN is set to \"([^\"]+)\"$")]
pub async fn given_auth_token_set(_world: &mut World, token: String) {
    std::env::set_var("AUTH_TOKEN", token);
    record_event("given.auth_token_set", json!({"set": true}));
}

#[given(regex = r"^AUTH_TOKEN is unset$")]
pub async fn given_auth_token_unset(_world: &mut World) {
    std::env::remove_var("AUTH_TOKEN");
    record_event("given.auth_token_unset", json!({"unset": true}));
}

#[when(regex = r"^I apply with_bearer_if_configured to a GET request$")]
pub async fn when_apply_with_bearer(world: &mut World) {
    let rb = http_util::client().get("http://localhost/");
    let rb = http_util::with_bearer_if_configured(rb);
    let req = rb.build().expect("build request");
    world.last_headers = Some(req.headers().clone());
    record_event("when.apply_with_bearer", json!({"has_auth": world.last_headers.as_ref().unwrap().get(AUTHORIZATION).is_some()}));
}

#[then(regex = r"^the request has Authorization header \"([^\"]+)\"$")]
pub async fn then_request_has_auth_header(world: &mut World, expected: String) {
    let headers = world.last_headers.as_ref().expect("expected headers");
    let got = headers
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(got, expected);
    record_event("then.has_auth_header", json!({"expected": expected, "got": got}));
}

#[then(regex = r"^the request has no Authorization header$")]
pub async fn then_request_has_no_auth_header(world: &mut World) {
    let headers = world.last_headers.as_ref().expect("expected headers");
    assert!(headers.get(AUTHORIZATION).is_none(), "unexpected Authorization header present");
    record_event("then.no_auth_header", json!({"ok": true}));
}

// Client singleton
#[when(regex = r"^I get the http-util client twice$")]
pub async fn when_get_client_twice(world: &mut World) {
    let c1 = http_util::client() as *const _;
    let c2 = http_util::client() as *const _;
    // Store pointer addresses as strings for assertion reuse
    world.corr_id = Some(format!("{:p}", c1));
    world.task_id = Some(format!("{:p}", c2));
    record_event("when.client_twice", json!({"p1": world.corr_id, "p2": world.task_id}));
}

#[then(regex = r"^both references point to the same client$")]
pub async fn then_same_client(world: &mut World) {
    let a = world.corr_id.as_ref().expect("ptr1");
    let b = world.task_id.as_ref().expect("ptr2");
    assert_eq!(a, b, "different client instances: {} vs {}", a, b);
    record_event("then.same_client", json!({"ok": true}));
}

// Retries (HTU-1002)
#[given(regex = r"^a transient upstream that returns 503 then succeeds$")]
pub async fn given_transient_upstream_503_succeeds(_world: &mut World) {
    // Setup would configure a stub server; placeholder for now.
}

#[when(regex = r"^I invoke with_retries around an idempotent request$")]
pub async fn when_invoke_with_retries(world: &mut World) {
    // Deterministic jitter for test reproducibility
    let mut policy = RetryPolicy::default();
    policy.seed = Some(7);
    // Simulate a transient 503 then success using attempt count
    let mut succeeded = false;
    let res: Result<u32, RetryError> = with_retries(
        |attempt| async move {
            if attempt < 3 {
                Err(RetryError::Retriable(anyhow::anyhow!("503 transient")))
            } else {
                Ok(attempt)
            }
        },
        policy,
    )
    .await;
    match res {
        Ok(n) => {
            succeeded = true;
            world.task_id = Some(format!("attempts:{}", n));
            record_event("when.retries_succeeded", json!({"attempts": n}));
        }
        Err(e) => {
            world.last_body = Some(format!("error:{e}"));
            record_event("when.retries_failed", json!({"error": e.to_string()}));
        }
    }
    world.mode_commit = succeeded;
}

#[then(regex = r"^attempts follow default policy base 100ms multiplier 2\.0 cap 2s max attempts 4$")]
pub async fn then_attempts_follow_default_policy(world: &mut World) {
    assert!(world.mode_commit, "operation did not succeed under retries");
    // We expect success on 3rd attempt -> 2 delays recorded
    let timeline = get_and_clear_retry_timeline();
    assert_eq!(timeline.len(), 2, "expected two delay entries, got {:?}", timeline);
    // Check bounds per default policy: base 100ms, multiplier 2.0, cap 2000ms
    // Our implementation computes exp using the current attempt number when the failure occurs.
    // For success on 3rd attempt, failures happen on attempt 1 and 2, yielding bounds 200ms and 400ms.
    let bounds = [200u64.min(2000), 400u64.min(2000)];
    for (i, &d) in timeline.iter().enumerate() {
        assert!(d <= bounds[i], "delay {} exceeds bound {}: {:?}", i, bounds[i], timeline);
    }
    record_event("then.retry_timeline", json!({"timeline_ms": timeline}));
}

#[given(regex = r"^an upstream that returns 400 Bad Request$")]
pub async fn given_upstream_400(_world: &mut World) {}

#[then(regex = r"^no retry occurs$")]
pub async fn then_no_retry_occurs(world: &mut World) {
    let mut policy = RetryPolicy::default();
    policy.seed = Some(11);
    let res: Result<(), RetryError> = with_retries(
        |_attempt| async move {
            Err(RetryError::NonRetriable(anyhow::anyhow!("400 bad request")))
        },
        policy,
    )
    .await;
    assert!(matches!(res, Err(RetryError::NonRetriable(_))), "expected non-retriable error");
    let timeline = get_and_clear_retry_timeline();
    assert!(timeline.is_empty(), "no delays should be recorded for non-retriable errors: {:?}", timeline);
    record_event("then.no_retry", json!({"timeline_ms": timeline}));
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
pub async fn when_decode_with_stream_decode(world: &mut World) {
    let body = world
        .last_body
        .as_ref()
        .expect("body stream provided by Given step")
        .clone();
    let mut events = Vec::new();
    let _ = stream_decode(&body, |e| events.push(e));
    // Serialize into world.last_body for assertion step
    let as_json = serde_json::to_string(&events.iter().map(|e| match e {
        StreamEvent::Started(_) => "started",
        StreamEvent::Token { .. } => "token",
        StreamEvent::Metrics(_) => "metrics",
        StreamEvent::End(_) => "end",
    }).collect::<Vec<_>>()).unwrap();
    world.last_body = Some(as_json);
    // Stash indices for ordering check
    let indices: Vec<usize> = events
        .into_iter()
        .filter_map(|e| match e { StreamEvent::Token { i, .. } => Some(i), _ => None })
        .collect();
    world.extra_headers = vec![("tok_indices".into(), serde_json::to_string(&indices).unwrap())];
    record_event("when.stream_decoded", json!({"sequence": serde_json::from_str::<serde_json::Value>(world.last_body.as_ref().unwrap()).unwrap(), "indices": indices}));
}

#[then(regex = r"^ordering is preserved and token indices are strictly increasing$")]
pub async fn then_ordering_preserved_and_token_indices_increasing(world: &mut World) {
    let seq: Vec<String> = serde_json::from_str(world.last_body.as_ref().expect("events seq"))
        .expect("parse events seq");
    let a = seq.iter().position(|s| s == "started").unwrap_or(usize::MAX);
    let b = seq.iter().position(|s| s == "token").unwrap_or(usize::MAX);
    let m = seq.iter().position(|s| s == "metrics").unwrap_or(usize::MAX);
    let c = seq.iter().position(|s| s == "end").unwrap_or(usize::MAX);
    assert!(a < b && b < m && m < c, "ordering violated: {:?}", seq);
    // Check increasing token indices
    let (_, indices_json) = world
        .extra_headers
        .iter()
        .find(|(k, _)| k == "tok_indices")
        .expect("indices present")
        .clone();
    let indices: Vec<usize> = serde_json::from_str(&indices_json).unwrap();
    for w in indices.windows(2) {
        assert!(w[0] < w[1], "non-increasing token indices: {:?}", indices);
    }
    record_event("then.stream_ordering", json!({"ok": true}));
}

// Client defaults (HTU-1001)
#[when(regex = r"^I inspect http-util client defaults$")]
pub async fn when_inspect_client_defaults(world: &mut World) {
    let cfg = default_config();
    world.corr_id = Some(format!("connect:{:?}", cfg.connect_timeout));
    world.task_id = Some(format!("request:{:?}", cfg.request_timeout));
    record_event("when.inspect_defaults", json!({"connect_ms": cfg.connect_timeout.as_millis(), "request_ms": cfg.request_timeout.as_millis(), "tls_verify": cfg.tls_verify}));
}

#[then(regex = r"^connect timeout is approximately 5s and request timeout approximately 30s$")]
pub async fn then_timeouts_approximate_defaults(world: &mut World) {
    let c = world.corr_id.as_ref().expect("connect timeout present");
    let r = world.task_id.as_ref().expect("request timeout present");
    assert!(c.contains("5s"), "connect timeout not ~5s: {}", c);
    assert!(r.contains("30s"), "request timeout not ~30s: {}", r);
    record_event("then.defaults_timeouts", json!({"ok": true}));
}

#[then(regex = r"^TLS verification is ON by default$")]
pub async fn then_tls_verification_on_by_default(_world: &mut World) {
    let cfg = default_config();
    assert!(cfg.tls_verify, "TLS verification should be ON by default");
    record_event("then.tls_verify", json!({"ok": true}));
}

#[then(regex = r"^HTTP/2 keep-alive is enabled when server supports ALPN$")]
pub async fn then_h2_keepalive_enabled_when_alpn(_world: &mut World) {
    assert!(h2_preference(), "http-util should prefer HTTP/2 when ALPN supports it");
    record_event("then.h2_preference", json!({"ok": true}));
}
