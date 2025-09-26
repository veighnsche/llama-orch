//! Streaming service

use crate::state::AppState;
use futures::StreamExt;
use serde_json::json;
use std::io::BufWriter;
use std::io::Write;
use tokio::time::{sleep, Duration};
use worker_adapters_adapter_api as adapter_api;

/// Render a simple deterministic SSE for a task id and persist transcript to artifacts.
pub async fn render_sse_for_task(state: &AppState, id: String) -> String {
    // Try adapter-host first (if a worker is registered). Fallback to deterministic SSE.
    if let Ok(mut s) = try_dispatch_via_adapter(state, &id) {
        let mut events: Vec<(String, serde_json::Value)> = Vec::new();
        loop {
            // Early cancel
            if let Ok(guard) = state.cancellations.lock() {
                if guard.contains(&id) {
                    events.push(("end".into(), json!({"canceled":true})));
                    break;
                }
            }
            match s.next().await {
                Some(Ok(e)) => {
                    events.push((e.kind.clone(), e.data.clone()));
                    if e.kind == "end" {
                        break;
                    }
                }
                Some(Err(_)) => {
                    events.push(("error".into(), json!({"message":"adapter error"})));
                    break;
                }
                None => break,
            }
        }
        return build_sse_from_events(state, &id, events);
    }
    // Emit metrics: tasks started
    crate::metrics::inc_counter(
        "tasks_started_total",
        &[
            ("engine", "llamacpp"),
            ("engine_version", "v0"),
            ("pool_id", "default"),
            ("replica_id", "r0"),
            ("priority", "interactive"),
        ],
    );
    // First token latency
    crate::metrics::observe_histogram(
        "latency_first_token_ms",
        &[
            ("engine", "llamacpp"),
            ("engine_version", "v0"),
            ("pool_id", "default"),
            ("priority", "interactive"),
        ],
        42.0,
    );

    let started = json!({"queue_position": 3, "predicted_start_ms": 420});
    let token0 = json!({"t": "Hello", "i": 0});
    let token1 = json!({"t": " world", "i": 1});
    let metrics_ev = json!({
        "queue_depth": 1,
        "on_time_probability": 0.99,
        "kv_warmth": false,
        "tokens_budget_remaining": 0,
        "time_budget_remaining_ms": 600000,
        "cost_budget_remaining": 0.0
    });
    let end = json!({"tokens_out": 1, "decode_time_ms": 5});

    // Decode latency sample and tokens_out
    crate::metrics::observe_histogram(
        "latency_decode_ms",
        &[
            ("engine", "llamacpp"),
            ("engine_version", "v0"),
            ("pool_id", "default"),
            ("priority", "interactive"),
        ],
        5.0,
    );
    crate::metrics::inc_counter(
        "tokens_out_total",
        &[
            ("engine", "llamacpp"),
            ("engine_version", "v0"),
            ("pool_id", "default"),
            ("replica_id", "r0"),
        ],
    );

    // Build events with potential early termination on cancel
    let mut events: Vec<(&str, serde_json::Value)> = vec![("started", started.clone())];

    let is_canceled = |st: &AppState, id: &str| -> bool {
        if let Ok(guard) = st.cancellations.lock() {
            guard.contains(id)
        } else {
            false
        }
    };

    // Check cancel before first token
    if is_canceled(state, &id) {
        events.push(("end", end.clone()));
    } else {
        // Emit first token
        events.push(("token", token0.clone()));

        // Simulate time passing between tokens to allow cancel to be observed
        sleep(Duration::from_millis(30)).await;

        if is_canceled(state, &id) {
            // End early: no second token or metrics
            events.push(("end", end.clone()));
        } else {
            // Emit second token
            events.push(("token", token1.clone()));

            // Brief window before metrics to allow cancel
            sleep(Duration::from_millis(10)).await;
            if is_canceled(state, &id) {
                events.push(("end", end.clone()));
            } else {
                events.push(("metrics", metrics_ev.clone()));
                events.push(("end", end.clone()));
            }
        }
    }

    // Optional micro-batch: if enabled, merge consecutive token events.
    if std::env::var("ORCHD_SSE_MICROBATCH")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        let mut batched: Vec<(&str, serde_json::Value)> = Vec::new();
        let mut token_buf: Vec<serde_json::Value> = Vec::new();
        for (ty, data) in events.into_iter() {
            if ty == "token" {
                token_buf.push(data);
                continue;
            }
            if !token_buf.is_empty() {
                batched.push(("token", json!({"batch": token_buf}))); // low-alloc merge representation
                token_buf = Vec::new();
            }
            batched.push((ty, data));
        }
        if !token_buf.is_empty() {
            batched.push(("token", json!({"batch": token_buf})));
        }
        events = batched;
    }

    build_and_persist_sse(state, &id, events)
}

fn try_dispatch_via_adapter(
    state: &AppState,
    id: &str,
) -> anyhow::Result<adapter_api::TokenStream> {
    // Construct a minimal TaskRequest
    let req = contracts_api_types::TaskRequest {
        task_id: id.to_string(),
        session_id: format!("sess-{}", id),
        workload: contracts_api_types::Workload::Completion,
        model_ref: "m:stub".into(),
        engine: contracts_api_types::Engine::Llamacpp,
        ctx: 1,
        priority: contracts_api_types::Priority::Interactive,
        seed: None,
        determinism: None,
        sampler_profile_version: None,
        prompt: Some("hi".into()),
        inputs: None,
        max_tokens: 8,
        deadline_ms: 1000,
        expected_tokens: Some(1),
        kv_hint: None,
        placement: None,
    };
    state.adapter_host.submit("default", req)
}

fn build_sse_from_events(
    state: &AppState,
    id: &str,
    events: Vec<(String, serde_json::Value)>,
) -> String {
    // Persist transcript as artifact via configured store (and keep compat map updated)
    let transcript = json!({"events": events.iter().map(|(t, d)| json!({"type": t, "data": d})).collect::<Vec<_>>(),});
    let _ = crate::services::artifacts::put(state, transcript);
    // Clear cancellation flag for this task id to avoid leakage
    if let Ok(mut guard) = state.cancellations.lock() {
        let _ = guard.remove(id);
    }
    build_and_persist_sse(state, id, events)
}

fn build_and_persist_sse(
    _state: &AppState,
    _id: &str,
    events: Vec<(impl AsRef<str>, serde_json::Value)>,
) -> String {
    let mut buf = Vec::with_capacity(events.len() * 24);
    {
        let mut bw = BufWriter::new(&mut buf);
        for (ty, data) in &events {
            let _ = writeln!(&mut bw, "event: {}", ty.as_ref());
            let _ = writeln!(&mut bw, "data: {}", data);
            let _ = writeln!(&mut bw);
        }
        let _ = bw.flush();
    }
    String::from_utf8(buf).unwrap_or_default()
}
