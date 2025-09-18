//! Streaming service

use serde_json::json;
use crate::state::AppState;

/// Render a simple deterministic SSE for a task id and persist transcript to artifacts.
pub async fn render_sse_for_task(state: &AppState, _id: String) -> String {
    // Emit metrics: tasks started
    crate::metrics::inc_counter(
        "tasks_started_total",
        &[("engine","llamacpp"),("engine_version","v0"),("pool_id","default"),("replica_id","r0"),("priority","interactive")]
    );
    // First token latency
    crate::metrics::observe_histogram(
        "latency_first_token_ms",
        &[("engine","llamacpp"),("engine_version","v0"),("pool_id","default"),("priority","interactive")],
        42.0,
    );

    let started = json!({"queue_position": 3, "predicted_start_ms": 420});
    let token0 = json!({"t": "Hello", "i": 0});
    let metrics_ev = json!({
        "queue_depth": 1,
        "on_time_probability": 0.99,
        "kv_warmth": false,
        "tokens_budget_remaining": 0,
        "time_budget_remaining_ms": 600000,
        "cost_budget_remaining": 0.0
    });
    let end = json!({"tokens_out": 1, "decode_ms": 5});

    // Decode latency sample and tokens_out
    crate::metrics::observe_histogram(
        "latency_decode_ms",
        &[("engine","llamacpp"),("engine_version","v0"),("pool_id","default"),("priority","interactive")],
        5.0,
    );
    crate::metrics::inc_counter(
        "tokens_out_total",
        &[("engine","llamacpp"),("engine_version","v0"),("pool_id","default"),("replica_id","r0")]
    );

    let events = vec![
        ("started", started.clone()),
        ("token", token0.clone()),
        ("metrics", metrics_ev.clone()),
        ("end", end.clone()),
    ];

    // Persist transcript as artifact via configured store (and keep compat map updated)
    let transcript = json!({"events": events.iter().map(|(t, d)| json!({"type": t, "data": d})).collect::<Vec<_>>(),});
    let _ = crate::services::artifacts::put(state, transcript);

    // Build SSE text
    let sse = [
        "event: started".to_string(),
        format!("data: {}", started),
        "".to_string(),
        "event: token".to_string(),
        format!("data: {}", token0),
        "".to_string(),
        "event: metrics".to_string(),
        format!("data: {}", metrics_ev),
        "".to_string(),
        "event: end".to_string(),
        format!("data: {}", end),
        "".to_string(),
    ].join("\n");
    sse
}
