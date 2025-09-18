//! Streaming service

use serde_json::json;
use crate::state::AppState;
use tokio::time::{sleep, Duration};

/// Render a simple deterministic SSE for a task id and persist transcript to artifacts.
pub async fn render_sse_for_task(state: &AppState, id: String) -> String {
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
    let token1 = json!({"t": " world", "i": 1});
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

    // Build events with potential early termination on cancel
    let mut events: Vec<(&str, serde_json::Value)> = vec![("started", started.clone())];

    let is_canceled = |st: &AppState, id: &str| -> bool {
        if let Ok(guard) = st.cancellations.lock() { guard.contains(id) } else { false }
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

    // Persist transcript as artifact via configured store (and keep compat map updated)
    let transcript = json!({"events": events.iter().map(|(t, d)| json!({"type": t, "data": d})).collect::<Vec<_>>(),});
    let _ = crate::services::artifacts::put(state, transcript);
    // Clear cancellation flag for this task id to avoid leakage
    if let Ok(mut guard) = state.cancellations.lock() { let _ = guard.remove(&id); }

    // Build SSE text from events vector
    let mut lines = Vec::with_capacity(events.len() * 3);
    for (ty, data) in &events {
        lines.push(format!("event: {}", ty));
        lines.push(format!("data: {}", data));
        lines.push("".to_string());
    }
    // Ensure terminal newline
    let sse = lines.join("\n");
    sse
}
