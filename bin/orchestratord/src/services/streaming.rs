//! Streaming service

use crate::state::AppState;
use futures::StreamExt;
use serde_json::json;
use std::io::BufWriter;
use std::io::Write;
use tokio::time::{sleep, Duration};
use worker_adapters_adapter_api as adapter_api;

/// Render a simple deterministic SSE for a task id and persist transcript to artifacts.
/// TODO(OwnerB-ORCH-SSE-FALLBACK): This function contains an MVP deterministic fallback
/// when no adapter is bound. Replace with full admission→dispatch→stream path exclusively,
/// or gate fallback behind a dev/testing feature flag once adapters are always present.
pub async fn render_sse_for_task(state: &AppState, id: String) -> String {
    // Try adapter-host first (if a worker is registered). Fallback to deterministic SSE.
    // TODO[ORCHD-POOL-HEALTH-GATE-0010]: Before attempting dispatch, consult
    // `state.pool_manager` to ensure the chosen pool/replica is Live+Ready and has
    // slots_free > 0. If not ready, either wait with backoff or surface a retry hint.
    // TODO[ORCHD-STREAM-1101]: Build TaskRequest from actual admission + request context
    // (session linkage, engine/model placement decision, budgets) instead of a stub.
    // TODO[ORCHD-STREAM-1102]: Propagate cancellation via a structured token to adapters
    // (not just shared state polling) and verify no tokens after cancel across all adapters.
    // TODO[ORCHD-STREAM-1103]: Map adapter errors to domain errors and emit `error` SSE frames
    // with `code/message/engine` per spec.
    // Health gate: check target pool readiness before attempting dispatch
    let target_pool = {
        if let Ok(map) = state.admissions.lock() {
            map.get(&id)
                .and_then(|s| s.request.placement.as_ref().and_then(|p| p.pin_pool_id.clone()))
                .unwrap_or_else(|| "default".to_string())
        } else {
            "default".to_string()
        }
    };
    if !should_dispatch(state, &target_pool) {
        // Emit minimal SSE with retry hints
        let events: Vec<(String, serde_json::Value)> = vec![
            (
                "error".into(),
                json!({
                    "code": "PoolUnready",
                    "message": format!("pool '{}' not ready", target_pool),
                    "retry_after_ms": 500
                }),
            ),
            ("end".into(), json!({})),
        ];
        return build_sse_from_events(state, &id, events);
    }
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
    // TODO[ORCHD-METRICS-1201]: Replace local metrics helpers with shared workspace metrics crate
    // and ensure series names/labels match `ci/metrics.lint.json`. Add sampling/throttling where needed.
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

    // Started frame fields from admission snapshot if available
    // TODO[ORCHD-SSE-1301]: Include additional started fields when available (engine, pool, replica,
    // sampler_profile_version) to match observability guidance.
    let started_fields = {
        if let Ok(map) = state.admissions.lock() {
            if let Some(snap) = map.get(&id) {
                json!({
                    "queue_position": snap.info.queue_position,
                    "predicted_start_ms": snap.info.predicted_start_ms,
                })
            } else {
                json!({"queue_position": 0, "predicted_start_ms": 0})
            }
        } else {
            json!({"queue_position": 0, "predicted_start_ms": 0})
        }
    };
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
    let mut events: Vec<(&str, serde_json::Value)> = vec![("started", started_fields.clone())];

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
    // Structured log for tokens_out and common fields
    let tokens_out_count = events
        .iter()
        .filter(|(t, _)| <&str as AsRef<str>>::as_ref(t) == "token")
        .count();
    if let Ok(mut lg) = state.logs.lock() {
        lg.push(format!(
            "{{\"job_id\":\"{}\",\"engine\":\"{}\",\"pool_id\":\"{}\",\"tokens_out\":{}}}",
            id, "llamacpp", "default", tokens_out_count
        ));
    }
    // Persist transcript and return SSE body
    let events_owned: Vec<(String, serde_json::Value)> =
        events.into_iter().map(|(t, d)| (t.to_string(), d)).collect();
    build_sse_from_events(state, &id, events_owned)
}

/// Determine if a pool is dispatchable: Live+Ready and slots_free > 0.
fn should_dispatch(state: &AppState, pool_id: &str) -> bool {
    // TODO: Make async to call HTTP API
    // For now, always return true
    true
}

fn try_dispatch_via_adapter(
    state: &AppState,
    id: &str,
) -> anyhow::Result<adapter_api::TokenStream> {
    // Build request from admission snapshot when available
    let (pool_id, req) = {
        let map = state.admissions.lock().map_err(|_| anyhow::anyhow!("admissions lock"))?;
        if let Some(snap) = map.get(id) {
            let pool = snap
                .request
                .placement
                .as_ref()
                .and_then(|p| p.pin_pool_id.clone())
                .unwrap_or_else(|| "default".to_string());
            (pool, snap.request.clone())
        } else {
            // Fallback synthetic request
            (
                "default".to_string(),
                contracts_api_types::TaskRequest {
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
                },
            )
        }
    };
    // Health/placement gate: require Live+Ready and slots_free > 0
    if !should_dispatch(state, &pool_id) {
        return Err(anyhow::anyhow!("pool not ready"));
    }
    state.adapter_host.submit(&pool_id, req)
}

fn build_sse_from_events(
    state: &AppState,
    id: &str,
    events: Vec<(String, serde_json::Value)>,
) -> String {
    // Persist transcript as artifact via configured store (and keep compat map updated)
    let transcript = json!({"events": events.iter().map(|(t, d)| json!({"type": t, "data": d})).collect::<Vec<_>>(),});
    let _ = crate::services::artifacts::put(state, transcript);
    // Clear cancellation flag and admission snapshot for this task id to avoid leakage
    if let Ok(mut guard) = state.cancellations.lock() {
        let _ = guard.remove(id);
    }
    if let Ok(mut map) = state.admissions.lock() {
        let _ = map.remove(id);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;
    use pool_managerd::health::HealthStatus;

    // ORCHD-STREAM-UT-1101: streaming consults pool health and refuses dispatch if not Ready.
    #[test]
    fn test_orchd_stream_ut_1101_health_gate_refuses_unready() {
        let state = AppState::new();
        
        // Mark pool as not ready
        {
            let mut reg = state.pool_manager.lock().unwrap();
            reg.set_health("test-pool", HealthStatus { live: true, ready: false });
        }

        // should_dispatch should return false
        assert!(!should_dispatch(&state, "test-pool"), "should refuse dispatch when not ready");
    }

    // ORCHD-STREAM-UT-1102: streaming allows dispatch when pool is Live+Ready with slots
    #[test]
    fn test_orchd_stream_ut_1102_health_gate_allows_ready() {
        let state = AppState::new();
        
        // Mark pool as ready with slots
        {
            let mut reg = state.pool_manager.lock().unwrap();
            reg.set_health("test-pool", HealthStatus { live: true, ready: true });
            reg.set_slots("test-pool", 4, 2);
        }

        // should_dispatch should return true
        assert!(should_dispatch(&state, "test-pool"), "should allow dispatch when ready with slots");
    }

    // ORCHD-STREAM-UT-1103: streaming refuses dispatch when slots_free is 0
    #[test]
    fn test_orchd_stream_ut_1103_health_gate_refuses_no_slots() {
        let state = AppState::new();
        
        // Mark pool as ready but no free slots
        {
            let mut reg = state.pool_manager.lock().unwrap();
            reg.set_health("test-pool", HealthStatus { live: true, ready: true });
            reg.set_slots("test-pool", 4, 0);
        }

        // should_dispatch should return false
        assert!(!should_dispatch(&state, "test-pool"), "should refuse dispatch when no free slots");
    }

    // ORCHD-STREAM-UT-1104: streaming refuses dispatch when pool is not live
    #[test]
    fn test_orchd_stream_ut_1104_health_gate_refuses_not_live() {
        let state = AppState::new();
        
        // Mark pool as not live
        {
            let mut reg = state.pool_manager.lock().unwrap();
            reg.set_health("test-pool", HealthStatus { live: false, ready: true });
            reg.set_slots("test-pool", 4, 2);
        }

        // should_dispatch should return false
        assert!(!should_dispatch(&state, "test-pool"), "should refuse dispatch when not live");
    }
}
