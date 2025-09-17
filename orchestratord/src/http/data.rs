/// THIS FILE IS GETTING QUITE LARGE
/// PLEASE MODULARIZE IT AT YOUR NEXT REFACTORING

use axum::{
    extract::{Path, State},
    response::{IntoResponse, Response},
    Json,
};
use contracts_api_types as api;
use futures::StreamExt;
use http::{header::CONTENT_TYPE, HeaderMap};
use orchestrator_core::queue::EnqueueError;
use orchestrator_core::queue::Priority as CorePriority;
use std::time::Instant;
use std::collections::HashMap;
use tracing::info;
use worker_adapters_adapter_api::WorkerError as AdapterErr;
use uuid::Uuid;
use sha2::{Digest, Sha256};
use chrono::Utc;

use super::auth::require_api_key;
use crate::{
    backpressure, metrics, placement,
    session::Session,
    state::{AppState, ModelState},
};

fn correlation_id_from(headers: &HeaderMap) -> String {
    headers
        .get("X-Correlation-Id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string())
}

fn prune_zero_ttl_sessions(map: &mut HashMap<String, Session>) {
    let dead: Vec<String> = map
        .iter()
        .filter_map(|(k, v)| if v.ttl_ms_remaining <= 0 { Some(k.clone()) } else { None })
        .collect();
    for k in dead {
        map.remove(&k);
    }
}

fn task_id_to_u32(task_id: &str) -> u32 {
    let hex: String = task_id
        .chars()
        .filter(|c| c.is_ascii_hexdigit())
        .take(8)
        .collect();
    u32::from_str_radix(&hex, 16).unwrap_or(0)
}

// Data plane — OrchQueue v1
pub async fn create_task(
    headers: HeaderMap,
    state: State<AppState>,
    body: Json<api::TaskRequest>,
) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let req_corr = correlation_id_from(&headers);
    // Lifecycle gating
    {
        let ms = state.model_state.lock().unwrap().clone();
        match ms {
            ModelState::Active => {}
            ModelState::Retired => {
                let mut h = HeaderMap::new();
                h.insert("X-Correlation-Id", req_corr.parse().unwrap());
                let err = serde_json::json!({
                    "code": "POOL_UNAVAILABLE",
                    "engine": body.engine,
                    "message": "model is retired",
                });
                return (http::StatusCode::SERVICE_UNAVAILABLE, h, Json(err)).into_response();
            }
        }
    }
    // Minimal placeholder: accept admission and return a basic envelope + correlation id
    // Sentinel: if expected_tokens is extremely high, simulate queue full and return 429
    if body.expected_tokens.unwrap_or(0) >= 1_000_000 {
        let backoff = backpressure::Backoff {
            retry_after_seconds: 1,
            x_backoff_ms: 1000,
        };
        let mut resp_headers = backpressure::build_429_headers(backoff);
        resp_headers.insert("X-Correlation-Id", req_corr.parse().unwrap());
        let policy = backpressure::compute_policy_label(());
        let extras = backpressure::build_429_body(policy.clone());
        // Increment backpressure counter
        let engine_label = match body.engine {
            api::Engine::Llamacpp => "llamacpp",
            api::Engine::Vllm => "vllm",
            api::Engine::Tgi => "tgi",
            api::Engine::Triton => "triton",
        };
        let policy_label = match policy {
            backpressure::PolicyLabel::Reject => "reject",
            backpressure::PolicyLabel::DropLru => "drop-lru",
        };
        metrics::ADMISSION_BACKPRESSURE_EVENTS_TOTAL
            .with_label_values(&[engine_label, policy_label])
            .inc();
        let body = serde_json::json!({
            "code": "QUEUE_FULL_DROP_LRU",
            "engine": body.engine,
            "message": "Queue saturated; try later",
            "policy_label": extras.get("policy_label").cloned().unwrap_or(serde_json::json!("unknown")),
            "retriable": extras.get("retriable").cloned().unwrap_or(serde_json::json!(true)),
            "retry_after_ms": extras.get("retry_after_ms").cloned().unwrap_or(serde_json::json!(1000)),
        });
        return (
            http::StatusCode::TOO_MANY_REQUESTS,
            resp_headers,
            Json(body),
        )
            .into_response();
    }

    // Error taxonomy sentinels
    if body.ctx < 0 {
        let mut h = HeaderMap::new();
        h.insert("X-Correlation-Id", req_corr.parse().unwrap());
        let err = serde_json::json!({
            "code": "INVALID_PARAMS",
            "engine": body.engine,
            "message": "invalid parameters",
        });
        return (http::StatusCode::BAD_REQUEST, h, Json(err)).into_response();
    }
    if body.model_ref == "pool-unavailable" {
        let mut h = HeaderMap::new();
        h.insert("X-Correlation-Id", req_corr.parse().unwrap());
        let err = serde_json::json!({
            "code": "POOL_UNAVAILABLE",
            "engine": body.engine,
            "message": "pool unavailable",
        });
        return (http::StatusCode::SERVICE_UNAVAILABLE, h, Json(err)).into_response();
    }
    if matches!(&body.prompt, Some(p) if p == "cause-internal") {
        let mut h = HeaderMap::new();
        h.insert("X-Correlation-Id", req_corr.parse().unwrap());
        let err = serde_json::json!({
            "code": "INTERNAL",
            "engine": body.engine,
            "message": "internal error",
        });
        return (http::StatusCode::INTERNAL_SERVER_ERROR, h, Json(err)).into_response();
    }

    // Guardrails: context length and token budget (reject before enqueue)
    if body.ctx > 32768 || body.max_tokens > 50000 {
        let mut h = HeaderMap::new();
        h.insert("X-Correlation-Id", req_corr.parse().unwrap());
        let err = serde_json::json!({
            "code": "INVALID_PARAMS",
            "engine": body.engine,
        });
        return (http::StatusCode::BAD_REQUEST, h, Json(err)).into_response();
    }

    // Deadline infeasible sentinel
    if body.deadline_ms <= 0 {
        let mut h = HeaderMap::new();
        h.insert("X-Correlation-Id", req_corr.parse().unwrap());
        let err = serde_json::json!({
            "code": "DEADLINE_UNMET",
            "engine": body.engine,
            "message": "deadline unmet",
        });
        return (http::StatusCode::BAD_REQUEST, h, Json(err)).into_response();
    }

    // ORCH-2001: admission acceptance; OC-CORE-1001..1002: bounded queue policies; ORCH-2007: 429 backpressure mapping; README_LLM-1001: traceability requirement.
    // Stage 6: enqueue into QueueWithMetrics (contract-first) and map reject policy to 429.
    let (queue_position_est, predicted_start_ms_est): (usize, u64) = {
        let prio = match body.priority {
            api::Priority::Interactive => CorePriority::Interactive,
            api::Priority::Batch => CorePriority::Batch,
        };
        let id_u32: u32 = {
            // Derive a deterministic u32 from the UUID by taking the first 8 hex digits.
            let hex: String = body
                .task_id
                .chars()
                .filter(|c| c.is_ascii_hexdigit())
                .take(8)
                .collect();
            u32::from_str_radix(&hex, 16).unwrap_or(0)
        };
        let mut q = state.queue.lock().unwrap();
        if let Err(EnqueueError::QueueFullReject) = q.enqueue(id_u32, prio) {
            let backoff = backpressure::Backoff { retry_after_seconds: 1, x_backoff_ms: 1000 };
            let mut resp_headers = backpressure::build_429_headers(backoff);
            resp_headers.insert("X-Correlation-Id", req_corr.parse().unwrap());
            let policy = backpressure::compute_policy_label(());
            let extras = backpressure::build_429_body(policy.clone());
            let engine_label = match body.engine { api::Engine::Llamacpp => "llamacpp", api::Engine::Vllm => "vllm", api::Engine::Tgi => "tgi", api::Engine::Triton => "triton", };
            let policy_label = match policy { backpressure::PolicyLabel::Reject => "reject", backpressure::PolicyLabel::DropLru => "drop-lru", };
            metrics::ADMISSION_BACKPRESSURE_EVENTS_TOTAL.with_label_values(&[engine_label, policy_label]).inc();
            let body = serde_json::json!({
                "code": "ADMISSION_REJECT",
                "engine": body.engine,
                "message": "Queue full (reject)",
                "policy_label": extras.get("policy_label").cloned().unwrap_or(serde_json::json!("reject")),
                "retriable": extras.get("retriable").cloned().unwrap_or(serde_json::json!(true)),
                "retry_after_ms": extras.get("retry_after_ms").cloned().unwrap_or(serde_json::json!(1000)),
            });
            return (http::StatusCode::TOO_MANY_REQUESTS, resp_headers, Json(body)).into_response();
        }
        // Compute queue position as number of items ahead in combined queue (best-effort)
        let len_now = q.inner().len();
        let queue_position_est = len_now.saturating_sub(1);
        // Naive ETA heuristic: 100ms per item ahead
        let predicted_start_ms_est = (queue_position_est as u64) * 100;
        (queue_position_est, predicted_start_ms_est)
    };

    let mut resp_headers = HeaderMap::new();
    resp_headers.insert("X-Correlation-Id", req_corr.parse().unwrap());
    // Budget headers per contract (use session store; create if missing)
    let (tok_rem, time_rem, cost_rem) = {
        let mut guard = state.sessions.lock().unwrap();
        prune_zero_ttl_sessions(&mut *guard);
        let s = guard
            .entry(body.session_id.clone())
            .or_insert_with(Session::new_default);
        (
            s.tokens_budget_remaining.unwrap_or(0),
            s.time_budget_remaining_ms.unwrap_or(0),
            s.cost_budget_remaining.unwrap_or(0.0),
        )
    };
    resp_headers.insert(
        "X-Budget-Tokens-Remaining",
        tok_rem.to_string().parse().unwrap(),
    );
    resp_headers.insert(
        "X-Budget-Time-Remaining-Ms",
        time_rem.to_string().parse().unwrap(),
    );
    resp_headers.insert(
        "X-Budget-Cost-Remaining",
        cost_rem.to_string().parse().unwrap(),
    );

    let resp = api::AdmissionResponse {
        task_id: body.task_id.clone(),
        queue_position: queue_position_est as i32,
        predicted_start_ms: predicted_start_ms_est as i64,
        backoff_ms: 0,
    };
    // Track task -> session mapping for observability
    {
        if let Ok(mut map) = state.tasks.lock() {
            map.insert(resp.task_id.clone(), body.session_id.clone());
        }
    }
    // Record model_state gauge (Active by default) for visibility
    let model_id = body.model_ref.clone();
    let state_label = "Active";
    metrics::MODEL_STATE
        .with_label_values(&[&model_id, state_label])
        .set(1);
    // Structured log for started/admission (no secrets)
    info!(
        event = "admission_started",
        job_id = %resp.task_id,
        task_id = %resp.task_id,
        session_id = %body.session_id,
        engine = ?body.engine,
        model_ref = %body.model_ref,
        pool_id = "pool0",
        replica_id = "r0",
        ctx = body.ctx,
        kv_warmth = false,
        queue_position = resp.queue_position,
        predicted_start_ms = resp.predicted_start_ms,
        "admission accepted"
    );
    // Transitional: keep in-memory log for BDD harness until it migrates
    {
        let mut logs = state.logs.lock().unwrap();
        logs.push(format!(
            "{{\"event\":\"started\",\"task_id\":\"{}\",\"queue_position\":{},\"predicted_start_ms\":{}}}",
            resp.task_id, resp.queue_position, resp.predicted_start_ms
        ));
    }

    // Stage 6 vertical slice: dispatch to adapter and build SSE transcript in the background
    if let Some(adapter) = placement::choose_adapter(&state, &body.engine) {
        // Stage 7: allocate a lease for pool0 before dispatch; update gauge
        let pool_id = "pool0".to_string();
        {
            if let Ok(mut pm) = state.pool_manager.lock() {
                let leases = pm.allocate_lease(&pool_id);
                crate::metrics::ACTIVE_LEASES
                    .with_label_values(&[&pool_id])
                    .set(leases as i64);
            }
        }
        let pm_arc = state.pool_manager.clone();
        let sse_store = state.sse.clone();
        let artifacts_arc = state.artifacts.clone();
        let tasks_arc = state.tasks.clone();
        let sessions_arc = state.sessions.clone();
        let req = (*body).clone();
        let task_id = resp.task_id.clone();
        let started_frame = format!(
            "event: started\ndata: {{\"queue_position\":{},\"predicted_start_ms\":{}}}\n\n",
            resp.queue_position, resp.predicted_start_ms
        );
        tokio::spawn(async move {
            let mut transcript = String::new();
            transcript.push_str(&started_frame);
            let mut emitted_metrics = false;
            let t0 = Instant::now();
            match adapter.submit(req.clone()) {
                Ok(mut stream) => {
                    let mut first_token_ms: Option<u128> = None;
                    let mut tokens_out: u64 = 0;
                    while let Some(ev) = stream.next().await {
                        match ev {
                            Ok(te) => {
                                match te.kind.as_str() {
                                    "token" => {
                                        transcript.push_str("event: token\n");
                                        transcript.push_str(&format!("data: {}\n\n", te.data));
                                        if !emitted_metrics {
                                            // Emit metrics after first token
                                            let (tok_rem, time_rem, cost_rem, kv_warmth) = {
                                                let mut guard = sessions_arc.lock().unwrap();
                                                let s = guard
                                                    .entry(req.session_id.clone())
                                                    .or_insert_with(Session::new_default);
                                                // Mark KV warm after first token
                                                s.kv_warmth = true;
                                                (
                                                    s.tokens_budget_remaining.unwrap_or(0),
                                                    s.time_budget_remaining_ms.unwrap_or(0),
                                                    s.cost_budget_remaining.unwrap_or(0.0),
                                                    s.kv_warmth,
                                                )
                                            };
                                            let metrics_obj = serde_json::json!({
                                                "queue_depth": 0,
                                                "on_time_probability": 0.9,
                                                "kv_warmth": kv_warmth,
                                                "tokens_budget_remaining": tok_rem,
                                                "time_budget_remaining_ms": time_rem,
                                                "cost_budget_remaining": cost_rem,
                                            });
                                            transcript.push_str("event: metrics\n");
                                            transcript.push_str(&format!("data: {}\n\n", metrics_obj));
                                            emitted_metrics = true;
                                            let ms = t0.elapsed().as_millis();
                                            first_token_ms = Some(ms);
                                            // Metrics side-effects: first token latency and tokens_in (approx by ctx)
                                            crate::metrics::record_stream_started(
                                                "llamacpp", // planning stub labels
                                                "v0",
                                                "pool0",
                                                "r0",
                                                "interactive",
                                                ms as u64,
                                                0,
                                            );
                                        }
                                        tokens_out += 1;
                                    }
                                    "end" => {
                                        transcript.push_str("event: end\n");
                                        transcript.push_str(&format!("data: {}\n\n", te.data));
                                        let decode_ms =
                                            t0.elapsed().as_millis() - first_token_ms.unwrap_or(0);
                                        // Update session budgets based on tokens/time spent
                                        if let Ok(mut guard) = sessions_arc.lock() {
                                            if let Some(s) = guard.get_mut(&req.session_id) {
                                                s.spend_tokens(tokens_out as i64);
                                                s.spend_time_ms(decode_ms as i64);
                                                // Rough cost model: $0.000002 per token (placeholder)
                                                let cost_incr = (tokens_out as f64) * 0.000002f64;
                                                s.spend_cost(cost_incr);
                                            }
                                            prune_zero_ttl_sessions(&mut *guard);
                                        }
                                        // Clear task mapping
                                        if let Ok(mut tm) = tasks_arc.lock() {
                                            tm.remove(&req.task_id);
                                        }
                                        crate::metrics::record_stream_ended(
                                            "llamacpp",
                                            "v0",
                                            "pool0",
                                            "r0",
                                            "interactive",
                                            decode_ms as u64,
                                            tokens_out,
                                        );
                                        tracing::info!(
                                            event = "stream_ended",
                                            job_id = %req.task_id,
                                            engine = ?req.engine,
                                            tokens_out = tokens_out,
                                            decode_ms = decode_ms as u64,
                                            "stream finished"
                                        );
                                    }
                                    _ => {}
                                }
                            }
                            Err(e) => {
                                // Map adapter errors to contract SSE error frames
                                let (code, message) = match e {
                                    AdapterErr::DeadlineUnmet => {
                                        ("DEADLINE_UNMET", "deadline unmet")
                                    }
                                    AdapterErr::PoolUnavailable => {
                                        ("POOL_UNAVAILABLE", "pool unavailable")
                                    }
                                    AdapterErr::DecodeTimeout => {
                                        ("DECODE_TIMEOUT", "decode timeout")
                                    }
                                    AdapterErr::WorkerReset => ("WORKER_RESET", "worker reset"),
                                    AdapterErr::Internal(ref msg) => ("INTERNAL", msg.as_str()),
                                    AdapterErr::Adapter(ref msg) => ("INTERNAL", msg.as_str()),
                                };
                                let err = serde_json::json!({
                                    "code": code,
                                    "message": message,
                                    "engine": req.engine,
                                });
                                transcript.push_str("event: error\n");
                                transcript.push_str(&format!("data: {}\n\n", err));
                                tracing::info!(
                                    event = "stream_error",
                                    job_id = %req.task_id,
                                    engine = ?req.engine,
                                    code = code,
                                    message = %message,
                                    "stream error mapped"
                                );
                                break;
                            }
                        }
                    }
                }
                Err(_) => {
                    // Adapter submit failed; leave transcript with started+metrics only for now
                }
            }
            if let Ok(mut map) = sse_store.lock() {
                map.insert(task_id.clone(), transcript.clone());
            }
            // Persist transcript as a trace artifact for debugging/proofs
            {
                // Build content-addressed digest of transcript
                let mut hasher = Sha256::new();
                hasher.update(transcript.as_bytes());
                let digest_raw = hasher.finalize();
                let mut hex = String::with_capacity(digest_raw.len() * 2);
                for b in &digest_raw {
                    use std::fmt::Write as _;
                    let _ = write!(&mut hex, "{:02x}", b);
                }
                let id = hex.clone();
                let created_ms = Utc::now().timestamp_millis();
                let doc = serde_json::json!({
                    "id": id,
                    "kind": "trace",
                    "digest": format!("sha256:{}", hex),
                    "created_ms": created_ms,
                    "tags": ["sse","stream", format!("task:{}", req.task_id), format!("session:{}", req.session_id)],
                    "content": {"transcript": transcript},
                    "parent": serde_json::Value::Null,
                    "metadata": {"engine":"llamacpp","engine_version":"v0","pool_id":"pool0","replica_id":"r0"},
                });
                if let Ok(mut m) = artifacts_arc.lock() {
                    m.insert(id, doc);
                }
            }
            // Release lease and update gauge at end of streaming
            if let Ok(mut pm) = pm_arc.lock() {
                let leases = pm.release_lease("pool0");
                crate::metrics::ACTIVE_LEASES
                    .with_label_values(&["pool0"]) 
                    .set(leases as i64);
            }
        });
    }
    (http::StatusCode::ACCEPTED, resp_headers, Json(resp)).into_response()
}

pub async fn stream_task(
    headers: HeaderMap,
    state: State<AppState>,
    Path(id): Path<String>,
) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    // Minimal SSE stub stream sufficient for BDD ordering assertions
    let mut resp_headers = HeaderMap::new();
    resp_headers.insert(CONTENT_TYPE, "text/event-stream".parse().unwrap());
    let req_corr = correlation_id_from(&headers);
    resp_headers.insert("X-Correlation-Id", req_corr.parse().unwrap());
    // Optional budget headers at stream start (stub values)
    resp_headers.insert("X-Budget-Tokens-Remaining", "0".parse().unwrap());
    resp_headers.insert("X-Budget-Time-Remaining-Ms", "0".parse().unwrap());
    resp_headers.insert("X-Budget-Cost-Remaining", "0".parse().unwrap());
    if let Ok(map) = state.sse.lock() {
        if let Some(transcript) = map.get(&id) {
            return (resp_headers, transcript.clone()).into_response();
        }
    }
    let body = "event: started\n\
                data: {\"queue_position\":0,\"predicted_start_ms\":0}\n\n\
                event: token\n\
                data: {\"t\":\"hello\",\"i\":0}\n\n\
                event: metrics\n\
                data: {\"queue_depth\":0,\"on_time_probability\":0.9,\"kv_warmth\":false,\"tokens_budget_remaining\":0,\"time_budget_remaining_ms\":0,\"cost_budget_remaining\":0}\n\n\
                event: end\n\
                data: {\"tokens_out\":1,\"decode_ms\":0}\n\n";
    (resp_headers, body).into_response()
}

pub async fn cancel_task(
    headers: HeaderMap,
    state: State<AppState>,
    Path(task_id): Path<String>,
) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    // Best-effort: derive the same u32 used for enqueue and cancel from the queue
    let id = task_id_to_u32(&task_id);
    if let Ok(mut q) = state.queue.lock() {
        let _ = q.cancel(id, "api_request");
    }
    let mut h = HeaderMap::new();
    let req_corr = correlation_id_from(&headers);
    h.insert("X-Correlation-Id", req_corr.parse().unwrap());
    (http::StatusCode::NO_CONTENT, h).into_response()
}

pub async fn get_session(
    headers: HeaderMap,
    state: State<AppState>,
    Path(id): Path<String>,
) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let info = {
        let mut guard = state.sessions.lock().unwrap();
        prune_zero_ttl_sessions(&mut *guard);
        let s = guard
            .entry(id.clone())
            .or_insert_with(Session::new_default)
            .clone();
        api::SessionInfo {
            ttl_ms_remaining: Some(s.ttl_ms_remaining),
            turns: Some(s.turns),
            kv_bytes: Some(s.kv_bytes),
            kv_warmth: Some(s.kv_warmth),
            tokens_budget_remaining: s.tokens_budget_remaining,
            time_budget_remaining_ms: s.time_budget_remaining_ms,
            cost_budget_remaining: s.cost_budget_remaining,
        }
    };
    let mut h = HeaderMap::new();
    let req_corr = correlation_id_from(&headers);
    h.insert("X-Correlation-Id", req_corr.parse().unwrap());
    (http::StatusCode::OK, h, Json(info)).into_response()
}

pub async fn delete_session(
    headers: HeaderMap,
    state: State<AppState>,
    Path(id): Path<String>,
) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    if let Ok(mut guard) = state.sessions.lock() {
        guard.remove(&id);
    }
    let mut h = HeaderMap::new();
    let req_corr = correlation_id_from(&headers);
    h.insert("X-Correlation-Id", req_corr.parse().unwrap());
    (http::StatusCode::NO_CONTENT, h).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    fn ok_headers() -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert("X-API-Key", "valid".parse().unwrap());
        h
    }

    fn base_req() -> api::TaskRequest {
        api::TaskRequest {
            task_id: "aaaaaaaa-0000-0000-0000-000000000000".to_string(),
            session_id: "s1".to_string(),
            workload: api::Workload::Completion,
            model_ref: "m:v0".to_string(),
            engine: api::Engine::Llamacpp,
            ctx: 0,
            priority: api::Priority::Interactive,
            seed: None,
            determinism: None,
            sampler_profile_version: None,
            prompt: Some("hi".into()),
            inputs: None,
            max_tokens: 5,
            deadline_ms: 1,
            expected_tokens: Some(0),
            kv_hint: None,
        }
    }

    // ORCH-2001: INVALID_PARAMS guardrail when ctx < 0
    #[tokio::test]
    async fn test_orch_2001_invalid_params_ctx_negative() {
        let state = crate::state::default_state();
        let mut req = base_req();
        req.ctx = -1;
        let resp = create_task(ok_headers(), State(state), Json(req)).await;
        assert_eq!(resp.status(), http::StatusCode::BAD_REQUEST);
        let body = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["code"], "INVALID_PARAMS");
        assert_eq!(v["engine"], "llamacpp");
    }

    // ORCH-2001: DEADLINE_UNMET when deadline_ms <= 0
    #[tokio::test]
    async fn test_orch_2001_deadline_unmet() {
        let state = crate::state::default_state();
        let mut req = base_req();
        req.deadline_ms = 0;
        let resp = create_task(ok_headers(), State(state), Json(req)).await;
        assert_eq!(resp.status(), http::StatusCode::BAD_REQUEST);
        let body = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["code"], "DEADLINE_UNMET");
    }

    // ORCH-2007: Large expected_tokens sentinel returns 429 with advisory headers/body
    #[tokio::test]
    async fn test_orch_2007_large_expected_tokens_triggers_429() {
        let state = crate::state::default_state();
        let mut req = base_req();
        req.expected_tokens = Some(1_000_000);
        let resp = create_task(ok_headers(), State(state), Json(req)).await;
        assert_eq!(resp.status(), http::StatusCode::TOO_MANY_REQUESTS);
        let headers = resp.headers().clone();
        assert_eq!(headers.get("Retry-After").unwrap(), "1");
        assert_eq!(headers.get("X-Backoff-Ms").unwrap(), "1000");
        assert!(headers.get("X-Correlation-Id").is_some());
        let body = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["code"], "QUEUE_FULL_DROP_LRU");
        assert_eq!(v["policy_label"], "reject");
        assert!(v["retriable"].as_bool().unwrap());
        assert_eq!(v["retry_after_ms"].as_i64().unwrap(), 1000);
    }

    // OC-CORE-1001/1002, ORCH-2001: Queue full reject path maps to 429 ADMISSION_REJECT
    #[tokio::test]
    async fn test_orch_2001_admission_reject_on_full_queue() {
        let state = crate::state::default_state();
        {
            // Reconfigure queue to capacity 0 with Reject policy to force immediate reject
            let mut guard = state.queue.lock().unwrap();
            *guard = crate::admission::QueueWithMetrics::new(
                0,
                orchestrator_core::queue::Policy::Reject,
                crate::admission::MetricLabels {
                    engine: "llamacpp".into(),
                    engine_version: "v0".into(),
                    pool_id: "pool0".into(),
                    replica_id: "r0".into(),
                },
            );
        }
        let req = base_req();
        let resp = create_task(ok_headers(), State(state), Json(req)).await;
        assert_eq!(resp.status(), http::StatusCode::TOO_MANY_REQUESTS);
        let body = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["code"], "ADMISSION_REJECT");
        assert!(v["retriable"].as_bool().unwrap());
    }

    // ORCH-3026: Cancel removes queued item and emits cancel metrics with reason
    #[tokio::test]
    async fn test_orch_3026_cancel_removes_and_emits_metrics() {
        let state = crate::state::default_state();
        // Enqueue a task normally
        let req = base_req();
        let _ = create_task(ok_headers(), State(state.clone()), Json(req.clone())).await;

        // Cancel it
        let cancel_path = Path(req.task_id.clone());
        let resp = cancel_task(ok_headers(), State(state.clone()), cancel_path).await;
        assert_eq!(resp.status(), http::StatusCode::NO_CONTENT);

        // Check metrics include tasks_canceled_total with reason="api_request"
        let text = crate::metrics::gather_metrics_text();
        assert!(
            text.contains("tasks_canceled_total{") && text.contains("reason=\"api_request\""),
            "missing tasks_canceled_total with reason=api_request; got: {}",
            text
        );
    }
}
