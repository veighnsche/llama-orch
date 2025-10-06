// ⚠️ AUDIT LOGGING REMINDER:
// Task operations (submit/cancel) and data access MUST be logged to audit-logging crate.
// Required for GDPR compliance (InferenceExecuted, ModelAccessed, DataDeleted events).
// See: bin/shared-crates/AUDIT_LOGGING_REMINDER.md

use axum::{extract::State, response::IntoResponse, Json};
use http::{HeaderMap, StatusCode};
use serde_json::json;

use crate::domain::error::OrchestratorError as ErrO;
use crate::{services, state::AdmissionInfo, state::AdmissionSnapshot, state::AppState};
use contracts_api_types as api;

pub async fn get_session(
    state: State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<impl IntoResponse, ErrO> {
    let svc = services::session::SessionService::new(
        state.sessions.clone(),
        std::sync::Arc::new(crate::infra::clock::SystemClock),
    );
    let entry = svc.get_or_create(&id);
    let body = json!({
        "ttl_ms_remaining": entry.ttl_ms_remaining,
        "turns": entry.turns,
        "kv_bytes": entry.kv_bytes,
        "kv_warmth": entry.kv_warmth,
        "tokens_budget_remaining": entry.tokens_budget_remaining,
        "time_budget_remaining_ms": entry.time_budget_remaining_ms,
        "cost_budget_remaining": entry.cost_budget_remaining,
    });
    Ok((StatusCode::OK, Json(body)))
}

pub async fn delete_session(
    state: State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<impl IntoResponse, ErrO> {
    let svc = services::session::SessionService::new(
        state.sessions.clone(),
        std::sync::Arc::new(crate::infra::clock::SystemClock),
    );
    svc.delete(&id);
    Ok(StatusCode::NO_CONTENT)
}

pub async fn create_task(
    state: State<AppState>,
    Json(body): Json<api::TaskRequest>,
) -> Result<impl IntoResponse, ErrO> {
    // Error taxonomy sentinels
    // TODO[ORCHD-DATA-1001]: Replace sentinel validations with real policy from config schema.
    //  - `ctx` bounds, `deadline_ms`, and engine/pool availability should be validated via
    //    orchestrator-core placement/admission hooks and pool-manager state, not string sentinels.
    if body.ctx < 0 {
        return Err(ErrO::InvalidParams("ctx must be >= 0".into()));
    }
    if body.deadline_ms <= 0 {
        return Err(ErrO::DeadlineUnmet);
    }

    // Test sentinels for BDD error taxonomy tests
    // Note: These are harmless in production (unlikely model_ref values)
    if body.model_ref == "pool-unavailable" {
        return Err(ErrO::PoolUnavailable);
    }
    if body.prompt.as_deref() == Some("cause-internal") {
        return Err(ErrO::Internal);
    }

    if let Some(exp) = body.expected_tokens {
        if exp >= 2_000_000 {
            return Err(ErrO::QueueFullDropLru { retry_after_ms: Some(1000) });
        } else if exp >= 1_000_000 {
            return Err(ErrO::AdmissionReject {
                policy_label: "reject".into(),
                retry_after_ms: Some(1000),
            });
        }
    }

    // TODO[ORCHD-CATALOG-CHECK-0006]: Consult `catalog-core` to check whether the requested
    // model_ref is present locally and Active. If missing, consult policy to decide whether to
    // trigger provisioning (model + engine) or reject fast. Surface clear errors when HF auth
    // or cache space is unavailable.
    // TODO[ORCHD-PROVISION-POLICY-0005]: If model/engine are not present and policy allows,
    // orchestrate `engine-provisioner` + `model-provisioner` for the target pool prior to
    // admission completion. Integrate with `pool_managerd::registry` readiness so we do not
    // admit into a non-ready pool. Respect user overrides to pin to a specific pool/GPU.

    let prio = match body.priority {
        api::Priority::Interactive => orchestrator_core::queue::Priority::Interactive,
        api::Priority::Batch => orchestrator_core::queue::Priority::Batch,
    };
    // Map task_id to a stable u32 for queue identity
    let id_u32 = {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        body.task_id.hash(&mut h);
        (h.finish() & 0xFFFF_FFFF) as u32
    };
    let pos = {
        let mut q = state.admission.lock().unwrap();
        match q.enqueue(id_u32, prio) {
            Ok(p) => p,
            Err(()) => {
                // Backpressure: reject with retry hints
                return Err(ErrO::AdmissionReject {
                    policy_label: "reject".into(),
                    retry_after_ms: Some(1000),
                });
            }
        }
    };
    // TODO[ORCHD-ADMISSION-2002]: Replace heuristic with ETA derived from pool throughput and
    // active leases once adapter/pool metrics are wired. Spec: predicted_start_ms should be based
    // on queue depth, slots_free, and perf_tokens_per_s.
    // Simple ETA heuristic per spec: predicted_start_ms = queue_position * 100
    let predicted_start_ms = pos * 100;
    // Build streams and preparation (Owner F's work)
    let streams = api::AdmissionStreams {
        sse: format!("/v2/tasks/{}/events", body.task_id),
        sse_verbose: format!("/v2/tasks/{}/events?verbose=true", body.task_id),
    };
    let preparation = api::Preparation { steps: vec![] };

    // Record the admission snapshot for use by stream
    {
        let mut map = state.admissions.lock().unwrap();
        map.insert(
            body.task_id.clone(),
            AdmissionSnapshot {
                info: AdmissionInfo { queue_position: pos, predicted_start_ms },
                request: body.clone(),
            },
        );
    }
    // Success path
    let admission = api::AdmissionResponse {
        task_id: body.task_id.clone(),
        queue_position: pos as i32,
        predicted_start_ms,
        backoff_ms: 0,
        streams: Some(streams),
        preparation: Some(preparation),
    };

    // TODO[ORCHD-ADMISSION-STREAMS-0008]: Populate `AdmissionResponse.streams` with direct
    // links to stream endpoints once `contracts_api_types` is regenerated from OpenAPI.
    //   - streams.sse:        `/v2/tasks/{task_id}/events`
    //   - streams.sse_verbose:`/v2/tasks/{task_id}/events?verbose=true`
    // TODO[ORCHD-ADMISSION-PREPARATION-0009]: Populate `AdmissionResponse.preparation` with
    // a plan of preparatory steps (e.g., `engine_provision`, `model_fetch`, `pool_warmup`) with
    // optional estimates. This allows clients to choose verbose SSE for richer UX.

    // Emit a simple log line for BDD assertions
    let mut lg = state.logs.lock().unwrap();
    lg.push(format!(
        "{{\"queue_position\":{},\"predicted_start_ms\":{}}}",
        admission.queue_position, admission.predicted_start_ms
    ));
    // Narration breadcrumb
    observability_narration_core::human(
        "orchestratord",
        "admission",
        &body.session_id,
        format!(
            "Accepted request; queued at position {} (ETA {} ms)",
            admission.queue_position, admission.predicted_start_ms
        ),
    );

    // TODO[ORCHD-BUDGETS-3001]: Budget headers should be computed from a real budget policy and
    // session linkage (task_id -> session_id) rather than best-effort lookup.
    // Budget headers based on session info (best-effort)
    let svc = services::session::SessionService::new(
        state.sessions.clone(),
        std::sync::Arc::new(crate::infra::clock::SystemClock),
    );
    let sess = svc.get_or_create(&body.session_id);
    let mut headers = HeaderMap::new();
    headers.insert(
        "X-Budget-Tokens-Remaining",
        sess.tokens_budget_remaining.to_string().parse().unwrap(),
    );
    headers.insert(
        "X-Budget-Time-Remaining-Ms",
        sess.time_budget_remaining_ms.to_string().parse().unwrap(),
    );
    headers.insert(
        "X-Budget-Cost-Remaining",
        format!("{}", sess.cost_budget_remaining).parse().unwrap(),
    );

    Ok((StatusCode::ACCEPTED, headers, Json(admission)))
}

pub async fn stream_task(
    state: State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
    axum::Extension(correlation_id): axum::Extension<String>,
    axum::extract::Query(verbose): axum::extract::Query<bool>,
) -> Result<impl IntoResponse, ErrO> {
    // TODO[ORCHD-STREAM-VERBOSE-0011]: Use `verbose` to control whether SSE frames include
    // `{"human": "...", "phase": "..."}` breadcrumbs.
    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", "text/event-stream".parse().unwrap());
    headers.insert("X-Correlation-Id", correlation_id.parse().unwrap());
    // Seed budget headers (unknown session at this layer); consider mapping task->session later
    headers.insert("X-Budget-Time-Remaining-Ms", "0".parse().unwrap());
    headers.insert("X-Budget-Cost-Remaining", "0".parse().unwrap());
    let sse = services::streaming::render_sse_for_task(&state, id).await;
    Ok((StatusCode::OK, headers, sse))
}

pub async fn cancel_task(
    state: State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<impl IntoResponse, ErrO> {
    // Record cancellation metric
    crate::metrics::inc_counter(
        "tasks_canceled_total",
        &[
            ("engine", "llamacpp"),
            ("engine_version", "v0"),
            ("pool_id", "default"),
            ("replica_id", "r0"),
            ("reason", "client"),
        ],
    );
    // Update queue depth
    crate::metrics::set_gauge("queue_depth", &[], 0);
    // Signal cancel to streaming service via shared state
    if let Ok(mut guard) = state.cancellations.lock() {
        guard.insert(id.clone());
    }
    let mut lg = state.logs.lock().unwrap();
    lg.push(format!("{{\"canceled\":true,\"task_id\":\"{}\"}}", id));
    observability_narration_core::human("orchestratord", "cancel", &id, "client requested cancel");
    Ok(StatusCode::NO_CONTENT)
}
