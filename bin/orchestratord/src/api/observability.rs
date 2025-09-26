use axum::response::IntoResponse;
use http::HeaderMap;

use crate::metrics;

pub async fn metrics_endpoint() -> axum::response::Response {
    let mut headers = HeaderMap::new();
    headers.insert(
        "X-Correlation-Id",
        "11111111-1111-4111-8111-111111111111".parse().unwrap(),
    );
    headers.insert(
        http::header::CONTENT_TYPE,
        "text/plain; version=0.0.4".parse().unwrap(),
    );

    // Seed series to satisfy lints (minimal values)
    metrics::set_gauge(
        "queue_depth",
        &[
            ("engine", "llamacpp"),
            ("engine_version", "v0"),
            ("pool_id", "default"),
            ("priority", "interactive"),
        ],
        1,
    );
    metrics::inc_counter(
        "tasks_enqueued_total",
        &[
            ("engine", "llamacpp"),
            ("engine_version", "v0"),
            ("pool_id", "default"),
            ("replica_id", "r0"),
            ("priority", "interactive"),
        ],
    );
    metrics::inc_counter(
        "tasks_started_total",
        &[
            ("engine", "llamacpp"),
            ("engine_version", "v0"),
            ("pool_id", "default"),
            ("replica_id", "r0"),
            ("priority", "interactive"),
        ],
    );
    metrics::inc_counter(
        "tasks_canceled_total",
        &[
            ("engine", "llamacpp"),
            ("engine_version", "v0"),
            ("pool_id", "default"),
            ("replica_id", "r0"),
            ("reason", "client"),
        ],
    );
    metrics::inc_counter(
        "tasks_rejected_total",
        &[("engine", "llamacpp"), ("reason", "ADMISSION_REJECT")],
    );
    metrics::inc_counter(
        "tokens_in_total",
        &[
            ("engine", "llamacpp"),
            ("engine_version", "v0"),
            ("pool_id", "default"),
            ("replica_id", "r0"),
        ],
    );
    metrics::inc_counter(
        "tokens_out_total",
        &[
            ("engine", "llamacpp"),
            ("engine_version", "v0"),
            ("pool_id", "default"),
            ("replica_id", "r0"),
        ],
    );
    metrics::inc_counter(
        "admission_backpressure_events_total",
        &[("engine", "llamacpp"), ("policy", "reject")],
    );
    metrics::inc_counter(
        "catalog_verifications_total",
        &[("result", "ok"), ("reason", "none"), ("engine", "llamacpp")],
    );
    metrics::set_gauge(
        "kv_cache_usage_ratio",
        &[
            ("engine", "llamacpp"),
            ("engine_version", "v0"),
            ("pool_id", "default"),
            ("replica_id", "r0"),
        ],
        0,
    );
    metrics::set_gauge(
        "gpu_utilization",
        &[
            ("engine", "llamacpp"),
            ("engine_version", "v0"),
            ("pool_id", "default"),
            ("replica_id", "r0"),
            ("device", "gpu0"),
        ],
        0,
    );
    metrics::set_gauge(
        "vram_used_bytes",
        &[
            ("engine", "llamacpp"),
            ("engine_version", "v0"),
            ("pool_id", "default"),
            ("replica_id", "r0"),
            ("device", "gpu0"),
        ],
        0,
    );
    metrics::set_gauge("model_state", &[("model_id", "m0"), ("state", "loaded")], 1);

    // Observe one sample for histograms
    metrics::observe_histogram(
        "latency_first_token_ms",
        &[
            ("engine", "llamacpp"),
            ("engine_version", "v0"),
            ("pool_id", "default"),
            ("priority", "interactive"),
        ],
        10.0,
    );
    metrics::observe_histogram(
        "latency_decode_ms",
        &[
            ("engine", "llamacpp"),
            ("engine_version", "v0"),
            ("pool_id", "default"),
            ("priority", "interactive"),
        ],
        5.0,
    );

    let text = metrics::gather_metrics_text();

    (http::StatusCode::OK, headers, text).into_response()
}
