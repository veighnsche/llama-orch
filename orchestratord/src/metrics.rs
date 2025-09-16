use once_cell::sync::Lazy;
use prometheus::{
    register_gauge_vec_with_registry, register_histogram_vec_with_registry,
    register_int_counter_vec_with_registry, register_int_gauge_vec_with_registry, Encoder,
    GaugeVec, HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec, Opts, Registry, TextEncoder,
};

pub static REGISTRY: Lazy<Registry> = Lazy::new(Registry::new);

// Counters
pub static TASKS_ENQUEUED_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    register_int_counter_vec_with_registry!(
        Opts::new(
            "tasks_enqueued_total",
            "Number of tasks accepted into the queue."
        ),
        &[
            "engine",
            "engine_version",
            "pool_id",
            "replica_id",
            "priority",
        ],
        &*REGISTRY
    )
    .expect("register tasks_enqueued_total")
});

pub static TASKS_STARTED_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    register_int_counter_vec_with_registry!(
        Opts::new(
            "tasks_started_total",
            "Number of tasks that started decoding."
        ),
        &[
            "engine",
            "engine_version",
            "pool_id",
            "replica_id",
            "priority"
        ],
        &*REGISTRY
    )
    .expect("register tasks_started_total")
});

pub static TASKS_CANCELED_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    register_int_counter_vec_with_registry!(
        Opts::new("tasks_canceled_total", "Number of canceled tasks."),
        &[
            "engine",
            "engine_version",
            "pool_id",
            "replica_id",
            "reason"
        ],
        &*REGISTRY
    )
    .expect("register tasks_canceled_total")
});

pub static TASKS_REJECTED_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    // Spec exception: omit engine_version
    register_int_counter_vec_with_registry!(
        Opts::new(
            "tasks_rejected_total",
            "Number of rejected tasks by reason."
        ),
        &["engine", "reason"],
        &*REGISTRY
    )
    .expect("register tasks_rejected_total")
});

pub static ADMISSION_BACKPRESSURE_EVENTS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    register_int_counter_vec_with_registry!(
        Opts::new(
            "admission_backpressure_events_total",
            "Number of backpressure decisions taken.",
        ),
        &["engine", "policy"],
        &*REGISTRY
    )
    .expect("register admission_backpressure_events_total")
});

pub static TOKENS_IN_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    register_int_counter_vec_with_registry!(
        Opts::new("tokens_in_total", "Total prompt tokens processed."),
        &["engine", "engine_version", "pool_id", "replica_id"],
        &*REGISTRY
    )
    .expect("register tokens_in_total")
});

pub static TOKENS_OUT_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    register_int_counter_vec_with_registry!(
        Opts::new("tokens_out_total", "Total output tokens generated."),
        &["engine", "engine_version", "pool_id", "replica_id"],
        &*REGISTRY
    )
    .expect("register tokens_out_total")
});

pub static CATALOG_VERIFICATIONS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    register_int_counter_vec_with_registry!(
        Opts::new(
            "catalog_verifications_total",
            "Model artifact verification outcomes."
        ),
        &["result", "reason", "engine"],
        &*REGISTRY
    )
    .expect("register catalog_verifications_total")
});

pub static PREEMPTIONS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    register_int_counter_vec_with_registry!(
        Opts::new("preemptions_total", "Number of preemption events."),
        &["mode", "engine"],
        &*REGISTRY
    )
    .expect("register preemptions_total")
});

pub static RESUMPTIONS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    register_int_counter_vec_with_registry!(
        Opts::new(
            "resumptions_total",
            "Number of resumed jobs after preemption."
        ),
        &["engine"],
        &*REGISTRY
    )
    .expect("register resumptions_total")
});

// Gauges
pub static QUEUE_DEPTH: Lazy<IntGaugeVec> = Lazy::new(|| {
    register_int_gauge_vec_with_registry!(
        Opts::new("queue_depth", "Current queue length per pool/priority."),
        &["engine", "engine_version", "pool_id", "priority"],
        &*REGISTRY
    )
    .expect("register queue_depth")
});

pub static MODEL_STATE: Lazy<IntGaugeVec> = Lazy::new(|| {
    register_int_gauge_vec_with_registry!(
        Opts::new("model_state", "Current lifecycle state per model."),
        &["model_id", "state"],
        &*REGISTRY
    )
    .expect("register model_state")
});

pub static KV_CACHE_USAGE_RATIO: Lazy<GaugeVec> = Lazy::new(|| {
    register_gauge_vec_with_registry!(
        Opts::new("kv_cache_usage_ratio", "KV cache usage ratio."),
        &["engine", "engine_version", "pool_id", "replica_id"],
        &*REGISTRY
    )
    .expect("register kv_cache_usage_ratio")
});

pub static GPU_UTILIZATION: Lazy<GaugeVec> = Lazy::new(|| {
    register_gauge_vec_with_registry!(
        Opts::new("gpu_utilization", "GPU utilization percent."),
        &[
            "engine",
            "engine_version",
            "pool_id",
            "replica_id",
            "device"
        ],
        &*REGISTRY
    )
    .expect("register gpu_utilization")
});

pub static VRAM_USED_BYTES: Lazy<IntGaugeVec> = Lazy::new(|| {
    register_int_gauge_vec_with_registry!(
        Opts::new("vram_used_bytes", "VRAM used by worker processes."),
        &[
            "engine",
            "engine_version",
            "pool_id",
            "replica_id",
            "device"
        ],
        &*REGISTRY
    )
    .expect("register vram_used_bytes")
});

pub static ADMISSION_SHARE: Lazy<GaugeVec> = Lazy::new(|| {
    register_gauge_vec_with_registry!(
        Opts::new(
            "admission_share",
            "EWMA of observed admission share for fairness."
        ),
        &["tenant", "priority"],
        &*REGISTRY
    )
    .expect("register admission_share")
});

pub static DEADLINES_MET_RATIO: Lazy<GaugeVec> = Lazy::new(|| {
    register_gauge_vec_with_registry!(
        Opts::new("deadlines_met_ratio", "Ratio of jobs meeting deadlines."),
        &["priority"],
        &*REGISTRY
    )
    .expect("register deadlines_met_ratio")
});

// Histograms
pub static LATENCY_FIRST_TOKEN_MS: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec_with_registry!(
        HistogramOpts::new(
            "latency_first_token_ms",
            "Time to first token since admission."
        ),
        &["engine", "engine_version", "pool_id", "priority"],
        &*REGISTRY
    )
    .expect("register latency_first_token_ms")
});

pub static LATENCY_DECODE_MS: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec_with_registry!(
        HistogramOpts::new("latency_decode_ms", "Decode time for full stream per job."),
        &["engine", "engine_version", "pool_id", "priority"],
        &*REGISTRY
    )
    .expect("register latency_decode_ms")
});

pub fn gather_metrics_text() -> String {
    // Ensure all metric families are registered and have at least one child instantiated
    let _ = &*TASKS_ENQUEUED_TOTAL;
    let _ = &*TASKS_STARTED_TOTAL;
    let _ = &*TASKS_CANCELED_TOTAL;
    let _ = &*TASKS_REJECTED_TOTAL;
    let _ = &*ADMISSION_BACKPRESSURE_EVENTS_TOTAL;
    let _ = &*TOKENS_IN_TOTAL;
    let _ = &*TOKENS_OUT_TOTAL;
    let _ = &*CATALOG_VERIFICATIONS_TOTAL;
    let _ = &*PREEMPTIONS_TOTAL;
    let _ = &*RESUMPTIONS_TOTAL;
    let _ = &*QUEUE_DEPTH;
    let _ = &*MODEL_STATE;
    let _ = &*KV_CACHE_USAGE_RATIO;
    let _ = &*GPU_UTILIZATION;
    let _ = &*VRAM_USED_BYTES;
    let _ = &*ADMISSION_SHARE;
    let _ = &*DEADLINES_MET_RATIO;
    let _ = &*LATENCY_FIRST_TOKEN_MS;
    let _ = &*LATENCY_DECODE_MS;

    ensure_samples();
    let metric_families = REGISTRY.gather();
    let mut buf = Vec::new();
    let encoder = TextEncoder::new();
    encoder.encode(&metric_families, &mut buf).unwrap();
    String::from_utf8(buf).unwrap_or_default()
}

// Streaming emissions helpers (to be called by data-plane streaming once implemented)
pub fn record_stream_started(
    engine: &str,
    engine_version: &str,
    pool_id: &str,
    replica_id: &str,
    priority: &str,
    first_token_ms: u64,
    tokens_in: u64,
) {
    TASKS_STARTED_TOTAL
        .with_label_values(&[engine, engine_version, pool_id, replica_id, priority])
        .inc();
    LATENCY_FIRST_TOKEN_MS
        .with_label_values(&[engine, engine_version, pool_id, priority])
        .observe(first_token_ms as f64);
    TOKENS_IN_TOTAL
        .with_label_values(&[engine, engine_version, pool_id, replica_id])
        .inc_by(tokens_in);
}

pub fn record_stream_ended(
    engine: &str,
    engine_version: &str,
    pool_id: &str,
    replica_id: &str,
    priority: &str,
    decode_ms: u64,
    tokens_out: u64,
) {
    LATENCY_DECODE_MS
        .with_label_values(&[engine, engine_version, pool_id, priority])
        .observe(decode_ms as f64);
    TOKENS_OUT_TOTAL
        .with_label_values(&[engine, engine_version, pool_id, replica_id])
        .inc_by(tokens_out);
}

fn ensure_samples() {
    // Use a single placeholder label set to instantiate one child per vector.
    let eng = "llamacpp";
    let engv = "v0";
    let pool = "pool0";
    let rep = "r0";
    let prio = "interactive";
    let reason = "INTERNAL";
    let policy = "reject";
    let result = "pass";
    let device = "0";
    let model = "model0";
    let state = "Draft";
    let tenant = "t0";
    let mode = "soft";

    let _ = TASKS_ENQUEUED_TOTAL.with_label_values(&[eng, engv, pool, rep, prio]);
    let _ = TASKS_STARTED_TOTAL.with_label_values(&[eng, engv, pool, rep, prio]);
    let _ = TASKS_CANCELED_TOTAL.with_label_values(&[eng, engv, pool, rep, reason]);
    let _ = TASKS_REJECTED_TOTAL.with_label_values(&[eng, reason]);
    let _ = ADMISSION_BACKPRESSURE_EVENTS_TOTAL.with_label_values(&[eng, policy]);
    let _ = TOKENS_IN_TOTAL.with_label_values(&[eng, engv, pool, rep]);
    let _ = TOKENS_OUT_TOTAL.with_label_values(&[eng, engv, pool, rep]);
    let _ = CATALOG_VERIFICATIONS_TOTAL.with_label_values(&[result, reason, eng]);
    let _ = PREEMPTIONS_TOTAL.with_label_values(&[mode, eng]);
    let _ = RESUMPTIONS_TOTAL.with_label_values(&[eng]);

    let _ = QUEUE_DEPTH.with_label_values(&[eng, engv, pool, prio]);
    let _ = MODEL_STATE.with_label_values(&[model, state]);
    let _ = KV_CACHE_USAGE_RATIO.with_label_values(&[eng, engv, pool, rep]);
    let _ = GPU_UTILIZATION.with_label_values(&[eng, engv, pool, rep, device]);
    let _ = VRAM_USED_BYTES.with_label_values(&[eng, engv, pool, rep, device]);
    let _ = ADMISSION_SHARE.with_label_values(&[tenant, prio]);
    let _ = DEADLINES_MET_RATIO.with_label_values(&[prio]);
    let _ = LATENCY_FIRST_TOKEN_MS.with_label_values(&[eng, engv, pool, prio]);
    let _ = LATENCY_DECODE_MS.with_label_values(&[eng, engv, pool, prio]);
}
