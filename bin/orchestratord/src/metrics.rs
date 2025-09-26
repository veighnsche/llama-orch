use std::collections::HashMap;
use std::sync::Mutex;

use once_cell::sync::Lazy;

#[derive(Clone, Hash, PartialEq, Eq)]
struct LabelsKey(String);

static COUNTERS: Lazy<Mutex<HashMap<String, HashMap<LabelsKey, i64>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static GAUGES: Lazy<Mutex<HashMap<String, HashMap<LabelsKey, i64>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static HISTOGRAMS: Lazy<Mutex<HashMap<String, HashMap<LabelsKey, Histogram>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

// Throttle counters for noisy metrics (e.g., per token)
static THROTTLE: Lazy<Mutex<HashMap<String, u64>>> = Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Clone, Debug)]
struct Histogram {
    buckets: Vec<f64>,
    counts: Vec<u64>,
    sum: f64,
    count: u64,
}

fn labels_to_key(pairs: &[(&str, &str)]) -> LabelsKey {
    let mut s = String::new();
    let mut first = true;
    for (k, v) in pairs {
        if !first {
            s.push(',');
        } else {
            first = false;
        }
        s.push_str(&format!("{}=\"{}\"", k, v));
    }
    LabelsKey(s)
}

pub fn inc_counter(name: &str, labels: &[(&str, &str)]) {
    let mut c = COUNTERS.lock().unwrap();
    let entry = c.entry(name.to_string()).or_default();
    let key = labels_to_key(labels);
    *entry.entry(key).or_insert(0) += 1;
}

pub fn set_gauge(name: &str, labels: &[(&str, &str)], value: i64) {
    let mut g = GAUGES.lock().unwrap();
    let entry = g.entry(name.to_string()).or_default();
    let key = labels_to_key(labels);
    entry.insert(key, value);
}

/// Observe a histogram sample with default buckets.
pub fn observe_histogram(name: &str, labels: &[(&str, &str)], value_ms: f64) {
    let mut h = HISTOGRAMS.lock().unwrap();
    let entry = h.entry(name.to_string()).or_default();
    let key = labels_to_key(labels);
    let hist = entry.entry(key).or_insert_with(|| Histogram {
        buckets: vec![50.0, 100.0, 200.0, 500.0, 1000.0, 5000.0, 10000.0],
        counts: vec![0; 7],
        sum: 0.0,
        count: 0,
    });
    hist.sum += value_ms;
    hist.count += 1;
    for (i, b) in hist.buckets.iter().enumerate() {
        if value_ms <= *b {
            hist.counts[i] += 1;
        }
    }
}

/// Observe every N samples to reduce cardinality load.
pub fn observe_histogram_throttled(
    name: &str,
    labels: &[(&str, &str)],
    value_ms: f64,
    every_n: u64,
) {
    let key = format!("{}|{}", name, labels_to_key(labels).0);
    let mut t = THROTTLE.lock().unwrap();
    let cnt = t.entry(key).or_insert(0);
    *cnt += 1;
    if *cnt % every_n == 0 {
        observe_histogram(name, labels, value_ms);
    }
}

/// Pre-register common metrics with zero values for consistent exposition.
pub fn pre_register() {
    // counters
    for name in [
        "tasks_enqueued_total",
        "tasks_started_total",
        "tasks_canceled_total",
        "tasks_rejected_total",
        "tokens_in_total",
        "tokens_out_total",
        "admission_backpressure_events_total",
        "catalog_verifications_total",
    ] {
        let _ = {
            let mut c = COUNTERS.lock().unwrap();
            c.entry(name.to_string()).or_default();
        };
    }
    // gauges
    for name in [
        "queue_depth",
        "kv_cache_usage_ratio",
        "gpu_utilization",
        "vram_used_bytes",
        "model_state",
    ] {
        let _ = {
            let mut g = GAUGES.lock().unwrap();
            g.entry(name.to_string()).or_default();
        };
    }
    // histograms
    for name in ["latency_first_token_ms", "latency_decode_ms"] {
        let _ = {
            let mut h = HISTOGRAMS.lock().unwrap();
            h.entry(name.to_string()).or_default();
        };
    }
}

pub fn gather_metrics_text() -> String {
    let mut out = String::new();
    // TYPE headers for required metrics (shows up even with no samples)
    out.push_str("# TYPE tasks_enqueued_total counter\n");
    out.push_str("# TYPE tasks_started_total counter\n");
    out.push_str("# TYPE tasks_canceled_total counter\n");
    out.push_str("# TYPE tasks_rejected_total counter\n");
    out.push_str("# TYPE tokens_in_total counter\n");
    out.push_str("# TYPE tokens_out_total counter\n");
    out.push_str("# TYPE admission_backpressure_events_total counter\n");
    out.push_str("# TYPE catalog_verifications_total counter\n");
    out.push_str("# TYPE queue_depth gauge\n");
    out.push_str("# TYPE kv_cache_usage_ratio gauge\n");
    out.push_str("# TYPE gpu_utilization gauge\n");
    out.push_str("# TYPE vram_used_bytes gauge\n");
    out.push_str("# TYPE model_state gauge\n");
    out.push_str("# TYPE latency_first_token_ms histogram\n");
    out.push_str("# TYPE latency_decode_ms histogram\n");
    out.push_str("\n");

    let c = COUNTERS.lock().unwrap();
    for (name, series) in c.iter() {
        for (labels, val) in series.iter() {
            out.push_str(&format!("{}{{{}}} {}\n", name, labels.0, val));
        }
    }
    let g = GAUGES.lock().unwrap();
    for (name, series) in g.iter() {
        for (labels, val) in series.iter() {
            out.push_str(&format!("{}{{{}}} {}\n", name, labels.0, val));
        }
    }
    let h = HISTOGRAMS.lock().unwrap();
    for (name, series) in h.iter() {
        for (labels, hist) in series.iter() {
            let mut cum = 0u64;
            for (i, b) in hist.buckets.iter().enumerate() {
                cum += hist.counts[i];
                out.push_str(&format!(
                    "{}_bucket{{{},le=\"{}\"}} {}\n",
                    name, labels.0, b, cum
                ));
            }
            // +Inf bucket
            out.push_str(&format!(
                "{}_bucket{{{},le=\"+Inf\"}} {}\n",
                name, labels.0, hist.count
            ));
            out.push_str(&format!("{}_sum{{{}}} {}\n", name, labels.0, hist.sum));
            out.push_str(&format!("{}_count{{{}}} {}\n", name, labels.0, hist.count));
        }
    }
    out
}
