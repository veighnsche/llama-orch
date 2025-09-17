use std::collections::HashMap;
use std::sync::Mutex;

use once_cell::sync::Lazy;

#[derive(Clone, Hash, PartialEq, Eq)]
struct LabelsKey(String);

static COUNTERS: Lazy<Mutex<HashMap<String, HashMap<LabelsKey, i64>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static GAUGES: Lazy<Mutex<HashMap<String, HashMap<LabelsKey, i64>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn labels_to_key(pairs: &[(&str, &str)]) -> LabelsKey {
    let mut s = String::new();
    let mut first = true;
    for (k, v) in pairs {
        if !first { s.push(','); } else { first = false; }
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

pub fn gather_metrics_text() -> String {
    let mut out = String::new();
    // headers for known metrics
    out.push_str("# TYPE tasks_enqueued_total counter\n\n");
    out.push_str("# TYPE queue_depth gauge\n");
    out.push_str("# TYPE admission_backpressure_events_total counter\n");
    out.push_str("# TYPE tasks_rejected_total counter\n");

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
    out
}
