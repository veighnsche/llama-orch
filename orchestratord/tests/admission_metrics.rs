use orchestrator_core::queue::{Policy, Priority};
use orchestratord::{admission, metrics};

fn labels(pool: &str, replica: &str) -> admission::MetricLabels {
    admission::MetricLabels {
        engine: "llamacpp".into(),
        engine_version: "v0".into(),
        pool_id: pool.into(),
        replica_id: replica.into(),
    }
}

fn contains_metric_line(text: &str, metric: &str, label_snips: &[(&str, &str)], value_suffix: &str) -> bool {
    for line in text.lines() {
        if line.starts_with(metric) {
            if label_snips.iter().all(|(k, v)| line.contains(&format!("{}=\"{}\"", k, v)))
                && line.trim_end().ends_with(value_suffix)
            {
                return true;
            }
        }
    }
    false
}

#[test]
fn enqueue_emits_enqueued_and_depth() {
    let mut q = admission::QueueWithMetrics::new(8, Policy::DropLru, labels("pool-test", "r1"));
    assert!(q.enqueue(1, Priority::Interactive).is_ok());

    let text = metrics::gather_metrics_text();
    assert!(contains_metric_line(
        &text,
        "tasks_enqueued_total",
        &[
            ("engine", "llamacpp"),
            ("engine_version", "v0"),
            ("pool_id", "pool-test"),
            ("replica_id", "r1"),
            ("priority", "interactive"),
        ],
        " 1",
    ));
    assert!(contains_metric_line(
        &text,
        "queue_depth",
        &[("engine", "llamacpp"), ("pool_id", "pool-test"), ("priority", "interactive")],
        " 1",
    ));
}

#[test]
fn full_queue_reject_emits_backpressure_and_reject() {
    let mut q = admission::QueueWithMetrics::new(1, Policy::Reject, labels("pool-reject", "r2"));
    assert!(q.enqueue(1, Priority::Batch).is_ok());
    let _ = q.enqueue(2, Priority::Batch); // should reject

    let text = metrics::gather_metrics_text();
    assert!(contains_metric_line(
        &text,
        "admission_backpressure_events_total",
        &[("engine", "llamacpp"), ("policy", "reject")],
        " 1",
    ));
    assert!(contains_metric_line(
        &text,
        "tasks_rejected_total",
        &[("engine", "llamacpp"), ("reason", "ADMISSION_REJECT")],
        " 1",
    ));
}

#[test]
fn full_queue_drop_lru_emits_backpressure_and_drop_reason() {
    let mut q = admission::QueueWithMetrics::new(1, Policy::DropLru, labels("pool-drop", "r3"));
    assert!(q.enqueue(1, Priority::Batch).is_ok());
    assert!(q.enqueue(2, Priority::Batch).is_ok()); // should drop-lru then accept

    let text = metrics::gather_metrics_text();
    assert!(contains_metric_line(
        &text,
        "admission_backpressure_events_total",
        &[("engine", "llamacpp"), ("policy", "drop-lru")],
        " 1",
    ));
    assert!(contains_metric_line(
        &text,
        "tasks_rejected_total",
        &[("engine", "llamacpp"), ("reason", "QUEUE_FULL_DROP_LRU")],
        " 1",
    ));
}
