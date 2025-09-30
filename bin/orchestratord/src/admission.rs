use orchestrator_core::queue::{InMemoryQueue, Policy, Priority};

use crate::metrics;

#[derive(Debug, Clone)]
pub struct MetricLabels {
    pub engine: String,
    pub engine_version: String,
    pub pool_id: String,
    pub replica_id: String,
}

#[derive(Debug)]
pub struct QueueWithMetrics {
    inner: InMemoryQueue,
    policy: Policy,
    labels: MetricLabels,
}

impl QueueWithMetrics {
    pub fn new(capacity: usize, policy: Policy, labels: MetricLabels) -> Self {
        Self { inner: InMemoryQueue::with_capacity_policy(capacity, policy), policy, labels }
    }

    /// Enqueue a task id with priority. Returns the queue position (0-based within the
    /// priority class) on success. Returns Err(()) when reject policy applies and item is not queued.
    pub fn enqueue(&mut self, id: u32, priority: Priority) -> Result<i64, ()> {
        use Priority::*;
        let prio = match priority {
            Interactive => "interactive",
            Batch => "batch",
        };
        let labels_common = [
            ("engine", self.labels.engine.as_str()),
            ("engine_version", self.labels.engine_version.as_str()),
            ("pool_id", self.labels.pool_id.as_str()),
            ("replica_id", self.labels.replica_id.as_str()),
            ("priority", prio),
        ];

        let mut enqueued = false;
        if self.inner.len() >= self.inner.capacity() {
            match self.policy {
                Policy::Reject => {
                    metrics::inc_counter(
                        "admission_backpressure_events_total",
                        &[("engine", self.labels.engine.as_str()), ("policy", "reject")],
                    );
                    metrics::inc_counter(
                        "tasks_rejected_total",
                        &[("engine", self.labels.engine.as_str()), ("reason", "ADMISSION_REJECT")],
                    );
                    // Do not enqueue
                    return Err(());
                }
                Policy::DropLru => {
                    metrics::inc_counter(
                        "admission_backpressure_events_total",
                        &[("engine", self.labels.engine.as_str()), ("policy", "drop-lru")],
                    );
                    metrics::inc_counter(
                        "tasks_rejected_total",
                        &[
                            ("engine", self.labels.engine.as_str()),
                            ("reason", "QUEUE_FULL_DROP_LRU"),
                        ],
                    );
                    // Let inner queue apply policy by enqueuing; LRU will be dropped internally
                    let _ = self.inner.enqueue(id, priority);
                    enqueued = true;
                }
            }
        } else {
            let _ = self.inner.enqueue(id, priority);
            enqueued = true;
        }

        if enqueued {
            metrics::inc_counter("tasks_enqueued_total", &labels_common);
        }
        let depth_labels = [
            ("engine", self.labels.engine.as_str()),
            ("pool_id", self.labels.pool_id.as_str()),
            ("priority", prio),
        ];
        let depth_now = self.inner.len() as i64;
        metrics::set_gauge("queue_depth", &depth_labels, depth_now);
        // Derive queue position within the priority class
        let pos = match priority {
            Interactive => self
                .inner
                .snapshot_priority(Priority::Interactive)
                .iter()
                .position(|&x| x == id)
                .unwrap_or(0) as i64,
            Batch => self
                .inner
                .snapshot_priority(Priority::Batch)
                .iter()
                .position(|&x| x == id)
                .unwrap_or(0) as i64,
        };
        Ok(pos)
    }
}
