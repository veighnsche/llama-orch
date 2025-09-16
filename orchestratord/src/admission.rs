use orchestrator_core::queue::{EnqueueError, InMemoryQueue, Policy, Priority};

/// Labels identifying where metrics are emitted from.
#[derive(Clone, Debug)]
pub struct MetricLabels {
    pub engine: String,
    pub engine_version: String,
    pub pool_id: String,
    pub replica_id: String,
}

/// A thin wrapper around InMemoryQueue that emits metrics on admission events.
#[derive(Debug)]
pub struct QueueWithMetrics {
    inner: InMemoryQueue,
    labels: MetricLabels,
}

impl QueueWithMetrics {
    pub fn new(capacity: usize, policy: Policy, labels: MetricLabels) -> Self {
        Self {
            inner: InMemoryQueue::with_capacity_policy(capacity, policy),
            labels,
        }
    }

    pub fn inner(&self) -> &InMemoryQueue {
        &self.inner
    }
    pub fn inner_mut(&mut self) -> &mut InMemoryQueue {
        &mut self.inner
    }

    fn priority_str(prio: Priority) -> &'static str {
        match prio {
            Priority::Interactive => "interactive",
            Priority::Batch => "batch",
        }
    }

    pub fn enqueue(&mut self, id: u32, prio: Priority) -> Result<(), EnqueueError> {
        let before = self.inner.len();
        let cap = self.inner.capacity();
        let res = self.inner.enqueue(id, prio);
        match res {
            Ok(()) => {
                // If queue was at capacity and policy is DropLru, then a drop occurred.
                if before >= cap {
                    crate::metrics::ADMISSION_BACKPRESSURE_EVENTS_TOTAL
                        .with_label_values(&[&self.labels.engine, "drop-lru"])
                        .inc();
                    crate::metrics::TASKS_REJECTED_TOTAL
                        .with_label_values(&[&self.labels.engine, "QUEUE_FULL_DROP_LRU"])
                        .inc();
                }
                // Record acceptance and depth
                crate::metrics::TASKS_ENQUEUED_TOTAL
                    .with_label_values(&[
                        &self.labels.engine,
                        &self.labels.engine_version,
                        &self.labels.pool_id,
                        &self.labels.replica_id,
                        Self::priority_str(prio),
                    ])
                    .inc();
                crate::metrics::QUEUE_DEPTH
                    .with_label_values(&[
                        &self.labels.engine,
                        &self.labels.engine_version,
                        &self.labels.pool_id,
                        Self::priority_str(prio),
                    ])
                    .set(self.inner.len() as i64);
                Ok(())
            }
            Err(EnqueueError::QueueFullReject) => {
                crate::metrics::ADMISSION_BACKPRESSURE_EVENTS_TOTAL
                    .with_label_values(&[&self.labels.engine, "reject"])
                    .inc();
                crate::metrics::TASKS_REJECTED_TOTAL
                    .with_label_values(&[&self.labels.engine, "ADMISSION_REJECT"])
                    .inc();
                Err(EnqueueError::QueueFullReject)
            }
        }
    }

    pub fn cancel(&mut self, id: u32, reason: &str) -> bool {
        let was = self.inner.len();
        let ok = self.inner.cancel(id);
        if ok {
            // Emit cancel counter
            crate::metrics::TASKS_CANCELED_TOTAL
                .with_label_values(&[
                    &self.labels.engine,
                    &self.labels.engine_version,
                    &self.labels.pool_id,
                    &self.labels.replica_id,
                    reason,
                ])
                .inc();
            // Depth may have decreased; update both priorities gauges for simplicity
            // In a real implementation, we'd decrement the specific priority's gauge.
            for prio in [Priority::Interactive, Priority::Batch] {
                crate::metrics::QUEUE_DEPTH
                    .with_label_values(&[
                        &self.labels.engine,
                        &self.labels.engine_version,
                        &self.labels.pool_id,
                        Self::priority_str(prio),
                    ])
                    .set(self.inner.len() as i64);
            }
        }
        debug_assert!(self.inner.len() <= was);
        ok
    }
}
