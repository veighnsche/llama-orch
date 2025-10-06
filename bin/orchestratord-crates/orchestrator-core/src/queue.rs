//! Simple in-memory queue to support core invariants property tests.
//! Traceability (SPEC):
//! - OC-CORE-1001: Bounded queues and admission behavior
//! - OC-CORE-1002: Queue full policies (reject, drop-lru)
//! - OC-CORE-1004: FIFO within the same priority class

use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Priority {
    Interactive,
    Batch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    Reject,
    DropLru,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum EnqueueError {
    #[error("queue full (reject policy)")]
    QueueFullReject,
}

#[derive(Debug)]
pub struct InMemoryQueue {
    interactive: VecDeque<u32>,
    batch: VecDeque<u32>,
    capacity: usize,
    policy: Policy,
}

impl InMemoryQueue {
    #[must_use] 
    pub fn with_capacity_policy(capacity: usize, policy: Policy) -> Self {
        Self { interactive: VecDeque::new(), batch: VecDeque::new(), capacity, policy }
    }

    #[must_use] 
    pub fn len(&self) -> usize {
        self.interactive.len() + self.batch.len()
    }

    #[must_use] 
    pub fn is_empty(&self) -> bool {
        self.interactive.is_empty() && self.batch.is_empty()
    }

    #[must_use] 
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn enqueue(&mut self, id: u32, prio: Priority) -> Result<(), EnqueueError> {
        if self.len() >= self.capacity {
            match self.policy {
                Policy::Reject => return Err(EnqueueError::QueueFullReject),
                Policy::DropLru => {
                    // Prefer to drop the oldest batch item first; if none, drop oldest overall.
                    if !self.batch.is_empty() {
                        self.batch.pop_front();
                    } else if !self.interactive.is_empty() {
                        self.interactive.pop_front();
                    }
                    // If both empty, nothing to drop; fall through.
                }
            }
        }
        match prio {
            Priority::Interactive => self.interactive.push_back(id),
            Priority::Batch => self.batch.push_back(id),
        }
        Ok(())
    }

    pub fn cancel(&mut self, id: u32) -> bool {
        // Remove first occurrence from either queue.
        if let Some(pos) = self.interactive.iter().position(|&x| x == id) {
            self.interactive.remove(pos);
            return true;
        }
        if let Some(pos) = self.batch.iter().position(|&x| x == id) {
            self.batch.remove(pos);
            return true;
        }
        false
    }

    #[must_use] 
    pub fn snapshot_priority(&self, prio: Priority) -> Vec<u32> {
        match prio {
            Priority::Interactive => self.interactive.iter().copied().collect(),
            Priority::Batch => self.batch.iter().copied().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // OC-CORE-1001, OC-CORE-1002: bounded queues and reject policy on full
    #[test]
    fn test_oc_core_1001_reject_when_full() {
        let mut q = InMemoryQueue::with_capacity_policy(1, Policy::Reject);
        assert!(q.enqueue(1, Priority::Interactive).is_ok());
        let err = q.enqueue(2, Priority::Interactive).unwrap_err();
        assert_eq!(err, EnqueueError::QueueFullReject);
        assert_eq!(q.len(), 1);
    }

    // OC-CORE-1002: drop-lru prefers oldest batch item
    #[test]
    fn test_oc_core_1002_drop_lru_prefers_batch() {
        let mut q = InMemoryQueue::with_capacity_policy(2, Policy::DropLru);
        // Oldest batch first, then an interactive
        assert!(q.enqueue(1, Priority::Batch).is_ok());
        assert!(q.enqueue(2, Priority::Interactive).is_ok());
        // Now full; enqueue another interactive should drop oldest batch (id=1)
        assert!(q.enqueue(3, Priority::Interactive).is_ok());
        assert_eq!(q.snapshot_priority(Priority::Batch), Vec::<u32>::new());
        assert_eq!(q.snapshot_priority(Priority::Interactive), vec![2, 3]);
        assert_eq!(q.len(), 2);
    }

    // OC-CORE-1004: FIFO within the same priority class
    #[test]
    fn test_oc_core_1004_fifo_within_class() {
        let mut q = InMemoryQueue::with_capacity_policy(3, Policy::Reject);
        assert!(q.enqueue(10, Priority::Interactive).is_ok());
        assert!(q.enqueue(11, Priority::Interactive).is_ok());
        assert_eq!(q.snapshot_priority(Priority::Interactive), vec![10, 11]);
    }
}
