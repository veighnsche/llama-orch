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
    pub fn with_capacity_policy(capacity: usize, policy: Policy) -> Self {
        Self {
            interactive: VecDeque::new(),
            batch: VecDeque::new(),
            capacity,
            policy,
        }
    }

    pub fn len(&self) -> usize {
        self.interactive.len() + self.batch.len()
    }

    pub fn is_empty(&self) -> bool {
        self.interactive.is_empty() && self.batch.is_empty()
    }

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

    pub fn snapshot_priority(&self, prio: Priority) -> Vec<u32> {
        match prio {
            Priority::Interactive => self.interactive.iter().copied().collect(),
            Priority::Batch => self.batch.iter().copied().collect(),
        }
    }
}
