//! batching — worker-orcd batching scheduler (skeleton)
//!
//! Scope: coalescing, fairness, cancel-aware decode loop per resident handle.
//! Engine specifics belong in adapter crates implementing `BatchDecodeEngine`.

#![forbid(unsafe_code)]

use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;
use std::time::{Duration, Instant};

// Basic types (preview; refine as needed)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SequenceId(pub String);

#[derive(Clone, Debug)]
pub struct SequenceCtx {
    pub id: SequenceId,
    pub arrived_at: Instant,
    pub cancelled: bool,
}

#[derive(Clone, Debug, Default)]
pub struct SeqMetrics {
    pub batch_wait_ms: u32,
}

#[derive(Clone, Debug, Default)]
pub struct StepOutput {
    pub tokens: Vec<(SequenceId, String)>,
    pub finished: Vec<SequenceId>,
    pub errors: Vec<(SequenceId, &'static str, String)>,
}

pub trait BatchDecodeEngine {
    type HandleId: Clone + Eq + Hash + Debug;

    fn supports_continuous_batching(&self, handle: &Self::HandleId) -> bool;

    /// Perform one decode step over the active sequences for `handle`.
    fn decode_step(
        &self,
        handle: &Self::HandleId,
        active: &mut [SequenceCtx],
    ) -> StepOutput;
}

pub trait BatchAdapterEvents {
    fn on_started(&self, seq: &SequenceId);
    fn on_token(&self, seq: &SequenceId, token: String);
    fn on_metrics(&self, seq: &SequenceId, m: SeqMetrics);
    fn on_end(&self, seq: &SequenceId);
    fn on_error(&self, seq: &SequenceId, code: &'static str, msg: String);
}

#[derive(Clone, Debug)]
pub struct BatchConfig {
    pub batch_window_ms: u32,
    pub max_batch_size: u32,
    pub per_handle_queue_max: u32,
    pub per_handle_concurrency_cap: u32, // 0 = unlimited
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_window_ms: 10,
            max_batch_size: 8,
            per_handle_queue_max: 256,
            per_handle_concurrency_cap: 0,
        }
    }
}

pub struct BatchScheduler<E: BatchDecodeEngine, A: BatchAdapterEvents> {
    engine: E,
    events: A,
    cfg: BatchConfig,
    // per-handle queues and running sets
    ready: HashMap<E::HandleId, VecDeque<SequenceCtx>>, // FIFO
    running: HashMap<E::HandleId, Vec<SequenceCtx>>,    // active set
}

impl<E: BatchDecodeEngine, A: BatchAdapterEvents> BatchScheduler<E, A> {
    pub fn new(engine: E, events: A, cfg: BatchConfig) -> Self {
        Self { engine, events, cfg, ready: HashMap::new(), running: HashMap::new() }
    }

    pub fn enqueue(&mut self, handle: E::HandleId, seq: SequenceCtx) -> Result<(), &'static str> {
        let q = self.ready.entry(handle).or_default();
        if (q.len() as u32) >= self.cfg.per_handle_queue_max {
            return Err("QUEUE_FULL");
        }
        // Admission timestamp is set by caller; ensure not cancelled
        if seq.cancelled { return Err("CANCELLED"); }
        q.push_back(seq);
        Ok(())
    }

    /// Perform one scheduling tick for a given handle: admit a batch and run one decode step.
    pub fn tick_handle(&mut self, handle: &E::HandleId) {
        let running = self.running.entry(handle.clone()).or_default();
        let ready = self.ready.entry(handle.clone()).or_default();

        // Admit from ready into running respecting caps
        let cap = self.cfg.per_handle_concurrency_cap;
        let max_admit = if cap == 0 {
            self.cfg.max_batch_size as usize
        } else {
            (cap as usize).saturating_sub(running.len()).min(self.cfg.max_batch_size as usize)
        };
        for _ in 0..max_admit {
            if let Some(mut seq) = ready.pop_front() {
                self.events.on_started(&seq.id);
                // compute batch wait
                let waited = seq.arrived_at.elapsed().as_millis() as u32;
                self.events.on_metrics(&seq.id, SeqMetrics { batch_wait_ms: waited });
                running.push(seq);
            } else { break; }
        }

        if running.is_empty() { return; }

        // One decode step
        let out = self.engine.decode_step(handle, running);
        for (seq, tok) in out.tokens { self.events.on_token(&seq, tok); }
        for (seq, code, msg) in out.errors { self.events.on_error(&seq, code, msg); }

        // Remove finished/cancelled
        if !out.finished.is_empty() || running.iter().any(|s| s.cancelled) {
            running.retain(|s| {
                let done = out.finished.iter().any(|f| *f == s.id) || s.cancelled;
                if done {
                    if s.cancelled { /* on_error likely already emitted by adapter */ } else { self.events.on_end(&s.id); }
                }
                !done
            });
        }
    }
}
