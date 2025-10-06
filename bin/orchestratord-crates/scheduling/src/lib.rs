//! placement — Worker selection and placement logic
//!
//! Selects optimal worker for job based on model, capacity, locality, and load.
//!
//! # Key Responsibilities
//!
//! - Worker selection strategies (round-robin, least-loaded, locality-aware)
//! - Capacity checking (slots available, VRAM sufficient)
//! - Model affinity (prefer workers with model already loaded)
//! - Load balancing across workers
//! - Placement constraints (GPU type, region)
//!
//! # Example
//!
//! ```rust
//! use placement::{PlacementEngine, PlacementStrategy, WorkerInfo};
//!
//! let engine = PlacementEngine::new(PlacementStrategy::LeastLoaded);
//!
//! let workers = vec![/* worker info */];
//! let selected = engine.select_worker(&workers, model_ref, required_vram)?;
//! ```

// High-importance crate: TIER 2 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![warn(clippy::indexing_slicing)]
#![warn(clippy::arithmetic_side_effects)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::missing_errors_doc)]

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PlacementError {
    #[error("no workers available")]
    NoWorkersAvailable,
    #[error("insufficient capacity")]
    InsufficientCapacity,
}

pub type Result<T> = std::result::Result<T, PlacementError>;

/// Placement strategy
#[derive(Debug, Clone, Copy)]
pub enum PlacementStrategy {
    RoundRobin,
    LeastLoaded,
    LocalityAware,
}

/// Worker information
#[derive(Debug, Clone)]
pub struct WorkerInfo {
    pub worker_id: String,
    pub slots_total: u32,
    pub slots_free: u32,
    pub vram_total: u64,
    pub vram_free: u64,
    pub models_loaded: Vec<String>,
}

/// Placement engine
pub struct PlacementEngine {
    strategy: PlacementStrategy,
    round_robin_index: usize,
}

impl PlacementEngine {
    pub fn new(strategy: PlacementStrategy) -> Self {
        Self { strategy, round_robin_index: 0 }
    }

    /// Select worker for job
    pub fn select_worker(
        &mut self,
        workers: &[WorkerInfo],
        _model_ref: &str,
        _required_vram: u64,
    ) -> Result<String> {
        if workers.is_empty() {
            return Err(PlacementError::NoWorkersAvailable);
        }

        let selected = match self.strategy {
            PlacementStrategy::RoundRobin => {
                let idx = self.round_robin_index % workers.len();
                self.round_robin_index = self.round_robin_index.wrapping_add(1);
                workers.get(idx).ok_or(PlacementError::NoWorkersAvailable)?
            }
            PlacementStrategy::LeastLoaded => workers
                .iter()
                .max_by_key(|w| w.slots_free)
                .ok_or(PlacementError::NoWorkersAvailable)?,
            PlacementStrategy::LocalityAware => {
                // TODO: Implement locality-aware selection
                workers.first().ok_or(PlacementError::NoWorkersAvailable)?
            }
        };

        if selected.slots_free == 0 {
            return Err(PlacementError::InsufficientCapacity);
        }

        Ok(selected.worker_id.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_placement() {
        let mut engine = PlacementEngine::new(PlacementStrategy::LeastLoaded);

        let workers = vec![
            WorkerInfo {
                worker_id: "worker-1".to_string(),
                slots_total: 4,
                slots_free: 2,
                vram_total: 24_000_000_000,
                vram_free: 12_000_000_000,
                models_loaded: vec![],
            },
            WorkerInfo {
                worker_id: "worker-2".to_string(),
                slots_total: 4,
                slots_free: 3,
                vram_total: 24_000_000_000,
                vram_free: 18_000_000_000,
                models_loaded: vec![],
            },
        ];

        let selected = engine.select_worker(&workers, "model", 1_000_000_000).ok();
        assert_eq!(selected, Some("worker-2".to_string())); // Least loaded
    }
}
