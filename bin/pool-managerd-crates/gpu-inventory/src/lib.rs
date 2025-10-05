//! GPU inventory tracking for pool manager.
//!
//! Tracks VRAM capacity across multiple GPUs and worker allocations.
//! Used by pool manager to determine GPU placement for new workers.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Multi-GPU VRAM capacity tracker.
#[derive(Debug, Clone)]
pub struct GpuInventory {
    gpus: HashMap<u32, GpuState>,
}

/// State of a single GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuState {
    pub device_id: u32,
    pub total_vram_bytes: u64,
    pub allocated_vram_bytes: u64,
    pub workers: Vec<WorkerAllocation>,
}

/// Worker VRAM allocation on a GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerAllocation {
    pub worker_id: String,
    pub model_ref: String,
    pub vram_bytes: u64,
}

impl GpuInventory {
    /// Create inventory for given GPUs.
    pub fn new(gpu_capacities: Vec<(u32, u64)>) -> Self {
        let gpus = gpu_capacities
            .into_iter()
            .map(|(device_id, total_vram_bytes)| {
                (
                    device_id,
                    GpuState {
                        device_id,
                        total_vram_bytes,
                        allocated_vram_bytes: 0,
                        workers: Vec::new(),
                    },
                )
            })
            .collect();

        Self { gpus }
    }

    /// Get available VRAM on a specific GPU.
    pub fn available_vram(&self, gpu_id: u32) -> Option<u64> {
        self.gpus
            .get(&gpu_id)
            .map(|gpu| gpu.total_vram_bytes.saturating_sub(gpu.allocated_vram_bytes))
    }

    /// Find a GPU that can fit the given model size.
    /// Returns GPU ID with most free VRAM.
    pub fn can_fit_model(&self, model_size: u64) -> Option<u32> {
        self.gpus
            .values()
            .filter(|gpu| {
                let available = gpu.total_vram_bytes.saturating_sub(gpu.allocated_vram_bytes);
                available >= model_size
            })
            .max_by_key(|gpu| gpu.total_vram_bytes.saturating_sub(gpu.allocated_vram_bytes))
            .map(|gpu| gpu.device_id)
    }

    /// Register a worker allocation.
    pub fn register_worker(
        &mut self,
        worker_id: String,
        model_ref: String,
        gpu_id: u32,
        vram_bytes: u64,
    ) {
        if let Some(gpu) = self.gpus.get_mut(&gpu_id) {
            gpu.allocated_vram_bytes = gpu.allocated_vram_bytes.saturating_add(vram_bytes);
            gpu.workers.push(WorkerAllocation { worker_id, model_ref, vram_bytes });
        }
    }

    /// Unregister a worker allocation.
    pub fn unregister_worker(&mut self, worker_id: &str) {
        for gpu in self.gpus.values_mut() {
            if let Some(pos) = gpu.workers.iter().position(|w| w.worker_id == worker_id) {
                let removed = gpu.workers.remove(pos);
                gpu.allocated_vram_bytes =
                    gpu.allocated_vram_bytes.saturating_sub(removed.vram_bytes);
            }
        }
    }

    /// Get snapshot of all GPUs.
    pub fn snapshot(&self) -> Vec<GpuState> {
        let mut states: Vec<_> = self.gpus.values().cloned().collect();
        states.sort_by_key(|gpu| gpu.device_id);
        states
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_can_fit_model_selects_gpu_with_most_free_vram() {
        let inventory = GpuInventory::new(vec![
            (0, 24_000_000_000), // 24GB
            (1, 24_000_000_000), // 24GB
        ]);

        // 10GB model should fit on either, picks GPU 0 (tie-break)
        assert_eq!(inventory.can_fit_model(10_000_000_000), Some(0));
    }

    #[test]
    fn test_register_worker_updates_allocation() {
        let mut inventory = GpuInventory::new(vec![(0, 24_000_000_000)]);

        inventory.register_worker(
            "worker-1".to_string(),
            "llama-7b".to_string(),
            0,
            14_000_000_000,
        );

        assert_eq!(inventory.available_vram(0), Some(10_000_000_000));
    }

    #[test]
    fn test_unregister_worker_frees_vram() {
        let mut inventory = GpuInventory::new(vec![(0, 24_000_000_000)]);

        inventory.register_worker(
            "worker-1".to_string(),
            "llama-7b".to_string(),
            0,
            14_000_000_000,
        );
        inventory.unregister_worker("worker-1");

        assert_eq!(inventory.available_vram(0), Some(24_000_000_000));
    }
}
