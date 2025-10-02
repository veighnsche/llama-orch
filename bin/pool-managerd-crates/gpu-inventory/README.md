# gpu-inventory

**Multi-GPU capacity tracking and worker placement queries**

## Purpose

Pool manager crate for tracking VRAM capacity across all GPUs and determining which GPU can fit a new worker.

## Responsibilities

- **Track GPU state**: Total/used VRAM per GPU
- **Worker allocations**: Which workers are using which GPU + how much VRAM
- **Placement queries**: `can_fit_model(size) -> Option<gpu_id>`
- **Capacity reporting**: Provide state snapshots to orchestrator

## NOT Responsible For

- ❌ Scheduling decisions (orchestrator does this)
- ❌ Worker lifecycle (lifecycle crate does this)
- ❌ VRAM enforcement (worker's vram-policy does this)

## API

```rust
pub struct GpuInventory {
    gpus: Vec<GpuState>,
}

pub struct GpuState {
    device_id: u32,
    total_vram_bytes: u64,
    allocated_vram_bytes: u64,
    workers: Vec<WorkerAllocation>,
}

pub struct WorkerAllocation {
    worker_id: String,
    model_ref: String,
    vram_bytes: u64,
}

impl GpuInventory {
    pub fn available_vram(&self, gpu_id: u32) -> u64;
    pub fn can_fit_model(&self, model_size: u64) -> Option<u32>;
    pub fn register_worker(&mut self, worker_id: String, gpu_id: u32, vram: u64);
    pub fn unregister_worker(&mut self, worker_id: &str);
    pub fn snapshot(&self) -> Vec<GpuState>;
}
```

## Data Flow

```
Orchestrator: "Can you fit llama-13b (26GB)?"
         ↓
Pool Manager (gpu-inventory):
  GPU 0: 24GB total, 16GB used → 8GB free  ❌
  GPU 1: 24GB total, 4GB used  → 20GB free ✅
         ↓
Response: "Yes, GPU 1"
```

## Status

- **Version**: 0.0.0 (stub, not implemented)
- **License**: GPL-3.0-or-later
