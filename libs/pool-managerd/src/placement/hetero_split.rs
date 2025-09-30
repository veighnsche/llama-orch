//! Heterogeneous split planner (planning-only).
//!
//! TODO: Implement tensor split ratio planning for heterogeneous multi-GPU setups.
//! Spec: ORCH-3052 (cross-GPU tensor splits opt-in via tensor_split ratios; respect smallest GPU's VRAM).
//! Checklist: CHECKLIST.md "Device Discovery & Snapshots" → "Optional: MIG partitions; present a stable mask per pool."
//! Usage: Called during pool initialization when multiple GPUs with different VRAM are configured.
//! Expected API:
//!   - `SplitPlan::compute(devices: &[DeviceSnapshot], model_size_bytes: u64) -> Result<Self>`
//!   - `SplitPlan::ratios(&self) -> &[f32]` (per-GPU split ratios, sum to 1.0)
//!   - `SplitPlan::validate(&self) -> Result<()>` (ensure smallest GPU can hold its share)
//! Integration: Used by preload module to configure engine with --tensor-split flags (llama.cpp).
//! Tests: BDD step "per-GPU resident KV is capped for smallest GPU" (test-harness/bdd/src/steps/pool_manager.rs:64).
//! Note: Default assumes no split (single GPU); this is opt-in only.

#[derive(Debug, Clone)]
pub struct SplitPlan;
