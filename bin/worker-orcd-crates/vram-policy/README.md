# vram-policy

**VRAM-only enforcement for single-model workers**

## Purpose

Worker-side crate to enforce VRAM-only policy for a single model loaded at startup.

## Responsibilities

- **VRAM-only enforcement**: Disable UMA, zero-copy, pinned host memory
- **Load model to VRAM**: Allocate and copy model bytes to GPU
- **Verify residency**: Health checks that model stays in VRAM
- **Report usage**: Return actual VRAM bytes used

## NOT Responsible For

- ❌ Multi-model orchestration (worker is tied to ONE model)
- ❌ Placement decisions (orchestrator does this)
- ❌ VRAM capacity tracking across GPUs (pool manager's gpu-inventory does this)

## API

```rust
pub struct VramPolicy {
    gpu_device: u32,
}

impl VramPolicy {
    pub fn new(gpu_device: u32) -> Result<Self>;
    pub fn enforce_vram_only(&self) -> Result<()>;
    pub fn load_model_to_vram(&self, model_bytes: &[u8]) -> Result<u64>;
    pub fn verify_vram_residency(&self) -> Result<()>;
}
```

## Data Flow

```
Worker startup:
  1. VramPolicy::new(gpu_device)
  2. enforce_vram_only() → Disable UMA/zero-copy
  3. load_model_to_vram(model_bytes) → Returns actual VRAM used
  4. Report to pool manager: "Ready, using 26GB VRAM"

Worker health check:
  verify_vram_residency() → Ensure no RAM fallback
```

## VRAM-Only Policy

**Enforced at runtime:**
- ✅ Model weights MUST reside entirely in GPU VRAM during inference
- ✅ Unified memory (UMA) disabled
- ✅ Zero-copy and pinned host memory disabled
- ✅ Fail fast if VRAM capacity insufficient

## Status

- **Version**: 0.0.0 (stub, not implemented)
- **License**: GPL-3.0-or-later
