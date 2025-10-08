# Quick Start: TEAM-001 Candle Integration

**TL;DR:** Use Candle for math, keep our architecture. Simple as that.

---

## The 3-Step Plan

### 1. Add Dependencies (5 minutes)

```toml
# bin/llorch-candled/Cargo.toml

[dependencies]
# Add these two lines:
candle-core = { path = "../../reference/candle/candle-core" }
candle-nn = { path = "../../reference/candle/candle-nn" }

# Update cuda feature:
[features]
cuda = ["candle-kernels", "cudarc", "candle-core/cuda", "candle-nn/cuda"]
```

### 2. Use Candle Functions (Per Checkpoint)

```rust
// Checkpoint 1: RMSNorm
use candle_nn::ops::rms_norm;
let output = rms_norm(&input, &weight, eps)?;

// Checkpoint 2: QKV
use candle_nn::linear_no_bias;
let q = linear_no_bias(&x, &wq)?;

// Checkpoint 4: Softmax
use candle_nn::ops::softmax;
let probs = softmax(&scores, D::Minus1)?;

// Checkpoint 6: SwiGLU
use candle_nn::ops::silu;
let gate = silu(&x.matmul(&w_gate)?)?;
```

### 3. Keep Our Structure

```rust
// Our model, Candle's math
pub struct Llama2Model {
    blocks: Vec<TransformerBlock>,  // Our structure
}

impl Llama2Model {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Use Candle functions for math
        let x = rms_norm(&x, &self.norm, eps)?;
        
        // Our logic
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        x
    }
}
```

---

## What to Import

### ✅ DO Import
- `candle_core::Tensor` - Core tensor type
- `candle_core::Device` - CPU/CUDA selection
- `candle_nn::ops::rms_norm` - RMSNorm function
- `candle_nn::ops::silu` - SiLU activation
- `candle_nn::ops::softmax` - Softmax function
- `candle_nn::linear_no_bias` - Matrix multiply

### ❌ DON'T Import
- `candle_transformers::*` - Full models (too high-level)
- `candle_nn::optim::*` - Optimizers (we don't train)
- `candle_core::backprop::*` - Backprop (inference only)

---

## Example: Checkpoint 1 with Candle

```rust
//! RMSNorm using Candle
//! Created by: TEAM-001

use candle_core::{Tensor, Result, Device};
use candle_nn::ops::rms_norm;

pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // That's it! Candle handles CPU/CUDA automatically
        rms_norm(x, &self.weight, self.eps as f32)
    }
}
```

---

## CUDA: Automatic!

```rust
// Just create device
let device = Device::cuda_if_available(0)?;

// Candle picks the right kernel
let output = rms_norm(&input, &weight, eps)?;
// ↑ Uses CUDA kernel if device is CUDA
// ↑ Uses CPU fallback if device is CPU
```

No manual kernel management needed!

---

## Full Catalog (Copy-Paste Ready)

```rust
// At top of file
use candle_core::{Tensor, Device, Result, Module, D};
use candle_nn::ops::{rms_norm, silu, softmax};
use candle_nn::linear_no_bias;

// Checkpoint 1: RMSNorm
fn apply_rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    rms_norm(x, weight, eps)
}

// Checkpoint 2: Linear (no bias)
fn apply_linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    linear_no_bias(x, w)
}

// Checkpoint 4: Softmax
fn apply_softmax(x: &Tensor) -> Result<Tensor> {
    softmax(x, D::Minus1)
}

// Checkpoint 6: SiLU
fn apply_silu(x: &Tensor) -> Result<Tensor> {
    silu(x)
}

// Checkpoint 6: SwiGLU
fn apply_swiglu(x: &Tensor, w_gate: &Tensor, w_up: &Tensor, w_down: &Tensor) -> Result<Tensor> {
    let gate = silu(&x.matmul(w_gate)?)?;
    let up = x.matmul(w_up)?;
    let hidden = gate.mul(&up)?;
    hidden.matmul(w_down)
}
```

---

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_with_candle() {
        let device = Device::Cpu;
        
        // Create tensors
        let weight = Tensor::ones((4096,), &device).unwrap();
        let input = Tensor::randn(0f32, 1.0, (1, 2, 4096), &device).unwrap();
        
        // Use Candle function
        let output = rms_norm(&input, &weight, 1e-6).unwrap();
        
        // Validate
        assert_eq!(output.shape(), input.shape());
    }
}
```

---

## Migration from ndarray

If you have ndarray code, convert to Candle:

```rust
// OLD (ndarray)
use ndarray::Array2;
let arr = Array2::zeros((10, 20));

// NEW (Candle)
use candle_core::Tensor;
let tensor = Tensor::zeros((10, 20), &device)?;
```

---

## Troubleshooting

### "Cannot find candle_core"
→ Add to Cargo.toml dependencies

### "CUDA not available"
→ That's fine! Candle falls back to CPU automatically

### "Tensor shape mismatch"
→ Check dimensions, Candle is strict about shapes

---

## Next Steps

1. **Add dependencies** (see step 1 above)
2. **Implement Checkpoint 1** with `rms_norm()`
3. **Validate** against llama.cpp
4. **Continue** through checkpoints using catalog

---

## Full Documentation

See `TEAM_001_CANDLE_CATALOG_PLAN.md` for:
- Complete function catalog
- Architecture decisions
- Advanced usage
- Performance tips

---

**Status:** 🚀 Ready to implement  
**Difficulty:** Easy (Candle does the hard work)  
**Timeline:** Week 2

Built by TEAM-000 🌊 for TEAM-001 🧮
