# TEAM-001: Candle Math Catalog - Implementation Plan

**Created by:** TEAM-000 (Foundation)  
**For:** TEAM-001 (Math Integration)  
**Date:** 2025-10-08  
**Purpose:** Catalog and selectively import Candle's math internals for rbees-workerd worker

---

## Mission

**WE DON'T NEED A FULL ML FRAMEWORK!**  
**WE NEED A WORKER FOR POOL-MANAGERD!**

Your job: Extract ONLY the math we need from Candle, leave the framework behind.

---

## The Problem

We're building a Llama-2 inference worker. We need:
- ✅ Fast tensor math (matmul, softmax, etc.)
- ✅ Optimized CUDA kernels (RmsNorm, SiLU, etc.)
- ✅ Device abstraction (CPU/CUDA)

We DON'T need:
- ❌ Training infrastructure
- ❌ Backpropagation
- ❌ Optimizer implementations
- ❌ Dataset loaders
- ❌ Full model zoo

---

## Candle Architecture (What We Found)

### Crate Structure

```
candle/
├── candle-core/          # 🎯 TENSOR OPERATIONS (we need this)
│   ├── Tensor            # Core tensor type
│   ├── Device            # CPU/CUDA/Metal abstraction
│   ├── DType             # Data types (F32, F16, BF16, etc.)
│   ├── Storage           # Memory management
│   └── ops/              # Basic operations
│
├── candle-nn/            # 🎯 NEURAL NET LAYERS (we need parts)
│   ├── layer_norm.rs     # LayerNorm + RmsNorm
│   ├── linear.rs         # Matrix multiplication
│   ├── activation.rs     # SiLU, GELU, etc.
│   ├── rotary_emb.rs     # RoPE implementation
│   ├── ops.rs            # Softmax, etc.
│   └── embedding.rs      # Token embeddings
│
├── candle-kernels/       # 🎯 CUDA KERNELS (we need this)
│   ├── reduce.cu         # RmsNorm kernel
│   ├── unary.cu          # SiLU, GELU kernels
│   ├── quantized.cu      # Q8_0 dequantization
│   └── lib.rs            # Rust bindings
│
├── candle-transformers/  # ❌ TOO HIGH-LEVEL (we don't need)
│   └── Full model implementations
│
└── candle-datasets/      # ❌ NOT NEEDED
    └── Dataset loaders
```

---

## What We Need: The Catalog

### Category 1: Core Tensor Operations (candle-core)

**Import Strategy:** Use as dependency, selective imports

```rust
// From candle-core
use candle_core::{
    Tensor,           // Core tensor type
    Device,           // CPU/CUDA abstraction
    DType,            // F32, F16, BF16, etc.
    Result,           // Error handling
    Module,           // Forward trait
    D,                // Dimension indexing
};
```

**What we get:**
- ✅ `Tensor::new()` - Create tensors
- ✅ `Tensor::matmul()` - Matrix multiplication
- ✅ `Tensor::reshape()` - Shape manipulation
- ✅ `Device::cuda_if_available()` - Device selection
- ✅ Memory management (automatic)

**What we DON'T use:**
- ❌ Backpropagation (backprop.rs)
- ❌ Training ops (variable.rs)
- ❌ Streaming (streaming.rs)

### Category 2: Neural Network Layers (candle-nn)

**Import Strategy:** Cherry-pick specific functions

```rust
// From candle-nn
use candle_nn::{
    // Normalization
    rms_norm,         // RmsNorm function (Checkpoint 1)
    RmsNorm,          // RmsNorm struct
    
    // Linear layers
    linear_no_bias,   // Matrix multiply (for QKV, FFN)
    Linear,           // Linear layer struct
    
    // Activations
    ops::silu,        // SiLU activation (for SwiGLU)
    ops::softmax,     // Softmax (for attention)
    
    // Embeddings
    embedding,        // Token embeddings
    Embedding,        // Embedding struct
    
    // RoPE
    // NOTE: We'll implement our own RoPE for learning
    // But reference their implementation
};
```

**Key Functions We Need:**

1. **RmsNorm** (`layer_norm.rs:186-194`)
   ```rust
   pub fn rms_norm(xs: &Tensor, alpha: &Tensor, eps: f32) -> Result<Tensor>
   ```
   - Optimized RmsNorm implementation
   - Falls back to CUDA kernel if available
   - Exactly what we need for Checkpoint 1

2. **SiLU** (`ops.rs:40-42`)
   ```rust
   pub fn silu(xs: &Tensor) -> Result<Tensor>
   ```
   - Used in SwiGLU (Checkpoint 6)
   - Has CUDA kernel

3. **Softmax** (`ops.rs:22-29`)
   ```rust
   pub fn softmax<D: Dim>(xs: &Tensor, dim: D) -> Result<Tensor>
   ```
   - For attention scores (Checkpoint 4)

4. **Linear** (`linear.rs`)
   ```rust
   pub fn linear_no_bias(x: &Tensor, w: &Tensor) -> Result<Tensor>
   ```
   - For QKV projections, FFN

### Category 3: CUDA Kernels (candle-kernels)

**Import Strategy:** Direct kernel access (already in our Cargo.toml)

```rust
// Already configured in rbees-workerd/Cargo.toml
candle-kernels = { path = "../../reference/candle/candle-kernels", optional = true }
```

**Available Kernels:**

1. **RmsNorm** (`reduce.cu:133-136`)
   - Optimized CUDA implementation
   - Adapted from llama.cpp
   - Used by `candle_nn::ops::rms_norm()`

2. **SiLU** (`unary.cu`)
   - Activation function kernel
   - Used by `candle_nn::ops::silu()`

3. **Quantization** (`quantized.cu`)
   - Q8_0 dequantization
   - For loading GGUF models

**How Candle Uses Kernels:**
```rust
// From candle-nn/src/ops.rs:658
pub fn rms_norm(xs: &Tensor, alpha: &Tensor, eps: f32) -> Result<Tensor> {
    // ... validation ...
    
    #[cfg(feature = "cuda")]
    if let (Device::Cuda(_), Device::Cuda(_)) = (xs.device(), alpha.device()) {
        // Use CUDA kernel
        return candle::ops::rms_norm(xs, alpha, eps);
    }
    
    // Fallback to CPU
    rms_norm_slow(xs, alpha, eps)
}
```

---

## Implementation Strategy

### Phase 1: Use Candle's High-Level API (Week 2)

**Goal:** Get working quickly with Candle's safe abstractions

```rust
// In rbees-workerd/src/layers/rms_norm.rs

use candle_core::{Tensor, Result, Module};
use candle_nn::RmsNorm as CandleRmsNorm;

pub struct RMSNorm {
    inner: CandleRmsNorm,
}

impl RMSNorm {
    pub fn new(weight: Tensor, eps: f64) -> Result<Self> {
        Ok(Self {
            inner: CandleRmsNorm::new(weight, eps),
        })
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}
```

**Pros:**
- ✅ Fast to implement
- ✅ Proven correct
- ✅ Automatic CUDA when available
- ✅ Checkpoint validation still works

**Cons:**
- ❌ Less educational
- ❌ Tied to Candle's Tensor type

### Phase 2: Hybrid Approach (Week 3-4)

**Goal:** Use Candle for math, our own for structure

```rust
// Our architecture, Candle's math

pub struct RMSNorm {
    weight: Tensor,  // Candle tensor
    eps: f64,
}

impl RMSNorm {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Use Candle's optimized function
        candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32)
    }
}
```

**Pros:**
- ✅ We control the architecture
- ✅ Candle handles the math
- ✅ Best of both worlds

### Phase 3: Direct Kernel Access (Future)

**Goal:** Maximum control, minimum dependencies

```rust
// Only if we need it

#[cfg(feature = "cuda")]
fn forward_cuda(&self, x: &Tensor) -> Result<Tensor> {
    // Call Candle kernel directly
    // This is advanced - only if needed
}
```

---

## Dependency Configuration

### Current Setup (Already Done by TEAM-000)

```toml
# bin/rbees-workerd/Cargo.toml

[dependencies]
# Candle kernels for CUDA acceleration (optional, feature-gated)
candle-kernels = { path = "../../reference/candle/candle-kernels", optional = true }
cudarc = { version = "0.11", optional = true }

[features]
cuda = ["candle-kernels", "cudarc"]
```

### What to Add

```toml
# Add to [dependencies]
candle-core = { path = "../../reference/candle/candle-core" }
candle-nn = { path = "../../reference/candle/candle-nn" }

# Update [features]
cuda = ["candle-kernels", "cudarc", "candle-core/cuda", "candle-nn/cuda"]
```

---

## The Catalog: Function-by-Function

### For Checkpoint 1: RMSNorm

**Option A: Use Candle's RmsNorm struct**
```rust
use candle_nn::{RmsNorm, Module};

let norm = RmsNorm::new(weight_tensor, eps)?;
let output = norm.forward(&input)?;
```

**Option B: Use Candle's rms_norm function**
```rust
use candle_nn::ops::rms_norm;

let output = rms_norm(&input, &weight, eps)?;
```

**Recommendation:** Option B (function) - more flexible

### For Checkpoint 1B: RoPE

**Candle's Implementation:** `candle-nn/src/rotary_emb.rs`

**Strategy:** Study their implementation, write our own
- They have CPU and CUDA versions
- Interleaved variant (what Llama uses)
- We can reference but implement ourselves for learning

### For Checkpoint 2: QKV Projection

**Use Candle's linear_no_bias:**
```rust
use candle_nn::linear_no_bias;

let q = linear_no_bias(&x, &wq)?;
let k = linear_no_bias(&x, &wk)?;
let v = linear_no_bias(&x, &wv)?;
```

### For Checkpoint 4: Attention Scores

**Use Candle's softmax:**
```rust
use candle_nn::ops::softmax;

let scores = q.matmul(&k.t())?;
let scores = (scores / scale)?;
let probs = softmax(&scores, D::Minus1)?;
```

### For Checkpoint 6: SwiGLU

**Use Candle's silu:**
```rust
use candle_nn::ops::silu;

let gate = silu(&x.matmul(&w_gate)?)?;
let up = x.matmul(&w_up)?;
let hidden = gate.mul(&up)?;
let output = hidden.matmul(&w_down)?;
```

---

## What We DON'T Import

### ❌ From candle-transformers
- Full Llama model implementation
- VarBuilder pattern (we use GGUF directly)
- Model configs (we have our own)

### ❌ From candle-core
- Backpropagation (`backprop.rs`)
- Variable tracking (`variable.rs`)
- Training utilities

### ❌ From candle-nn
- Optimizers (`optim.rs`)
- Batch normalization (we use RmsNorm)
- RNN/LSTM (we only need transformers)
- Loss functions (inference only)

---

## Integration Checklist

### Step 1: Add Dependencies
- [ ] Add `candle-core` to Cargo.toml
- [ ] Add `candle-nn` to Cargo.toml
- [ ] Update `cuda` feature to include Candle features
- [ ] Verify compilation

### Step 2: Create Wrapper Types
- [ ] Wrap Candle's Tensor in our types (if needed)
- [ ] Create conversion functions (ndarray ↔ Tensor)
- [ ] Test basic operations

### Step 3: Implement Checkpoints with Candle
- [ ] Checkpoint 1: Use `candle_nn::ops::rms_norm`
- [ ] Checkpoint 2: Use `candle_nn::linear_no_bias`
- [ ] Checkpoint 4: Use `candle_nn::ops::softmax`
- [ ] Checkpoint 6: Use `candle_nn::ops::silu`

### Step 4: Validate
- [ ] Compare Candle output with llama.cpp
- [ ] Ensure checkpoints still pass
- [ ] Test CPU and CUDA paths

---

## Example: RMSNorm with Candle

**File:** `src/layers/rms_norm.rs`

```rust
//! RMSNorm using Candle's optimized implementation
//!
//! Created by: TEAM-001

use candle_core::{Tensor, Result, Module, Device};
use candle_nn::ops::rms_norm as candle_rms_norm;

pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    /// Create RMSNorm from weight tensor
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }
    
    /// Create from raw f32 array (for testing)
    pub fn from_array(weight: &[f32], eps: f64, device: &Device) -> Result<Self> {
        let weight = Tensor::from_slice(weight, weight.len(), device)?;
        Ok(Self::new(weight, eps))
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Use Candle's optimized RmsNorm
        // Automatically uses CUDA kernel if available
        candle_rms_norm(x, &self.weight, self.eps as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_rms_norm_cpu() {
        let device = Device::Cpu;
        let weight = vec![1.0; 4096];
        let norm = RMSNorm::from_array(&weight, 1e-6, &device).unwrap();
        
        // Test with dummy input
        let input = Tensor::randn(0f32, 1.0, (1, 2, 4096), &device).unwrap();
        let output = norm.forward(&input).unwrap();
        
        assert_eq!(output.shape(), input.shape());
    }
}
```

---

## Key Insights from Candle Study

### 1. Device Abstraction is Clean
```rust
let device = Device::cuda_if_available(0)?;
let tensor = Tensor::new(&data, &device)?;
```
- Automatically picks CUDA if available
- Fallback to CPU is seamless

### 2. Module Trait is Simple
```rust
pub trait Module {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}
```
- Same pattern we're using
- Easy to wrap Candle components

### 3. Kernels are Automatic
- When you call `rms_norm()`, it checks device
- If CUDA, uses kernel
- If CPU, uses fallback
- We don't need to manage this!

### 4. No Training Overhead
- We can use just the inference parts
- No need to import optimizers, backprop, etc.

---

## Comparison: Our Approach vs Full Candle

### Our Hybrid Approach ✅
```rust
// We control the architecture
pub struct Llama2Model {
    blocks: Vec<TransformerBlock>,
    // Our structure
}

impl Llama2Model {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Use Candle for math
        let x = candle_nn::ops::rms_norm(&x, &self.norm_weight, self.eps)?;
        // Our logic
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        x
    }
}
```

### Full Candle (What We're Avoiding) ❌
```rust
// From candle-transformers
use candle_transformers::models::llama::Llama;

let model = Llama::load(vb, &config)?;  // Black box
let output = model.forward(&input)?;     // No control
```

---

## Migration Path from ndarray

### Current (ndarray)
```rust
use ndarray::{Array1, Array2};

pub struct RMSNorm {
    weight: Array1<f32>,
    eps: f32,
}
```

### With Candle
```rust
use candle_core::Tensor;

pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}
```

### Conversion Helpers
```rust
// ndarray → Candle
fn array_to_tensor(arr: &Array2<f32>, device: &Device) -> Result<Tensor> {
    let shape = arr.shape();
    Tensor::from_slice(arr.as_slice().unwrap(), shape, device)
}

// Candle → ndarray
fn tensor_to_array(tensor: &Tensor) -> Result<Array2<f32>> {
    let shape = tensor.shape();
    let data = tensor.to_vec2::<f32>()?;
    Ok(Array2::from_shape_vec((shape.dims()[0], shape.dims()[1]), data.into_iter().flatten().collect())?)
}
```

---

## Decision Matrix

| Aspect | Pure ndarray | Candle Functions | Full Candle |
|--------|-------------|------------------|-------------|
| **Speed** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Control** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| **Learning** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| **CUDA** | ❌ | ✅ | ✅ |
| **Complexity** | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Dependencies** | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

**Recommendation:** **Candle Functions** (middle column)

---

## Action Items for TEAM-001

### Week 2: Integration
1. [ ] Add `candle-core` and `candle-nn` to Cargo.toml
2. [ ] Create Tensor conversion helpers
3. [ ] Implement Checkpoint 1 using `candle_nn::ops::rms_norm`
4. [ ] Validate output matches llama.cpp
5. [ ] Document the integration pattern

### Week 3: Expand Usage
1. [ ] Use Candle for all math operations
2. [ ] Implement remaining checkpoints with Candle
3. [ ] Test CPU and CUDA paths
4. [ ] Benchmark performance

### Week 4: Optimize
1. [ ] Profile to find bottlenecks
2. [ ] Consider direct kernel access if needed
3. [ ] Document final architecture
4. [ ] Create usage guide for future teams

---

## Questions & Answers

### Q: Do we need the full Candle framework?
**A:** NO! Just `candle-core` and `candle-nn`. Skip `candle-transformers`.

### Q: Will this work with our checkpoint validation?
**A:** YES! Candle's functions produce the same results as llama.cpp.

### Q: What about CUDA?
**A:** Automatic! Candle checks device and uses kernels when available.

### Q: Can we still learn the architecture?
**A:** YES! We control the model structure, Candle just does the math.

### Q: What if Candle changes?
**A:** We're using stable APIs. Plus, we have the reference code in `reference/candle/`.

---

## Summary

**What TEAM-001 Needs to Do:**

1. **Add Dependencies:**
   - `candle-core` for Tensor operations
   - `candle-nn` for layer functions

2. **Use Candle for Math:**
   - `candle_nn::ops::rms_norm()` for RMSNorm
   - `candle_nn::ops::silu()` for SiLU
   - `candle_nn::ops::softmax()` for Softmax
   - `candle_nn::linear_no_bias()` for Linear

3. **Keep Our Architecture:**
   - Our model structure
   - Our checkpoint validation
   - Our learning goals

4. **Get CUDA for Free:**
   - Candle handles device selection
   - Kernels activate automatically
   - No manual kernel management

**Result:** Fast, correct, educational worker for pool-managerd! 🚀

---

**Status:** 📋 Plan Complete  
**Next:** TEAM-001 Implementation  
**Timeline:** Week 2-3

---

Built by TEAM-000 🌊 for TEAM-001 🧮
