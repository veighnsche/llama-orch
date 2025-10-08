# Candle Integration Investigation - Handoff Document

**Created by:** TEAM-008  
**Date:** 2025-10-08  
**Purpose:** Investigation into using Candle components for llorch-cpud worker  
**For:** Next implementation team

---

## Executive Summary

After investigating Candle and Mistral.rs, we've identified a **hybrid approach** that gives us the best of both worlds:

**✅ RECOMMENDED:** Use Candle's low-level components, not the full framework  
**❌ NOT RECOMMENDED:** Using full Candle framework (too much abstraction)  
**✅ KEEP:** Your checkpoint-driven validation approach  
**✅ KEEP:** Your HTTP server architecture

---

## What is Candle?

Candle is Hugging Face's ML framework in Rust, similar to PyTorch but Rust-native.

**Repository:** `reference/candle/`

**Architecture:**
```
candle/
├── candle-core/          # Tensor operations, Device abstraction
├── candle-nn/            # Neural network layers (Linear, LayerNorm, etc.)
├── candle-kernels/       # CUDA kernels (CRITICAL - this is what we want!)
├── candle-transformers/  # Full model implementations
└── candle-flash-attn/    # Flash Attention kernels
```

---

## How Mistral.rs Uses Candle

**Repository:** `reference/mistral.rs/`

### Dependency Structure

**From `mistral.rs/Cargo.toml`:**
```toml
[workspace.dependencies]
candle-core = { git = "...", version = "0.9.1", rev = "7511e510" }
candle-nn = { git = "...", version = "0.9.1", rev = "7511e510" }
candle-flash-attn = { git = "...", optional = true }
```

**From `mistralrs-core/Cargo.toml`:**
```toml
[dependencies]
candle-core.workspace = true
candle-nn.workspace = true
candle-flash-attn = { workspace = true, optional = true }
mistralrs-quant.workspace = true  # Their own quantization layer
mistralrs-paged-attn.workspace = true  # Their own paged attention
```

### Key Insight: Layered Architecture

Mistral.rs uses Candle in **layers**:

1. **Bottom:** `candle-core` - Tensor operations, Device
2. **Middle:** `candle-nn` - Basic layers (Linear, RmsNorm, etc.)
3. **Top:** `mistralrs-quant` - Their own quantization (wraps Candle)
4. **Top:** `mistralrs-paged-attn` - Their own optimizations

**They DON'T use:** `candle-transformers` (too high-level)

---

## Candle Components Breakdown

### 1. `candle-core` - Tensor & Device Abstraction

**What it provides:**
```rust
use candle_core::{Device, Tensor, DType, Result};

// Device abstraction
let device = Device::cuda_if_available(0)?;
let device = Device::Cpu;
let device = Device::Metal(0)?;

// Tensor operations
let t = Tensor::new(&[1.0f32, 2.0, 3.0], &device)?;
let result = t.matmul(&other)?;
```

**Pros:**
- ✅ Unified device abstraction (CPU/CUDA/Metal)
- ✅ Automatic memory management
- ✅ Type-safe tensor operations

**Cons:**
- ❌ Adds abstraction layer
- ❌ Different from your ndarray approach

---

### 2. `candle-nn` - Neural Network Layers

**What it provides:**
```rust
use candle_nn::{Linear, RmsNorm, Embedding, Module};

// RmsNorm (from layers.rs:56-68)
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Candle's optimized RmsNorm implementation
    }
}

// Linear layer
let linear = Linear::new(weight, Some(bias));
let output = linear.forward(&input)?;
```

**Pros:**
- ✅ Production-tested implementations
- ✅ Optimized for Candle tensors
- ✅ Supports quantization

**Cons:**
- ❌ Requires using Candle's Tensor type
- ❌ Different API from your checkpoint approach

---

### 3. `candle-kernels` - **THE GOLDMINE** 🎯

**Location:** `reference/candle/candle-kernels/src/`

**What's inside:**
```
candle-kernels/src/
├── reduce.cu          # RmsNorm kernel (line 136!)
├── unary.cu           # SiLU, GELU, etc.
├── binary.cu          # Element-wise ops
├── quantized.cu       # Q8_0 dequantization (158KB!)
├── affine.cu          # Matrix operations
├── indexing.cu        # Tensor indexing
└── lib.rs             # Rust bindings
```

**RmsNorm kernel (from reduce.cu:133-136):**
```cuda
// RmsNorm implementation adapted from ggml, accumulation is made using f32.
// https://github.com/ggerganov/llama.cpp/blob/d59bd97065cd7ded6c4ecab54b1d5e0b1b11e318/ggml-cuda.cu#L523
template <typename T>
__device__ void rmsnorm(const T * x, T * dst, const T * alpha, const int ncols, const int block_size, const float eps) {
```

**This is EXACTLY what you want!**

**Pros:**
- ✅ Pure CUDA kernels (no framework)
- ✅ Optimized (from llama.cpp!)
- ✅ MIT licensed
- ✅ Can use directly with cudarc

**Cons:**
- ❌ Requires CUDA (but you're targeting NVIDIA anyway)
- ❌ Need to understand CUDA kernel invocation

---

### 4. `mistralrs-quant` - Quantization Layer

**What it provides:**
```rust
use mistralrs_quant::{
    QuantMethod,
    QuantizedConfig,
    ShardedVarBuilder,
    ColumnParallelLayer,
    RowParallelLayer,
};
```

**Key insight:** This is their **abstraction over Candle** for:
- GGUF quantization
- GPTQ quantization
- AWQ quantization
- Tensor parallelism

**They built this ON TOP of Candle, not using Candle's built-in quant**

---

## How Mistral.rs Implements Llama

**File:** `mistralrs-core/src/models/llama.rs`

### Model Structure

```rust
pub struct Llama {
    wte: Embedding,              // Token embeddings
    blocks: Vec<Block>,          // 32 transformer blocks
    ln_f: RmsNorm,               // Final RmsNorm
    lm_head: Arc<dyn QuantMethod>, // Output projection
    // ... config, device, etc.
}

struct Block {
    attn: CausalSelfAttention,   // Attention layer
    mlp: Mlp,                     // FFN layer
    attn_norm: RmsNorm,           // Pre-attention norm
    mlp_norm: RmsNorm,            // Pre-FFN norm
}

struct CausalSelfAttention {
    q_proj: Arc<dyn QuantMethod>,  // Separate Q projection
    k_proj: Arc<dyn QuantMethod>,  // Separate K projection
    v_proj: Arc<dyn QuantMethod>,  // Separate V projection
    o_proj: Arc<dyn QuantMethod>,  // Output projection
    rotary_emb: Arc<Llama3RotaryEmbedding>, // RoPE
    // ... attention params
}
```

### Weight Loading Pattern

**VarBuilder "cd" pattern:**
```rust
// From llama.rs:174-200
fn load(vb: ShardedVarBuilder, cfg: &Config) -> Result<Self> {
    // "cd" into q_proj weights
    let q_proj = ColumnParallelLayer::new(
        size_in,
        size_q,
        &cfg.quantization_config,
        false,
        comm,
        vb.pp("q_proj"),  // "cd q_proj" - loads "model.layers.0.q_proj.weight"
    )?;
    
    // "cd" into k_proj weights
    let k_proj = ColumnParallelLayer::new_with_shard(
        size_in,
        size_kv,
        &cfg.quantization_config,
        false,
        comm,
        vb.pp("k_proj"),  // "cd k_proj" - loads "model.layers.0.k_proj.weight"
    )?;
}
```

**This matches PyTorch's state_dict naming!**

---

## Three Integration Approaches

### Approach 1: Full Candle Framework ❌ NOT RECOMMENDED

**What:** Use `candle-core` + `candle-nn` + `candle-transformers`

```toml
[dependencies]
candle-core = "0.9"
candle-nn = "0.9"
candle-transformers = "0.9"
```

```rust
use candle_transformers::models::llama;

let model = llama::Llama::load(vb, &config)?;
let output = model.forward(&input)?;
```

**Pros:**
- ✅ Fastest to implement (models already exist)
- ✅ Production-tested
- ✅ Full ecosystem support

**Cons:**
- ❌ Too much abstraction (defeats your learning purpose)
- ❌ Loses your checkpoint validation
- ❌ Becomes "just another Candle user"
- ❌ Not aligned with your educational goals

**Verdict:** ❌ **Don't do this** - you wanted to build from scratch

---

### Approach 2: Candle Kernels Only ✅ RECOMMENDED

**What:** Use ONLY `candle-kernels` for optimized CUDA kernels

```toml
[dependencies]
candle-kernels = { path = "../../reference/candle/candle-kernels" }
cudarc = "0.11"  # CUDA runtime
ndarray = "0.15"  # Keep your tensor abstraction
```

```rust
// Your architecture
pub struct RMSNorm {
    weight: Array1<f32>,  // Your ndarray
    eps: f32,
}

impl RMSNorm {
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // Option A: Call Candle's CUDA kernel directly
        unsafe {
            // Use candle-kernels::REDUCE for RmsNorm
            // Your checkpoint validation still works!
        }
        
        // Option B: Keep ndarray for CPU, use kernel for GPU
        match device {
            Device::Cpu => self.forward_cpu(x),  // Your ndarray impl
            Device::Cuda => self.forward_cuda(x), // Candle kernel
        }
    }
}
```

**Pros:**
- ✅ Best performance (optimized CUDA kernels)
- ✅ Keep your architecture
- ✅ Keep your checkpoint validation
- ✅ Educational (you learn CUDA kernel invocation)
- ✅ Minimal dependencies

**Cons:**
- ❌ Need to learn CUDA kernel invocation
- ❌ More complex than pure ndarray
- ❌ CUDA-only (but that's your target anyway)

**Verdict:** ✅ **RECOMMENDED** - Best of both worlds

---

### Approach 3: Candle Core + NN ⚠️ MIDDLE GROUND

**What:** Use `candle-core` + `candle-nn`, but your own model architecture

```toml
[dependencies]
candle-core = "0.9"
candle-nn = "0.9"
# NOT candle-transformers
```

```rust
use candle_core::{Device, Tensor};
use candle_nn::{RmsNorm, Linear, Module};

pub struct Llama2Model {
    blocks: Vec<TransformerBlock>,
    // ... your architecture
}

impl Llama2Model {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Use Candle's RmsNorm, Linear, etc.
        // But YOUR model architecture
        // YOUR checkpoint validation
    }
}
```

**Pros:**
- ✅ Production-tested layers
- ✅ Keep your architecture
- ✅ Device abstraction (CPU/CUDA/Metal)
- ✅ Can still do checkpoint validation

**Cons:**
- ❌ Requires switching from ndarray to Candle Tensor
- ❌ More abstraction than kernels-only
- ❌ Larger dependency footprint

**Verdict:** ⚠️ **VIABLE** - Good middle ground if you want device abstraction

---

## Recommended Implementation Plan

### Phase 1: Validate Current Approach (Week 2)

**Keep using ndarray for now:**
```rust
// bin/llorch-cpud/src/layers/rms_norm.rs
use ndarray::{Array1, Array2};

pub struct RMSNorm {
    weight: Array1<f32>,
    eps: f32,
}

impl RMSNorm {
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // Pure ndarray implementation
        // Validate with Checkpoint 1
    }
}
```

**Why:** Prove your architecture works first

---

### Phase 2: Add Candle Kernels (Week 3-4)

**Add CUDA acceleration:**
```toml
[dependencies]
ndarray = "0.15"  # Keep for CPU
candle-kernels = { path = "../../reference/candle/candle-kernels", optional = true }
cudarc = { version = "0.11", optional = true }

[features]
cuda = ["candle-kernels", "cudarc"]
```

```rust
// src/layers/rms_norm.rs
pub struct RMSNorm {
    weight: Array1<f32>,
    eps: f32,
    #[cfg(feature = "cuda")]
    cuda_kernel: Option<CudaRmsNormKernel>,
}

impl RMSNorm {
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        #[cfg(feature = "cuda")]
        if let Some(kernel) = &self.cuda_kernel {
            return self.forward_cuda(x, kernel);
        }
        
        // Fallback to CPU (your ndarray impl)
        self.forward_cpu(x)
    }
    
    #[cfg(feature = "cuda")]
    fn forward_cuda(&self, x: &Array2<f32>, kernel: &CudaRmsNormKernel) -> Array2<f32> {
        // Use Candle's optimized CUDA kernel
        // Still validate with checkpoints!
    }
    
    fn forward_cpu(&self, x: &Array2<f32>) -> Array2<f32> {
        // Your original ndarray implementation
    }
}
```

**Why:** Get performance without losing validation

---

### Phase 3: Optimize (Future)

**Consider switching to Candle Tensor if needed:**
```rust
// Only if ndarray becomes a bottleneck
use candle_core::Tensor;

pub struct RMSNorm {
    weight: Tensor,
    eps: f32,
}
```

**Why:** Only if you need device abstraction or better performance

---

## Key Learnings from Mistral.rs

### 1. VarBuilder Pattern

**Candle's weight loading:**
```rust
// "cd" pattern for hierarchical weights
let vb = ShardedVarBuilder::from_safetensors(...);

// Load layer 0 weights
let layer0_vb = vb.pp("model").pp("layers").pp("0");
let q_proj = layer0_vb.pp("q_proj").get((4096, 4096), "weight")?;
// Loads: "model.layers.0.q_proj.weight"
```

**This matches PyTorch exactly!**

**Your GGUF approach is different:**
```rust
// You parse GGUF and get tensors by name
let q_weight = gguf.get_tensor("blk.0.attn_q.weight")?;
```

**Both work, just different naming conventions**

---

### 2. Quantization Abstraction

**Mistral.rs pattern:**
```rust
pub trait QuantMethod {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
    fn quantized_act_type(&self) -> Option<DType>;
    // ... etc
}

// Then:
let q_proj: Arc<dyn QuantMethod> = /* load based on config */;
let output = q_proj.forward(&input)?;
```

**This lets them support:**
- GGUF (Q8_0, Q4_0, etc.)
- GPTQ
- AWQ
- FP8
- Unquantized

**You could adopt this pattern!**

---

### 3. Separate Q, K, V Projections

**From llama.rs:52-56:**
```rust
struct CausalSelfAttention {
    q_proj: Arc<dyn QuantMethod>,  // Separate
    k_proj: Arc<dyn QuantMethod>,  // Separate
    v_proj: Arc<dyn QuantMethod>,  // Separate
    o_proj: Arc<dyn QuantMethod>,
}
```

**This matches Llama-2 exactly** (not combined like GPT-2)

---

### 4. RoPE Implementation

**From layers.rs (Llama3RotaryEmbedding):**
```rust
pub struct Llama3RotaryEmbedding(RotaryEmbedding);

impl Llama3RotaryEmbedding {
    pub fn forward(&self, q: &Tensor, k: &Tensor, seqlen_offsets: &[usize]) 
        -> Result<(Tensor, Tensor)> {
        // Apply RoPE to Q and K
        // NOT to V
    }
}
```

**You can reference this for your RoPE implementation!**

---

## Concrete Recommendations

### For Week 2 (Current)

**✅ DO:**
1. Keep using ndarray
2. Implement RMSNorm, RoPE, SwiGLU in pure Rust
3. Validate with checkpoints
4. Reference Mistral.rs for architecture patterns

**❌ DON'T:**
1. Add Candle dependencies yet
2. Try to use Candle Tensor
3. Switch away from your checkpoint approach

---

### For Week 3-4 (Optimization)

**✅ DO:**
1. Add `candle-kernels` as optional dependency
2. Implement CUDA path using Candle's kernels
3. Keep CPU path with ndarray
4. Validate both paths produce same checkpoints

**❌ DON'T:**
1. Use full Candle framework
2. Lose your checkpoint validation
3. Switch to Candle Tensor unless necessary

---

### For Future (Production)

**✅ CONSIDER:**
1. Candle Tensor for device abstraction
2. `mistralrs-quant` patterns for quantization
3. Flash Attention from `candle-flash-attn`
4. Paged Attention patterns

**❌ DON'T:**
1. Use `candle-transformers` (too high-level)
2. Lose your architectural understanding
3. Become dependent on Candle's model implementations

---

## Files to Reference

### In Mistral.rs

**Model architecture:**
- `mistralrs-core/src/models/llama.rs` - Llama implementation
- `mistralrs-core/src/layers.rs` - RmsNorm, RoPE, etc.

**Quantization:**
- `mistralrs-quant/src/` - Quantization abstraction
- `mistralrs-quant/src/gguf/` - GGUF support

**Attention:**
- `mistralrs-core/src/attention/` - Attention implementations
- `mistralrs-paged-attn/src/` - Paged attention

### In Candle

**Kernels (THE GOLDMINE):**
- `candle-kernels/src/reduce.cu` - RmsNorm kernel
- `candle-kernels/src/unary.cu` - SiLU, GELU
- `candle-kernels/src/quantized.cu` - Quantization kernels

**Core:**
- `candle-core/src/` - Tensor, Device
- `candle-nn/src/` - Neural network layers

---

## Decision Matrix

| Approach | Speed | Control | Learning | Complexity | Recommended |
|----------|-------|---------|----------|------------|-------------|
| **Full Candle** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐ | ❌ No |
| **Kernels Only** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ **YES** |
| **Core + NN** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⚠️ Maybe |
| **Pure ndarray** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ✅ For now |

---

## Example: RMSNorm with Candle Kernels

```rust
// bin/llorch-cpud/src/layers/rms_norm.rs
// Created by: TEAM-008
// Uses: Candle kernels for CUDA, ndarray for CPU

use ndarray::{Array1, Array2};

#[cfg(feature = "cuda")]
use candle_kernels::REDUCE;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice};

pub struct RMSNorm {
    weight: Array1<f32>,
    eps: f32,
    
    #[cfg(feature = "cuda")]
    device: Option<CudaDevice>,
}

impl RMSNorm {
    pub fn new(weight: Array1<f32>, eps: f32) -> Self {
        Self {
            weight,
            eps,
            #[cfg(feature = "cuda")]
            device: CudaDevice::new(0).ok(),
        }
    }
    
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        #[cfg(feature = "cuda")]
        if let Some(device) = &self.device {
            return self.forward_cuda(x, device);
        }
        
        // CPU fallback (your original implementation)
        self.forward_cpu(x)
    }
    
    fn forward_cpu(&self, x: &Array2<f32>) -> Array2<f32> {
        // Your ndarray implementation
        // Validated with Checkpoint 1
        let mean_square = x.mapv(|v| v * v).mean_axis(Axis(1)).unwrap();
        let rms = (mean_square + self.eps).mapv(f32::sqrt);
        let normalized = x / &rms.insert_axis(Axis(1));
        normalized * &self.weight.insert_axis(Axis(0))
    }
    
    #[cfg(feature = "cuda")]
    fn forward_cuda(&self, x: &Array2<f32>, device: &CudaDevice) -> Array2<f32> {
        // Use Candle's optimized RmsNorm kernel
        // Still produces same output as CPU version
        // Still validates with Checkpoint 1
        
        // 1. Copy to GPU
        let x_gpu = device.htod_copy(x.as_slice().unwrap()).unwrap();
        let weight_gpu = device.htod_copy(self.weight.as_slice().unwrap()).unwrap();
        
        // 2. Call Candle's RmsNorm kernel
        unsafe {
            // Use REDUCE module from candle-kernels
            // This is the optimized CUDA kernel
        }
        
        // 3. Copy back to CPU
        let result = device.dtoh_sync_copy(&result_gpu).unwrap();
        Array2::from_shape_vec(x.dim(), result).unwrap()
    }
}
```

---

## Next Steps for Implementation Team

### Immediate (Week 2)

1. **Continue with pure ndarray**
   - Implement RMSNorm (Checkpoint 1)
   - Implement RoPE (Checkpoint 1B)
   - Implement separate QKV (Checkpoint 2)
   - Implement SwiGLU (Checkpoint 6)

2. **Reference Mistral.rs for patterns**
   - Look at `llama.rs` for architecture
   - Look at `layers.rs` for RoPE implementation
   - Don't copy code, understand patterns

3. **Validate everything**
   - Use checkpoint-driven validation
   - Compare with llama.cpp
   - Ensure correctness before optimization

### Short-term (Week 3-4)

1. **Add Candle kernels (optional)**
   - Add `candle-kernels` dependency
   - Implement CUDA path
   - Keep CPU path for validation
   - Feature-gate with `cuda` feature

2. **Performance comparison**
   - Benchmark CPU vs CUDA
   - Validate both produce same checkpoints
   - Document performance gains

### Long-term (Future)

1. **Consider Candle Tensor**
   - If device abstraction needed
   - If performance critical
   - Only if justified

2. **Adopt quantization patterns**
   - Study `mistralrs-quant`
   - Implement `QuantMethod` trait
   - Support multiple quant formats

3. **Production optimizations**
   - Flash Attention
   - Paged Attention
   - Tensor parallelism

---

## Conclusion

**The Answer:** Use Candle's **kernels**, not the **framework**

**Why:**
- ✅ Get optimized CUDA kernels (from llama.cpp!)
- ✅ Keep your architecture and learning
- ✅ Keep your checkpoint validation
- ✅ Best performance without abstraction overhead

**How:**
- Week 2: Pure ndarray, validate everything
- Week 3-4: Add Candle kernels for CUDA acceleration
- Future: Consider Candle Tensor if needed

**Don't:**
- ❌ Use full Candle framework
- ❌ Use `candle-transformers`
- ❌ Lose your checkpoint-driven approach

---

## Sign-off

**Investigated by:** TEAM-008  
**Date:** 2025-10-08  
**Status:** Investigation complete, recommendations clear

**Key Finding:** Candle kernels are the sweet spot - optimized math without framework overhead.

**Next Team:** Continue with ndarray for Week 2, consider Candle kernels for Week 3-4 optimization.

---

*"Use the tools, don't become the tools."*  
— TEAM-008, Foundation Implementation Division

**END HANDOFF**
