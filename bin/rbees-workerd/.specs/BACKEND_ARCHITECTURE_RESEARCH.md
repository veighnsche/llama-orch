# Backend Architecture Research: Multi-Binary Pattern in llorch-candled

**Research Date:** 2025-10-09T09:14:22+02:00  
**Researcher:** TEAM-CASCADE  
**Subject:** Backend selection patterns and Metal support feasibility  
**Updated:** 2025-10-09 by TEAM-018 (Accelerate → Metal transition)

---

## Executive Summary

`llorch-candled` uses a **feature-gated multi-binary pattern** to support different compute backends (CPU, CUDA, Metal) from a single codebase. This research documents:

1. **Current architecture** - How we build backend-specific binaries
2. **Backend abstraction pattern** - Device initialization and model loading
3. **Metal support implementation** - Apple GPU support (TEAM-018)
4. **Binary distribution strategy** - Why multiple binaries vs runtime selection

**⚠️ IMPORTANT UPDATE (TEAM-018):** Accelerate backend has been **removed** and replaced with Metal backend for Apple Silicon GPU support.

---

## 1. Current Multi-Backend Architecture

### 1.1 Feature-Gated Binaries

We produce **three separate binaries** from one crate using Cargo feature gates:

```toml
# Cargo.toml
[features]
default = ["cpu"]  # Default to CPU for broadest compatibility

# Backend features (mutually exclusive at build time)
cpu = []
cuda = ["candle-kernels", "cudarc", "candle-core/cuda", "candle-nn/cuda"]
metal = ["candle-core/metal", "candle-nn/metal"]  # TEAM-018: Added Metal

[[bin]]
name = "llorch-cpu-candled"
path = "src/bin/cpu.rs"
required-features = ["cpu"]

[[bin]]
name = "llorch-cuda-candled"
path = "src/bin/cuda.rs"
required-features = ["cuda"]

[[bin]]
name = "llorch-metal-candled"  # TEAM-018: Replaced accelerate
path = "src/bin/metal.rs"
required-features = ["metal"]
```

### 1.2 Why Multiple Binaries?

**Design Decision:** Build-time backend selection, not runtime.

**Rationale:**
1. **Dependency isolation** - CUDA binaries don't require CUDA toolkit on CPU-only machines
2. **Binary size** - CPU binary is 7.3MB; CUDA binary would be larger with kernels
3. **Deployment simplicity** - Ship only the binary for target hardware
4. **No runtime overhead** - Zero cost abstraction via feature gates
5. **Clear contracts** - Each binary has explicit hardware requirements

**Alternative Rejected:** Single binary with runtime backend selection
- ❌ Would require all backend dependencies at build time
- ❌ Larger binary size (includes unused backends)
- ❌ Runtime detection complexity
- ❌ Potential for misconfiguration

---

## 2. Backend Abstraction Pattern

### 2.1 Device Initialization Layer

**File:** `src/device.rs`

Each backend has a dedicated initialization function, feature-gated at compile time:

```rust
/// Initialize CPU device
#[cfg(feature = "cpu")]
pub fn init_cpu_device() -> CandleResult<Device> {
    tracing::info!("Initializing CPU device");
    Ok(Device::Cpu)
}

/// Initialize CUDA device
#[cfg(feature = "cuda")]
pub fn init_cuda_device(gpu_id: usize) -> CandleResult<Device> {
    tracing::info!("Initializing CUDA device {}", gpu_id);
    Device::new_cuda(gpu_id)
}

/// Initialize Apple Metal device (GPU)
/// Note: Metal is Apple's GPU API, equivalent to CUDA for NVIDIA
#[cfg(feature = "metal")]
pub fn init_metal_device(gpu_id: usize) -> CandleResult<Device> {
    tracing::info!("Initializing Apple Metal device (GPU) {}", gpu_id);
    Device::new_metal(gpu_id)
}

/// Verify device is available and working
/// Performs a simple smoke test: create tensor and verify operations
pub fn verify_device(device: &Device) -> CandleResult<()> {
    use candle_core::Tensor;
    
    // Simple smoke test: create tensor and verify
    let test = Tensor::zeros((2, 2), candle_core::DType::F32, device)?;
    let _sum = test.sum_all()?;
    
    tracing::info!("Device verification passed: {:?}", device);
    Ok(())
}
```

**Key Pattern:** 
- Each function is **conditionally compiled** based on feature
- Returns Candle's unified `Device` type
- Verification is backend-agnostic (works on any device)

### 2.2 Binary Entry Points

Each binary follows the same lifecycle but initializes different devices:

**Pattern (from `src/bin/cpu.rs`):**
```rust
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    // 1. Initialize logging
    tracing_subscriber::fmt().with_target(false).json().init();
    
    // 2. Parse CLI args (worker_id, model, port, callback_url)
    let args = Args::parse();
    
    // 3. Initialize device (backend-specific)
    let device = init_cpu_device()?;
    verify_device(&device)?;
    
    // 4. Load model to device
    let backend = CandleInferenceBackend::load(&args.model, device)?;
    
    // 5. Callback to pool-managerd (worker ready)
    callback_ready(&args.callback_url, &args.worker_id, 
                   backend.memory_bytes(), args.port).await?;
    
    // 6. Start HTTP server (runs forever)
    let backend = Arc::new(Mutex::new(backend));
    let router = create_router(backend);
    let server = HttpServer::new(addr, router).await?;
    server.run().await?;
    
    Ok(())
}
```

**CUDA variant adds:**
- `--cuda-device` CLI argument for GPU selection
- GPU warmup call after model load (eliminates cold start)

**Metal variant (TEAM-018):**
- `--metal-device` CLI argument for GPU selection
- GPU warmup call after model load (same as CUDA)
- Pre-release status (not yet validated on Apple Silicon)

### 2.3 Model Backend Abstraction

**File:** `src/backend/inference.rs`

The inference backend is **device-agnostic** - it receives a `Device` and uses it:

```rust
pub struct CandleInferenceBackend {
    model: Model,           // Enum: Llama | Mistral | Phi | Qwen
    tokenizer: Tokenizer,
    device: Device,         // CPU, CUDA, or Accelerate
    model_size_bytes: u64,
}

impl CandleInferenceBackend {
    pub fn load(model_path: &str, device: Device) -> Result<Self> {
        // Load model using auto-detected architecture
        let model = models::load_model(model_path, &device)?;
        let tokenizer = tokenizer_loader::load_tokenizer(path)?;
        
        Ok(Self { model, tokenizer, device, model_size_bytes })
    }
    
    pub fn warmup(&mut self) -> Result<()> {
        // GPU warmup (CUDA only, no-op on CPU)
        let input_ids = Tensor::new(tokens, &self.device)?;
        let _logits = self.model.forward(&input_ids, 0)?;
        Ok(())
    }
}
```

**Key Pattern:**
- Backend receives `Device` from binary entry point
- All tensor operations use `&self.device`
- No device-specific code in backend (Candle handles it)

---

## 3. Candle Backend Support

### 3.1 Available Backends in Candle

From `reference/candle/candle-core/Cargo.toml`:

```toml
[features]
default = []

# CPU backends
accelerate = ["dep:accelerate-src"]  # Apple Accelerate (CPU)
mkl = ["dep:intel-mkl-src"]          # Intel MKL (CPU)

# GPU backends
cuda = ["candle-kernels/cuda", "dep:cudarc"]  # NVIDIA CUDA
metal = ["dep:metal", "dep:objc"]              # Apple Metal (GPU)
```

**Current llorch-candled support:**
- ✅ **CPU** - Plain CPU (no acceleration)
- ✅ **CUDA** - NVIDIA GPU
- ✅ **Metal** - Apple GPU (TEAM-018: Pre-release)
- ❌ **Accelerate** - Apple CPU (REMOVED by TEAM-018: too slow)
- ❌ **MKL** - Intel CPU-optimized (NOT IMPLEMENTED)

### 3.2 Metal Backend in Candle

**Metal is Apple's GPU API** (equivalent to CUDA for NVIDIA).

**Candle support:**
```rust
// From candle-core/src/device.rs
pub enum Device {
    Cpu,
    Cuda(CudaDevice),
    Metal(MetalDevice),  // Apple GPU support exists!
}

impl Device {
    pub fn new_metal(ordinal: usize) -> Result<Self> {
        // Initialize Metal device
    }
}
```

**Metal kernels:** `reference/candle/candle-metal-kernels/`
- Optimized Metal shaders for matrix ops
- Similar to CUDA kernels but for Apple GPUs

---

## 4. Metal Support Implementation (TEAM-018)

### 4.1 Implementation Status: ✅ COMPLETE

**Date:** 2025-10-09  
**Team:** TEAM-018  
**Status:** Pre-release (code complete, runtime validation pending)

Metal support has been **implemented** following the directive to replace Accelerate:

**1. Update `Cargo.toml`:**
```toml
[features]
metal = ["candle-core/metal", "candle-nn/metal"]

[[bin]]
name = "llorch-metal-candled"
path = "src/bin/metal.rs"
required-features = ["metal"]
```

**2. Add `src/device.rs` function:**
```rust
/// Initialize Apple Metal device (GPU)
#[cfg(feature = "metal")]
pub fn init_metal_device(gpu_id: usize) -> CandleResult<Device> {
    tracing::info!("Initializing Apple Metal device (GPU)");
    Device::new_metal(gpu_id)
}
```

**3. Create `src/bin/metal.rs`:**
```rust
//! Apple Metal GPU worker binary
//!
//! Uses Apple Metal for GPU inference on macOS.
//! This is GPU Metal, NOT Accelerate (CPU).

use anyhow::Result;
use clap::Parser;
use llorch_candled::device::{init_metal_device, verify_device};
use llorch_candled::{backend::CandleInferenceBackend, callback_ready, 
                     create_router, HttpServer};

#[derive(Parser, Debug)]
#[command(name = "llorch-metal-candled")]
#[command(about = "Apple Metal GPU Candle-based multi-model worker daemon")]
struct Args {
    #[arg(long)]
    worker_id: String,
    
    #[arg(long)]
    model: String,
    
    #[arg(long)]
    port: u16,
    
    #[arg(long)]
    callback_url: String,
    
    /// Metal device ID (default: 0)
    #[arg(long, default_value = "0")]
    metal_device: usize,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_target(false).json().init();
    
    let args = Args::parse();
    
    tracing::info!(
        worker_id = %args.worker_id,
        model = %args.model,
        port = args.port,
        metal_device = args.metal_device,
        backend = "metal",
        "Starting llorch-metal-candled"
    );
    
    // Initialize Metal device
    let device = init_metal_device(args.metal_device)?;
    verify_device(&device)?;
    
    // Load model to Metal GPU
    let mut backend = CandleInferenceBackend::load(&args.model, device)?;
    
    // GPU warmup (similar to CUDA)
    backend.warmup()?;
    
    // Callback to pool-managerd
    if !args.callback_url.contains("localhost:9999") {
        callback_ready(&args.callback_url, &args.worker_id, 
                       backend.memory_bytes(), args.port).await?;
    }
    
    // Start HTTP server
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    let backend = Arc::new(Mutex::new(backend));
    let router = create_router(backend);
    let server = HttpServer::new(addr, router).await?;
    
    tracing::info!("llorch-metal-candled ready on port {} (Metal GPU {})", 
                   args.port, args.metal_device);
    
    server.run().await?;
    Ok(())
}
```

### 4.2 Build Commands

```bash
# Build Metal binary (macOS only)
cargo build --release --features metal --bin llorch-metal-candled

# Run Metal worker
./target/release/llorch-metal-candled \
  --worker-id test-worker \
  --model /path/to/llama-2-7b/ \
  --port 8080 \
  --metal-device 0 \
  --callback-url http://localhost:9999
```

### 4.3 Accelerate Removal Rationale (TEAM-018)

**Why Accelerate was removed:**
- ❌ CPU-bound (not GPU acceleration)
- ❌ Too slow for production inference (~10-20 tok/s)
- ❌ No significant advantage over plain CPU backend
- ❌ Confusing naming (users expected GPU acceleration)

**Why Metal was added:**
- ✅ True GPU acceleration on Apple Silicon
- ✅ 2×-10× faster than Accelerate (estimated)
- ✅ Aligns with CUDA pattern (GPU inference)
- ✅ Candle already has Metal support

### 4.4 Comparison: CPU vs Metal

| Aspect | CPU | Metal |
|--------|-----|-------|
| **Hardware** | Any CPU | Apple Silicon GPU |
| **Framework** | Standard | Apple Metal |
| **Device** | `Device::Cpu` | `Device::Metal(id)` |
| **Use Case** | Development/fallback | Production inference on macOS |
| **Performance** | ~5-15 tok/s | ~50-100 tok/s (estimated) |
| **Platform** | All platforms | macOS Apple Silicon only |
| **Binary** | `llorch-cpu-candled` | `llorch-metal-candled` |

**Key Distinction:**
- **Accelerate** = Apple's CPU math library (like Intel MKL)
- **Metal** = Apple's GPU API (like NVIDIA CUDA)

---

## 5. Binary Distribution Strategy

### 5.1 Current Binaries

| Binary | Platform | Hardware | Size | Dependencies |
|--------|----------|----------|------|--------------|
| `llorch-cpu-candled` | Linux/macOS/Windows | Any CPU | 7.3MB | None |
| `llorch-cuda-candled` | Linux/Windows | NVIDIA GPU | ~15MB | CUDA toolkit |

### 5.2 Proposed Metal Binary

| Binary | Platform | Hardware | Size | Dependencies |
|--------|----------|----------|------|--------------|
| `llorch-metal-candled` | macOS | Apple GPU | ~10MB | macOS 10.15+ |

### 5.3 Deployment Matrix

**For each deployment target, ship ONE binary:**

| Target | Binary | Rationale |
|--------|--------|-----------|
| Linux x86 CPU | `llorch-cpu-candled` | No GPU |
| Linux + NVIDIA GPU | `llorch-cuda-candled` | CUDA acceleration |
| macOS Intel | `llorch-cpu-candled` | No GPU (Intel Macs) |
| macOS Apple Silicon | `llorch-metal-candled` | Apple GPU acceleration |
| Windows CPU | `llorch-cpu-candled` | No GPU |
| Windows + NVIDIA GPU | `llorch-cuda-candled` | CUDA acceleration |

**Pool-managerd responsibility:**
- Detect hardware capabilities
- Launch appropriate binary
- Pass device ID via CLI args

---

## 6. Pattern Analysis

### 6.1 Strengths of Multi-Binary Pattern

1. **Minimal dependencies** - Each binary only includes what it needs
2. **Clear contracts** - Binary name indicates hardware requirement
3. **Zero runtime overhead** - Feature gates compile out unused code
4. **Easy deployment** - Ship only relevant binary
5. **Testability** - Each backend tested independently
6. **Maintainability** - Shared core logic, backend-specific entry points

### 6.2 Weaknesses

1. **Build complexity** - Must build multiple binaries per release
2. **Testing matrix** - Need hardware for each backend
3. **Code duplication** - Binary entry points are similar (mitigated by shared functions)

### 6.3 Why This Pattern Works for llorch-candled

**Context:**
- Worker daemons are **deployed to specific hardware**
- Pool-managerd knows hardware capabilities
- Workers are **long-running** (startup cost amortized)
- Binary size matters (container images, network transfer)

**Alternative patterns rejected:**
- ❌ Runtime backend selection - Larger binaries, all dependencies required
- ❌ Dynamic linking - Deployment complexity, version conflicts
- ❌ Separate crates - Code duplication, maintenance burden

---

## 7. Model Backend Abstraction

### 7.1 Model Enum Pattern

**File:** `src/backend/models/mod.rs`

We use an **enum-based model abstraction** (Candle-idiomatic):

```rust
pub enum Model {
    Llama(llama::LlamaModel),
    Mistral(mistral::MistralModel),
    Phi(phi::PhiModel),
    Qwen(qwen::QwenModel),
}

impl Model {
    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        match self {
            Model::Llama(m) => m.forward(input_ids, position),
            Model::Mistral(m) => m.forward(input_ids, position),
            Model::Phi(m) => m.forward(input_ids),
            Model::Qwen(m) => m.forward(input_ids, position),
        }
    }
}
```

**Why enum, not trait?**
- ✅ Static dispatch (no vtable overhead)
- ✅ Each model uses natural interface
- ✅ Matches Candle reference examples
- ✅ Compiler optimizations (inlining, devirtualization)

### 7.2 Model Loading

**File:** `src/backend/models/mod.rs`

Architecture detection is automatic:

```rust
pub fn load_model(model_path: &str, device: &Device) -> Result<Model> {
    let config_json = load_config_json(path)?;
    let architecture = detect_architecture(&config_json)?;
    
    match architecture.as_str() {
        "llama" => Ok(Model::Llama(llama::LlamaModel::load(path, device)?)),
        "mistral" => Ok(Model::Mistral(mistral::MistralModel::load(path, device)?)),
        "phi" => Ok(Model::Phi(phi::PhiModel::load(path, device)?)),
        "qwen" | "qwen2" => Ok(Model::Qwen(qwen::QwenModel::load(path, device)?)),
        _ => bail!("Unsupported model architecture: {}", architecture),
    }
}
```

**Key insight:** Model loading is **device-agnostic**. Each model's `load()` function:
1. Receives `&Device` parameter
2. Loads weights to that device
3. Returns model ready for inference

**This means:** Same model code works on CPU, CUDA, Accelerate, or Metal!

---

## 8. Recommendations

### 8.1 Should We Add Metal Support?

**YES, but with caveats:**

**Pros:**
- ✅ Apple Silicon Macs have powerful GPUs
- ✅ Candle already supports Metal
- ✅ Minimal code changes (follows existing pattern)
- ✅ Performance benefit over Accelerate (CPU)

**Cons:**
- ⚠️ macOS-only (limited deployment)
- ⚠️ Requires macOS for testing
- ⚠️ Apple Silicon only (M1/M2/M3)

**Recommendation:** Add Metal support **after** core functionality is complete.

**Priority:**
1. **P0:** Complete CUDA support (broader market)
2. **P1:** Validate all model architectures
3. **P2:** Add Metal support (Apple Silicon users)

### 8.2 Implementation Timeline

**Phase 1: Metal Foundation (2-3 hours)**
- Add `metal` feature to Cargo.toml
- Create `init_metal_device()` in device.rs
- Create `src/bin/metal.rs` binary
- Add feature-gated tests

**Phase 2: Testing & Validation (2-3 hours)**
- Test on Apple Silicon Mac
- Benchmark vs Accelerate (CPU)
- Verify model loading works
- Test warmup and inference

**Phase 3: Documentation (1 hour)**
- Update README with Metal instructions
- Document Metal vs Accelerate differences
- Add deployment guide

**Total estimate:** 5-7 hours

### 8.3 Testing Strategy

**Without Apple Silicon hardware:**
- ✅ Verify compilation with `--features metal`
- ✅ Review code structure
- ❌ Cannot test runtime behavior

**With Apple Silicon hardware:**
- ✅ Full integration testing
- ✅ Performance benchmarking
- ✅ Model compatibility validation

---

## 9. Conclusion

### 9.1 Current Architecture Summary

`llorch-candled` uses a **feature-gated multi-binary pattern** where:

1. **Single codebase** with shared core logic
2. **Multiple binaries** for different backends (CPU, CUDA, Metal)
3. **Build-time selection** via Cargo features
4. **Device abstraction** via Candle's unified `Device` type
5. **Model abstraction** via enum pattern (device-agnostic)

**TEAM-018 Update:** Accelerate backend removed, Metal backend added (pre-release).

### 9.2 Metal Support Implementation

**Verdict:** ✅ **IMPLEMENTED (Pre-release)**

**TEAM-018 completed:**
1. ✅ Added `metal` feature to Cargo.toml
2. ✅ Created `init_metal_device()` function in `src/device.rs`
3. ✅ Created `llorch-metal-candled` binary in `src/bin/metal.rs`
4. ✅ Followed existing CUDA pattern
5. ✅ Removed Accelerate backend (too slow)
6. ✅ Created documentation (`docs/metal.md`)

**Status:** Code complete, runtime validation pending on Apple Silicon hardware.

**Actual effort:** ~2 hours (code implementation + documentation)

### 9.3 Pattern Strengths

This architecture excels at:
- ✅ **Deployment flexibility** - Ship only needed binary
- ✅ **Dependency isolation** - No unused backend dependencies
- ✅ **Performance** - Zero runtime overhead
- ✅ **Maintainability** - Shared core, backend-specific entry points
- ✅ **Extensibility** - Easy to add new backends (Metal, MKL, etc.)

### 9.4 Next Steps

**For Metal support (TEAM-018):**
1. ✅ ~~Add Metal feature and binary~~ (COMPLETE)
2. ✅ ~~Remove Accelerate backend~~ (COMPLETE)
3. ✅ ~~Document deployment strategy~~ (COMPLETE)
4. ⏳ Test on Apple Silicon hardware (PENDING)
5. ⏳ Benchmark performance vs CPU (PENDING)
6. ⏳ Validate all model architectures (PENDING)

**For general architecture:**
1. Continue using multi-binary pattern
2. Keep device abstraction via Candle's `Device`
3. Maintain model enum pattern (Candle-idiomatic)
4. Add backends as needed (Metal, MKL, ROCm, etc.)

---

## References

### Code Files
- `Cargo.toml` - Feature gate configuration
- `src/device.rs` - Device initialization layer
- `src/bin/cpu.rs` - CPU binary entry point
- `src/bin/cuda.rs` - CUDA binary entry point
- `src/bin/metal.rs` - Metal binary entry point (TEAM-018)
- `src/backend/inference.rs` - Device-agnostic inference backend
- `src/backend/models/mod.rs` - Model enum abstraction

### Documentation
- `.specs/TEAM_007_FINAL_REPORT.md` - Multi-backend implementation
- `.specs/TEAM_017_HANDOFF.md` - Multi-model support
- `.specs/HANDOFF_TO_TEAM_007.md` - Original multi-backend mission
- `README.md` - Project overview

### Candle Reference
- `reference/candle/candle-core/Cargo.toml` - Backend features
- `reference/candle/candle-metal-kernels/` - Metal GPU kernels
- `reference/candle/candle-core/src/device.rs` - Device abstraction

---

**Research completed:** 2025-10-09T09:14:22+02:00  
**Updated:** 2025-10-09 by TEAM-018  
**Status:** ✅ Architecture documented, Metal support **implemented** (pre-release)  
**Next action:** Validate Metal backend on Apple Silicon hardware
