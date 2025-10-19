# TEAM-133: Reusability Matrix

**Analysis of which crates can be reused by future worker types**

---

## WORKER TYPES (FUTURE)

1. **llm-worker-rbee** (current) - Text generation
2. **embedding-worker-rbee** (future) - Text embeddings
3. **vision-worker-rbee** (future) - Image generation/classification
4. **audio-worker-rbee** (future) - Audio generation/transcription
5. **multimodal-worker-rbee** (future) - Multi-modal models

---

## REUSABILITY MATRIX

| Crate | LLM | Embedding | Vision | Audio | Multimodal | Reusability |
|-------|-----|-----------|--------|-------|------------|-------------|
| **worker-rbee-error** | ✅ 100% | ✅ 100% | ✅ 100% | ✅ 100% | ✅ 100% | **100%** |
| **worker-rbee-startup** | ✅ 100% | ✅ 100% | ✅ 100% | ✅ 100% | ✅ 100% | **100%** |
| **worker-rbee-health** | ✅ 100% | ✅ 100% | ✅ 100% | ✅ 100% | ✅ 100% | **100%** |
| **worker-rbee-sse-streaming** | ✅ 100% | ⚠️ 70% | ⚠️ 70% | ⚠️ 70% | ✅ 90% | **80%** |
| **worker-rbee-http-server** | ✅ 100% | ✅ 95% | ✅ 95% | ✅ 95% | ✅ 95% | **96%** |
| **worker-rbee-inference-base** | ✅ 100% | ⚠️ 60% | ⚠️ 50% | ⚠️ 50% | ⚠️ 60% | **64%** |

**Overall Reusability: 85%** (weighted by LOC)

---

## DETAILED ANALYSIS

### 1. worker-rbee-error (100% Reusable)

**Why 100%?**
- Error types are generic (CUDA, InvalidRequest, Timeout, Unhealthy, Internal, InsufficientResources, InsufficientVram)
- HTTP status mapping is generic
- Retriability logic is generic
- No LLM-specific error types

**Changes needed for other workers:** NONE

**Example usage (embedding worker):**
```rust
use worker_rbee_error::WorkerError;

// Same error types!
if insufficient_memory {
    return Err(WorkerError::InsufficientVram("Need 8GB".into()));
}
```

---

### 2. worker-rbee-startup (100% Reusable)

**Why 100%?**
- Callback protocol is worker-agnostic
- `model_ref`, `backend`, `device` are generic parameters
- No LLM-specific callback logic

**Changes needed for other workers:** NONE

**Example usage (vision worker):**
```rust
use worker_rbee_startup::callback_ready;

callback_ready(
    &callback_url,
    "vision-worker-1",
    "openai/clip-vit-large-patch14", // Vision model
    "cuda",
    0,
    8_000_000_000,
    8080,
).await?;
```

---

### 3. worker-rbee-health (100% Reusable)

**Why 100%?**
- Heartbeat protocol is worker-agnostic
- Health status (Healthy, Degraded) is generic
- No LLM-specific health checks

**Changes needed for other workers:** NONE

**Example usage (audio worker):**
```rust
use worker_rbee_health::{HeartbeatConfig, start_heartbeat_task};

let config = HeartbeatConfig::new("audio-worker-1".into(), callback_url);
let _handle = start_heartbeat_task(config);
```

---

### 4. worker-rbee-sse-streaming (80% Reusable)

**Why only 80%?**
- `Token` event is LLM-specific (text tokens)
- Other events (Started, Metrics, Narration, End, Error) are generic
- Needs refactoring to support non-text outputs

**Changes needed for other workers:**

#### Current (LLM-specific):
```rust
pub enum InferenceEvent {
    Token { t: String, i: u32 },  // ← LLM-specific!
    // ...
}
```

#### Refactored (Generic):
```rust
pub enum InferenceEvent<T> {
    Started { job_id, model, started_at },
    Output(T),  // ← Generic output type!
    Metrics { tokens_per_sec, vram_bytes },
    Narration { ... },
    End { ... },
    Error { code, message },
}

// Specializations
pub type LlmEvent = InferenceEvent<TokenOutput>;
pub type EmbeddingEvent = InferenceEvent<EmbeddingOutput>;
pub type VisionEvent = InferenceEvent<ImageOutput>;
pub type AudioEvent = InferenceEvent<AudioChunkOutput>;

#[derive(Serialize)]
pub struct TokenOutput {
    pub t: String,  // Token text
    pub i: u32,     // Token index
}

#[derive(Serialize)]
pub struct EmbeddingOutput {
    pub embedding: Vec<f32>,  // Embedding vector
    pub dim: usize,           // Dimension
}

#[derive(Serialize)]
pub struct ImageOutput {
    pub patch: Vec<u8>,       // Image data
    pub format: String,       // "png", "jpeg"
    pub i: u32,               // Chunk index
}

#[derive(Serialize)]
pub struct AudioChunkOutput {
    pub samples: Vec<f32>,    // Audio samples
    pub sample_rate: u32,     // Sample rate (Hz)
    pub i: u32,               // Chunk index
}
```

**Example usage (embedding worker):**
```rust
use worker_rbee_sse_streaming::{InferenceEvent, EmbeddingOutput};

let event = InferenceEvent::Output(EmbeddingOutput {
    embedding: vec![0.1, 0.2, ...],
    dim: 768,
});
```

---

### 5. worker-rbee-http-server (96% Reusable)

**Why 96%?**
- Server lifecycle is generic
- Route configuration is generic
- Health/ready endpoints are generic
- Execute endpoint is generic via `InferenceBackend` trait
- Authentication is generic
- Only `validation.rs` needs worker-specific validation

**Changes needed for other workers:**

#### Current (Generic trait):
```rust
pub trait InferenceBackend: Send + Sync {
    async fn execute(&mut self, req: ExecuteRequest) -> Result<InferenceResult>;
    fn is_healthy(&self) -> bool;
    fn is_ready(&self) -> bool;
    fn memory_bytes(&self) -> u64;
}
```

#### Example usage (embedding worker):
```rust
use worker_rbee_http_server::{InferenceBackend, ExecuteRequest, InferenceResult};

struct EmbeddingBackend { ... }

impl InferenceBackend for EmbeddingBackend {
    async fn execute(&mut self, req: ExecuteRequest) -> Result<InferenceResult> {
        // Generate embedding instead of text
        let embedding = self.model.encode(&req.prompt)?;
        
        Ok(InferenceResult {
            tokens_out: 1, // One embedding vector
            decode_time_ms: elapsed,
            stop_reason: StopReason::Complete,
            // ...
        })
    }
    
    // ... other methods same as LLM worker
}
```

**Validation differences:**
- LLM: `max_tokens` (1-2048), `temperature` (0.0-2.0)
- Embedding: No `max_tokens`, no sampling params
- Vision: `width`, `height`, `num_images`, `steps`
- Audio: `duration`, `sample_rate`, `format`

**Solution:** Make validation pluggable via trait

---

### 6. worker-rbee-inference-base (64% Reusable)

**Why only 64%?**
- Device management: ✅ 100% reusable
- Model loading (SafeTensors, GGUF): ✅ 90% reusable
- Tokenizer loading: ❌ 0% reusable (LLM-specific)
- Inference loop (autoregressive): ❌ 0% reusable (LLM-specific)
- Sampling logic: ❌ 0% reusable (LLM-specific)

**Generic parts:**
```rust
// ✅ Generic - All workers need these
pub fn init_device(backend: &str, device_id: u32) -> Result<Device>;
pub fn load_safetensors(path: &Path, device: &Device) -> Result<VarBuilder>;
pub fn load_gguf(path: &Path, device: &Device) -> Result<GgufFile>;
pub fn calculate_model_size(path: &Path) -> Result<u64>;
```

**LLM-specific parts:**
```rust
// ❌ LLM-specific
pub fn load_tokenizer(path: &Path) -> Result<Tokenizer>;
pub fn autoregressive_generate(model: &mut Model, ...) -> Result<Vec<String>>;
pub fn sample_token(logits: &Tensor, config: &SamplingConfig) -> Result<u32>;
```

**Recommendation:**  
Split into two crates:
1. **`worker-rbee-device`** (generic device/model loading)
2. **`llm-worker-rbee-inference`** (LLM-specific generation)

Then:
- `embedding-worker-rbee-inference` (embedding-specific)
- `vision-worker-rbee-inference` (vision-specific)
- `audio-worker-rbee-inference` (audio-specific)

---

## FUTURE WORKER IMPLEMENTATION ESTIMATES

### embedding-worker-rbee
**Reusable:** 85% (4,270 LOC)  
**New code:** 15% (750 LOC)
- Embedding-specific inference (~600 LOC)
- Embedding validation (~150 LOC)

**Timeline:** 1 week

---

### vision-worker-rbee
**Reusable:** 80% (4,020 LOC)  
**New code:** 20% (1,000 LOC)
- Vision-specific inference (~750 LOC)
- Vision validation (~150 LOC)
- Image format handling (~100 LOC)

**Timeline:** 1.5 weeks

---

### audio-worker-rbee
**Reusable:** 80% (4,020 LOC)  
**New code:** 20% (1,000 LOC)
- Audio-specific inference (~750 LOC)
- Audio validation (~150 LOC)
- Audio format handling (~100 LOC)

**Timeline:** 1.5 weeks

---

### multimodal-worker-rbee
**Reusable:** 90% (4,520 LOC)  
**New code:** 10% (500 LOC)
- Multimodal inference (~400 LOC)
- Multimodal validation (~100 LOC)

**Timeline:** 1 week

---

## ROI CALCULATION

### Without Decomposition:
- embedding-worker: 5,000 LOC (from scratch)
- vision-worker: 5,000 LOC (from scratch)
- audio-worker: 5,000 LOC (from scratch)
- **Total:** 15,000 LOC for 3 workers

### With Decomposition:
- Decomposition cost: 5,026 LOC → 6 crates (2 weeks)
- embedding-worker: 750 LOC new + 4,270 LOC reused (1 week)
- vision-worker: 1,000 LOC new + 4,020 LOC reused (1.5 weeks)
- audio-worker: 1,000 LOC new + 4,020 LOC reused (1.5 weeks)
- **Total:** 5,026 + 750 + 1,000 + 1,000 = 7,776 LOC

**Savings:** 15,000 - 7,776 = **7,224 LOC saved (48%)**  
**Time savings:** 15 weeks → 6.5 weeks = **8.5 weeks saved (57%)**

**Break-even:** After 2nd worker (1 week into embedding-worker)

---

## RECOMMENDATION

**Proceed with decomposition** - The reusability benefits are substantial!

**Critical:** Refactor SSE events to use generics (enables 80%+ reusability)
