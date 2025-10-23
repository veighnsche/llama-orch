# rbee Architecture Overview - Part 10: Worker Types & Adapters

**Version:** 1.0.0  
**Date:** October 23, 2025  
**Status:** Living Document

---

## Overview

rbee supports **multiple worker types** to handle different inference workloads and integrate with existing inference engines.

**Two categories:**
1. **Bespoke Workers** - Custom workers built with Candle LM framework
2. **Adapters** - Wrappers around existing inference engines (llama.cpp, vLLM, ComfyUI, etc.)

**All workers communicate with hive via the same HTTP protocol and heartbeat mechanism.**

---

## Worker Architecture Pattern

### Standard Worker Lifecycle

```
Hive spawns worker → Worker loads model → Worker reports ready → Worker accepts requests
                                    ↓
                            Heartbeat every 5s (status, VRAM, metrics)
```

**Key Properties:**
- **Model pre-attached:** Worker spawned with specific model
- **Fixed VRAM:** VRAM allocated at spawn time
- **Single purpose:** One model, one device, one task type
- **Process isolation:** Each worker in separate process

### Variable-VRAM Workers (ComfyUI, vLLM)

```
Hive spawns worker → Worker starts server → Models loaded dynamically → VRAM changes
                                    ↓
                            Heartbeat reports current VRAM usage
```

**Key Differences:**
- **Dynamic models:** Models change within process
- **Variable VRAM:** VRAM allocation changes over time
- **Multi-purpose:** Can handle multiple models/tasks
- **Requires enhanced heartbeat:** Must report current state

---

## 1. Bespoke Workers (Candle LM)

### Purpose

Custom-built workers using the Candle ML framework for maximum control and optimization.

### Current Implementation

**llm-worker-rbee** (bin/30_llm_worker_rbee/)
- **Framework:** Candle + Hugging Face Transformers
- **Models:** Any Hugging Face LLM (Llama, Mistral, etc.)
- **Backends:** CUDA, Metal, CPU
- **VRAM:** Fixed at spawn time

### Architecture

```rust
// Worker spawned by hive
let worker = LlmWorkerRbee::new(
    model: "meta-llama/Llama-3-8b",
    device: "cuda:0",
    port: 9001,
)?;

// Worker lifecycle
worker.load_model().await?;              // Load model into VRAM
worker.start_http_server().await?;       // Accept inference requests
worker.start_heartbeat(hive_url).await?; // Report status every 5s
```

**Heartbeat Payload:**
```json
{
  "worker_id": "worker-abc123",
  "status": "ready",
  "model": "meta-llama/Llama-3-8b",
  "device": "cuda:0",
  "vram_used": 8589934592,    // 8 GB (fixed)
  "vram_total": 25769803776,  // 24 GB
  "requests_total": 42,
  "tokens_generated": 10523,
  "uptime_seconds": 3600
}
```

### Future Bespoke Workers

**cuda-llm-worker-rbee**
- CUDA-optimized LLM worker
- Flash Attention 2
- Tensor parallelism support

**metal-llm-worker-rbee**
- macOS Metal backend
- Optimized for M1/M2/M3 chips

**metal-stable-diffusion-worker-rbee**
- Stable Diffusion on Metal
- Image generation workload
- Different heartbeat (reports generation progress)

---

## 1.5 Distributed Inference (Multi-Worker GGUF)

### Purpose

**Distributed inference** allows a single large model to be split across multiple workers, enabling inference on models larger than a single GPU's VRAM.

**Use case:** Running 70B or 405B models across multiple GPUs (e.g., 4x 24GB GPUs for a 70B model).

### Candle Support

Candle provides native support for distributed inference with GGUF models:
- **Tensor parallelism** - Split model layers across GPUs
- **Pipeline parallelism** - Split model depth across GPUs
- **Minimal code changes** - Candle handles distribution

### Architecture

```
Hive spawns worker group → Workers coordinate → Model sharded across GPUs
                                    ↓
                            Worker 0: Layers 0-19 (GPU-0)
                            Worker 1: Layers 20-39 (GPU-1)
                            Worker 2: Layers 40-59 (GPU-2)
                            Worker 3: Layers 60-79 (GPU-3)
                                    ↓
                            Inference coordinated via shared memory/NCCL
```

**Key Properties:**
- **Single inference endpoint** - Client talks to coordinator worker
- **Transparent distribution** - Client doesn't know model is distributed
- **Automatic load balancing** - Candle handles inter-GPU communication
- **Fault tolerance** - If one GPU fails, entire group stops

### Worker Group Pattern

**Spawn Command:**
```rust
Operation::WorkerSpawnGroup {
    model: "meta-llama/Llama-3-70b",
    devices: vec!["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    coordinator_port: 9001,
}
```

**Hive Response:**
```json
{
  "group_id": "worker-group-abc",
  "coordinator": {
    "worker_id": "worker-abc-0",
    "device": "cuda:0",
    "port": 9001,
    "role": "coordinator"
  },
  "workers": [
    {"worker_id": "worker-abc-1", "device": "cuda:1", "role": "shard"},
    {"worker_id": "worker-abc-2", "device": "cuda:2", "role": "shard"},
    {"worker_id": "worker-abc-3", "device": "cuda:3", "role": "shard"}
  ],
  "total_vram": 103079215104,  // 96 GB (4x 24GB)
  "model_vram": 70000000000     // 70 GB (model size)
}
```

### Heartbeat (Group Coordinator)

**Coordinator reports group health:**
```json
{
  "worker_id": "worker-abc-0",
  "group_id": "worker-group-abc",
  "status": "ready",
  "role": "coordinator",
  "model": "meta-llama/Llama-3-70b",
  "devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
  "vram_per_device": {
    "cuda:0": 20000000000,  // Coordinator + first shard
    "cuda:1": 18000000000,  // Shard
    "cuda:2": 18000000000,  // Shard
    "cuda:3": 14000000000   // Shard + overhead
  },
  "vram_total": 103079215104,
  "group_health": "healthy",
  "shard_workers": [
    {"worker_id": "worker-abc-1", "status": "healthy"},
    {"worker_id": "worker-abc-2", "status": "healthy"},
    {"worker_id": "worker-abc-3", "status": "healthy"}
  ]
}
```

**Individual shards also send heartbeats:**
```json
{
  "worker_id": "worker-abc-1",
  "group_id": "worker-group-abc",
  "status": "ready",
  "role": "shard",
  "coordinator": "worker-abc-0",
  "device": "cuda:1",
  "vram_used": 18000000000,
  "vram_total": 25769803776
}
```

### Client Interaction

**Client connects to coordinator only:**
```rust
// Client perspective
let response = reqwest::post("http://localhost:9001/v1/infer")
    .json(&InferRequest {
        prompt: "Hello!",
        max_tokens: 100,
    })
    .send()
    .await?;

// Coordinator distributes work across shards transparently
```

**Client doesn't need to know about distribution!**

### Candle Implementation

```rust
// bin/30_llm_worker_rbee/src/distributed.rs
use candle_core::Device;
use candle_nn::VarBuilder;

pub struct DistributedWorkerGroup {
    coordinator: WorkerId,
    shards: Vec<WorkerId>,
    devices: Vec<Device>,
    model: DistributedModel,
}

impl DistributedWorkerGroup {
    pub async fn spawn(
        model: &str,
        devices: Vec<String>,
        coordinator_port: u16,
    ) -> Result<Self> {
        // 1. Parse devices
        let devices: Vec<Device> = devices.iter()
            .map(|d| Device::new_cuda(parse_cuda_index(d)?))
            .collect::<Result<Vec<_>>>()?;
        
        // 2. Load model with tensor parallelism
        let vb = VarBuilder::from_gguf(model_path, &devices)?;
        let model = LlamaModel::load_distributed(vb, &devices)?;
        
        // 3. Spawn coordinator worker
        let coordinator = spawn_coordinator_worker(
            model.clone(),
            devices[0].clone(),
            coordinator_port,
        ).await?;
        
        // 4. Spawn shard workers
        let mut shards = vec![];
        for (i, device) in devices[1..].iter().enumerate() {
            let shard = spawn_shard_worker(
                model.get_shard(i + 1),
                device.clone(),
                coordinator.clone(),
            ).await?;
            shards.push(shard);
        }
        
        Ok(Self {
            coordinator,
            shards,
            devices,
            model,
        })
    }
    
    pub async fn infer(&self, prompt: String) -> Result<String> {
        // Candle handles distribution automatically
        let tokens = self.model.generate(&prompt, 100)?;
        Ok(tokens)
    }
}
```

**Key Insight:** Candle abstracts away distribution complexity!

### Benefits

**For Users:**
- ✅ Run models larger than single GPU VRAM
- ✅ Transparent - same API as single-worker
- ✅ Better hardware utilization

**For rbee:**
- ✅ Candle handles complexity
- ✅ Standard worker pattern (coordinator is just another worker)
- ✅ Heartbeat tracks group health

### Failure Handling

**If one shard fails:**
1. Shard stops sending heartbeats
2. Coordinator detects missing heartbeat
3. Coordinator reports `group_health: "degraded"`
4. Hive marks entire group as unhealthy
5. New inference requests rejected
6. Option: Auto-respawn failed shard (M2 feature)

### Implementation Status

- ❌ Not implemented (M1/M2)
- ✅ Candle supports distributed inference
- ⚠️ Requires NCCL for multi-node (single-node first)

### Implementation Plan

**Phase 1: Single-Machine Multi-GPU (16-24 hours)**
- Implement WorkerSpawnGroup operation
- Coordinator/shard worker pattern
- Group heartbeat tracking
- Candle distributed model loading

**Phase 2: Fault Tolerance (8-12 hours)**
- Group health monitoring
- Graceful degradation
- Auto-respawn (optional)

**Phase 3: Multi-Node (M2 - 24-32 hours)**
- NCCL support for multi-machine
- Network-based shard communication
- Cross-machine group coordination

**Total Effort:** 48-68 hours (single-machine first, multi-node M2)

---

## 2. Adapters (Existing Inference Engines)

### Purpose

Wrap existing inference engines to work with rbee's HTTP protocol and heartbeat mechanism.

### Adapter Pattern

```
rbee Adapter → Existing Engine
   ↓              ↓
HTTP API      Native API
Heartbeat     Internal state
```

**Adapter Responsibilities:**
1. Translate rbee HTTP requests to engine-native API
2. Monitor engine state
3. Report status via heartbeat
4. Manage engine lifecycle (start/stop)

---

## 2.1 llama.cpp Adapter

**rbee-llama.cpp-adapter** (bin/35_adapters/rbee-llama.cpp-adapter/)

### Purpose

Integrate llama.cpp for maximum CPU performance and GGUF format support.

### Architecture

```
Hive → rbee-llama.cpp-adapter → llama.cpp (subprocess)
           ↓
       HTTP API (/v1/infer)
       Heartbeat (status)
```

### Implementation

```rust
pub struct LlamaCppAdapter {
    llama_process: Child,          // llama.cpp subprocess
    model_path: PathBuf,           // GGUF file
    context_size: usize,           // Context window
    threads: usize,                // CPU threads
}

impl LlamaCppAdapter {
    pub async fn start(&mut self) -> Result<()> {
        // Start llama.cpp server
        self.llama_process = Command::new("llama-server")
            .arg("--model").arg(&self.model_path)
            .arg("--ctx-size").arg(self.context_size.to_string())
            .arg("--threads").arg(self.threads.to_string())
            .arg("--port").arg("8080")
            .spawn()?;
        
        // Wait for server ready
        self.wait_for_ready().await?;
        
        Ok(())
    }
    
    pub async fn infer(&self, prompt: String) -> Result<String> {
        // Translate rbee request to llama.cpp format
        let request = serde_json::json!({
            "prompt": prompt,
            "n_predict": 100,
        });
        
        // Forward to llama.cpp
        let response = self.client
            .post("http://localhost:8080/completion")
            .json(&request)
            .send()
            .await?;
        
        // Extract result
        let result: LlamaCppResponse = response.json().await?;
        Ok(result.content)
    }
}
```

**Heartbeat:**
```json
{
  "worker_id": "adapter-llama-cpp-xyz",
  "status": "ready",
  "adapter_type": "llama.cpp",
  "model": "llama-3-8b.Q4_K_M.gguf",
  "backend": "cpu",
  "threads": 16,
  "vram_used": 0,           // CPU only
  "ram_used": 4294967296,   // 4 GB RAM
  "requests_total": 10
}
```

---

## 2.2 vLLM Adapter

**rbee-vllm-adapter** (bin/35_adapters/rbee-vllm-adapter/)

### Purpose

Integrate vLLM for high-throughput batch inference and advanced optimizations (PagedAttention, continuous batching).

### Special Considerations

**vLLM supports dynamic model loading:**
- Can load multiple models in same process
- VRAM usage changes as models load/unload
- **Requires enhanced heartbeat**

### Architecture

```
Hive → rbee-vllm-adapter → vLLM (Python subprocess)
           ↓                      ↓
       HTTP API            OpenAI-compatible API
       Heartbeat           Dynamic model loading
```

### Implementation

```rust
pub struct VllmAdapter {
    vllm_process: Child,              // vLLM subprocess
    loaded_models: Vec<String>,       // Currently loaded models
    vram_allocations: HashMap<String, u64>, // Per-model VRAM
}

impl VllmAdapter {
    pub async fn start(&mut self) -> Result<()> {
        // Start vLLM server
        self.vllm_process = Command::new("python")
            .arg("-m").arg("vllm.entrypoints.openai.api_server")
            .arg("--host").arg("0.0.0.0")
            .arg("--port").arg("8000")
            .spawn()?;
        
        Ok(())
    }
    
    pub async fn load_model(&mut self, model: String) -> Result<()> {
        // Load model dynamically (if vLLM supports it)
        // Track VRAM allocation
        let vram = self.query_vram_usage(&model).await?;
        self.loaded_models.push(model.clone());
        self.vram_allocations.insert(model, vram);
        
        Ok(())
    }
    
    pub async fn heartbeat_payload(&self) -> Result<HeartbeatPayload> {
        // Enhanced heartbeat with current state
        Ok(HeartbeatPayload {
            worker_id: self.worker_id.clone(),
            status: "ready",
            adapter_type: "vllm",
            loaded_models: self.loaded_models.clone(), // NEW!
            vram_used: self.total_vram_used(),         // Dynamic!
            vram_total: self.gpu_vram_total(),
            requests_total: self.requests_total,
        })
    }
}
```

**Heartbeat (Enhanced):**
```json
{
  "worker_id": "adapter-vllm-abc",
  "status": "ready",
  "adapter_type": "vllm",
  "loaded_models": [               // NEW: Multiple models
    "meta-llama/Llama-3-8b",
    "mistralai/Mistral-7B-v0.1"
  ],
  "vram_allocations": {            // NEW: Per-model VRAM
    "meta-llama/Llama-3-8b": 8589934592,
    "mistralai/Mistral-7B-v0.1": 7516192768
  },
  "vram_used": 16106127360,        // Total (dynamic!)
  "vram_total": 25769803776,
  "requests_total": 156
}
```

---

## 2.3 ComfyUI Adapter

**rbee-comfyui-adapter** (bin/35_adapters/rbee-comfyui-adapter/)

### Purpose

Integrate ComfyUI for visual workflow-based image generation (Stable Diffusion, ControlNet, etc.).

### Special Considerations

**ComfyUI is highly dynamic:**
- Workflows can load/unload models on-the-fly
- Multiple models per workflow (SD model, VAE, ControlNet, LoRA)
- VRAM usage changes dramatically between workflows
- Workflows can take minutes to complete
- **Requires most enhanced heartbeat**

### Architecture

```
Hive → rbee-comfyui-adapter → ComfyUI (Python subprocess)
           ↓                        ↓
       HTTP API              WebSocket API
       Heartbeat             Workflow execution
       Progress tracking     Model management
```

### Implementation

```rust
pub struct ComfyUIAdapter {
    comfyui_process: Child,                      // ComfyUI subprocess
    current_workflow: Option<Workflow>,          // Active workflow
    loaded_models: HashMap<String, ModelInfo>,   // All loaded models
    vram_used: u64,                              // Current VRAM usage
    workflow_progress: Option<WorkflowProgress>, // Current progress
}

impl ComfyUIAdapter {
    pub async fn execute_workflow(&mut self, workflow: Workflow) -> Result<()> {
        self.current_workflow = Some(workflow.clone());
        
        // Submit workflow to ComfyUI
        let prompt_id = self.submit_workflow(&workflow).await?;
        
        // Monitor execution via WebSocket
        self.monitor_workflow_execution(prompt_id).await?;
        
        Ok(())
    }
    
    async fn monitor_workflow_execution(&mut self, prompt_id: String) -> Result<()> {
        // Connect to ComfyUI WebSocket
        let mut ws = self.connect_websocket().await?;
        
        while let Some(msg) = ws.next().await {
            match msg? {
                ComfyUIMessage::Progress { node, value, max } => {
                    // Update progress for heartbeat
                    self.workflow_progress = Some(WorkflowProgress {
                        prompt_id: prompt_id.clone(),
                        current_node: node,
                        progress: value as f32 / max as f32,
                    });
                }
                ComfyUIMessage::Executing { node } => {
                    // Node started executing
                }
                ComfyUIMessage::Executed { node, output } => {
                    // Node finished executing
                }
                ComfyUIMessage::ExecutionError { error } => {
                    // Workflow failed
                    return Err(anyhow!("Workflow failed: {}", error));
                }
            }
        }
        
        Ok(())
    }
    
    pub async fn heartbeat_payload(&self) -> Result<HeartbeatPayload> {
        // Query current VRAM from ComfyUI
        let vram = self.query_current_vram().await?;
        self.vram_used = vram;
        
        Ok(HeartbeatPayload {
            worker_id: self.worker_id.clone(),
            status: if self.current_workflow.is_some() { "busy" } else { "ready" },
            adapter_type: "comfyui",
            loaded_models: self.loaded_models.keys().cloned().collect(),
            vram_used: self.vram_used,           // Queried dynamically
            vram_total: self.gpu_vram_total(),
            current_workflow: self.current_workflow.as_ref().map(|w| w.id.clone()),
            workflow_progress: self.workflow_progress.clone(), // NEW!
            requests_total: self.requests_total,
        })
    }
}
```

**Heartbeat (Most Enhanced):**
```json
{
  "worker_id": "adapter-comfyui-123",
  "status": "busy",                 // busy during workflow
  "adapter_type": "comfyui",
  "loaded_models": [                // All models in VRAM
    "sd-v1-5.safetensors",
    "controlnet-canny.safetensors",
    "vae-ft-mse.safetensors",
    "lora-style.safetensors"
  ],
  "vram_used": 18253611008,         // Queried dynamically
  "vram_total": 25769803776,
  "current_workflow": "workflow-xyz",     // NEW: Active workflow
  "workflow_progress": {                  // NEW: Progress tracking
    "prompt_id": "prompt-abc",
    "current_node": "sampler",
    "progress": 0.65
  },
  "requests_total": 23
}
```

---

## Heartbeat Protocol (Detailed)

### Current Implementation

**Location:** Worker sends heartbeat to hive every 5 seconds

**Endpoint:** `POST /v1/heartbeat`

**Standard Payload:**
```json
{
  "worker_id": "worker-abc123",
  "status": "ready" | "busy" | "error",
  "model": "model-name",
  "device": "cuda:0",
  "vram_used": 8589934592,
  "vram_total": 25769803776,
  "requests_total": 42,
  "tokens_generated": 10523,
  "uptime_seconds": 3600,
  "last_request_at": "2025-10-23T12:00:00Z"
}
```

### Enhanced Heartbeat (For Adapters)

**Additional fields for dynamic workers:**

```json
{
  // ... standard fields ...
  
  // NEW: For multi-model workers
  "loaded_models": ["model-1", "model-2"],
  "vram_allocations": {
    "model-1": 8589934592,
    "model-2": 7516192768
  },
  
  // NEW: For workflow-based workers (ComfyUI)
  "current_workflow": "workflow-id",
  "workflow_progress": {
    "prompt_id": "prompt-abc",
    "current_node": "sampler",
    "progress": 0.65
  },
  
  // NEW: For batch workers (vLLM)
  "batch_size": 8,
  "queue_depth": 12,
  "throughput_tokens_per_second": 256.5
}
```

### Hive Response

**Hive acknowledges heartbeat and can send commands:**

```json
{
  "ack": true,
  "commands": [
    {
      "type": "unload_model",
      "model": "old-model"
    },
    {
      "type": "shutdown",
      "reason": "maintenance"
    }
  ]
}
```

---

## Worker Type Comparison

| Feature | Bespoke Worker | llama.cpp Adapter | vLLM Adapter | ComfyUI Adapter |
|---------|----------------|-------------------|--------------|-----------------|
| **Framework** | Candle | llama.cpp | vLLM | ComfyUI |
| **Language** | Rust | C++ (wrapped) | Python (wrapped) | Python (wrapped) |
| **Models** | Single | Single | Multiple | Multiple |
| **VRAM** | Fixed | Fixed | Dynamic | Highly dynamic |
| **Workload** | LLM inference | LLM inference | Batch LLM | Image generation |
| **Heartbeat** | Standard | Standard | Enhanced | Most enhanced |
| **Complexity** | Medium | Low | Medium | High |
| **Performance** | High | Very high (CPU) | Very high (GPU batch) | Varies |

---

## Implementation Priority

### Phase 1: Core Workers (Current)
- ✅ **llm-worker-rbee** (Candle-based, done)

### Phase 2: High-Value Adapters (Next)
- ⚠️ **rbee-llama.cpp-adapter** - For CPU inference and GGUF support
- ⚠️ **rbee-vllm-adapter** - For high-throughput GPU inference

### Phase 3: Specialized Workers
- ❌ **cuda-llm-worker-rbee** - CUDA-optimized bespoke worker
- ❌ **metal-llm-worker-rbee** - Metal-optimized for macOS
- ❌ **metal-stable-diffusion-worker-rbee** - Image generation on Metal

### Phase 4: Advanced Adapters
- ❌ **rbee-comfyui-adapter** - Visual workflows
- ❌ **rbee-ollama-adapter** - Integrate Ollama
- ❌ **rbee-tensorrt-llm-adapter** - NVIDIA TensorRT-LLM

**Estimated Effort:**
- Simple adapter (llama.cpp): 12-16 hours
- Enhanced adapter (vLLM): 20-28 hours
- Complex adapter (ComfyUI): 32-48 hours

---

## Directory Structure

```
bin/
├── 30_llm_worker_rbee/          # Bespoke Candle worker (done)
├── 31_cuda_llm_worker_rbee/     # CUDA-optimized worker (future)
├── 32_metal_llm_worker_rbee/    # Metal-optimized worker (future)
├── 33_metal_sd_worker_rbee/     # Stable Diffusion worker (future)
└── 35_adapters/                 # Adapter binaries
    ├── rbee-llama.cpp-adapter/  # llama.cpp integration
    ├── rbee-vllm-adapter/        # vLLM integration
    ├── rbee-comfyui-adapter/     # ComfyUI integration
    ├── rbee-ollama-adapter/      # Ollama integration
    └── rbee-tensorrt-llm-adapter/ # TensorRT-LLM integration
```

---

## Adapter Development Guide

### Creating a New Adapter

**Template:**
```rust
// bin/35_adapters/rbee-{engine}-adapter/src/main.rs

use rbee_operations::Operation;
use observability_narration_core::NARRATE;

pub struct EngineAdapter {
    worker_id: String,
    engine_process: Option<Child>,
    port: u16,
    // Engine-specific state
}

impl EngineAdapter {
    pub async fn start(&mut self) -> Result<()> {
        // 1. Start engine subprocess
        self.engine_process = Some(self.spawn_engine().await?);
        
        // 2. Wait for engine ready
        self.wait_for_ready().await?;
        
        // 3. Start HTTP server (rbee API)
        self.start_http_server().await?;
        
        // 4. Start heartbeat
        self.start_heartbeat_loop().await?;
        
        Ok(())
    }
    
    pub async fn handle_infer(&self, request: InferRequest) -> Result<String> {
        // Translate rbee request to engine API
        let engine_request = self.translate_request(request)?;
        
        // Forward to engine
        let engine_response = self.engine_client.infer(engine_request).await?;
        
        // Translate response back
        let rbee_response = self.translate_response(engine_response)?;
        
        Ok(rbee_response)
    }
    
    async fn heartbeat_payload(&self) -> Result<HeartbeatPayload> {
        // Query engine state
        let state = self.query_engine_state().await?;
        
        Ok(HeartbeatPayload {
            worker_id: self.worker_id.clone(),
            status: state.status,
            adapter_type: "engine-name",
            // ... engine-specific fields
        })
    }
}
```

### Key Requirements

1. **HTTP API compatibility** - Must implement rbee worker API
2. **Heartbeat** - Must send heartbeat every 5s
3. **Process management** - Must manage engine subprocess lifecycle
4. **Error handling** - Must catch and report engine errors
5. **VRAM tracking** - Must report accurate VRAM usage
6. **Graceful shutdown** - Must cleanly stop engine process

---

## Future Enhancements

### 1. Adapter Plugin System

Allow users to write adapters without modifying rbee source:

```rust
// User writes adapter as plugin
pub struct MyEngineAdapter;

impl WorkerAdapter for MyEngineAdapter {
    fn start(&mut self) -> Result<()> { ... }
    fn infer(&self, request: InferRequest) -> Result<String> { ... }
    fn heartbeat(&self) -> Result<HeartbeatPayload> { ... }
}

// Load plugin dynamically
rbee-hive --adapter-plugin my-engine-adapter.so
```

### 2. Auto-Scaling Based on Heartbeat

Hive uses heartbeat VRAM data to decide when to spawn/stop workers:

```rust
// Hive decision logic
if vram_usage > 80% {
    // High VRAM pressure, don't spawn new workers
} else if vram_usage < 20% && idle_time > 5_minutes {
    // Low utilization, stop idle worker
}
```

### 3. Multi-GPU Workers

Workers that use multiple GPUs (model parallelism):

```json
{
  "worker_id": "worker-multi-gpu",
  "devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
  "vram_used_per_device": {
    "cuda:0": 20000000000,
    "cuda:1": 20000000000,
    "cuda:2": 20000000000,
    "cuda:3": 20000000000
  }
}
```

---

## Summary

**rbee supports diverse worker types:**
- **Bespoke workers** for maximum control (Candle)
- **Adapters** for existing engines (llama.cpp, vLLM, ComfyUI)
- **Variable-VRAM workers** with enhanced heartbeat
- **Extensible** via adapter pattern

**All workers speak the same protocol:**
- HTTP API for inference
- Heartbeat for status reporting
- Process isolation for reliability

**Heartbeat is the key:**
- Standard: Fixed VRAM, single model
- Enhanced: Dynamic VRAM, multiple models
- Most enhanced: Workflow progress, highly dynamic

---

**Created by:** TEAM-266  
**Status:** Design document  
**Next:** Implement llama.cpp and vLLM adapters (20-40 hours)
