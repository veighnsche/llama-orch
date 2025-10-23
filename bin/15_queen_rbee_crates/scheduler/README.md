# queen-rbee-inference-scheduler

**Category:** Orchestration  
**Pattern:** Strategy Pattern (pluggable scheduler implementations)  
**Status:** M0/M1 - Simple scheduler, M2+ - Rhai programmable scheduler

---

## Overview

Inference scheduler for queen-rbee. Decides which worker should handle an inference request.

### Current Implementation (M0/M1)

**SimpleScheduler** - First available worker:
- Find workers serving requested model
- Filter by online + available status
- Return first match (no load balancing)

### Future Implementation (M2+)

**RhaiScheduler** - Programmable routing via Rhai scripts:
- User-written Rhai scripts for custom routing
- 40+ built-in helper functions
- YAML config support (compiles to Rhai)
- Web UI policy builder
- Platform mode (immutable) vs Home/Lab mode (customizable)

See: `.business/stakeholders/RHAI_PROGRAMMABLE_SCHEDULER.md`

---

## Architecture

```
┌─────────────────────────────────────────┐
│  InferenceScheduler (trait)             │
└─────────────────┬───────────────────────┘
                  │
       ┌──────────┴──────────┐
       │                     │
┌──────▼──────┐      ┌──────▼──────────┐
│ SimpleScheduler │  │ RhaiScheduler   │
│ (M0/M1)        │  │ (M2+)           │
└────────────────┘  └─────────────────┘
```

---

## Usage

### Simple Scheduler (Current)

```rust
use queen_rbee_inference_scheduler::{SimpleScheduler, InferenceRequest};
use queen_rbee_worker_registry::WorkerRegistry;
use std::sync::Arc;

let registry = Arc::new(WorkerRegistry::new());
let scheduler = SimpleScheduler::new(registry);

let request = InferenceRequest {
    job_id: "job-123".to_string(),
    model: "meta-llama/Llama-3-8b".to_string(),
    prompt: "Hello!".to_string(),
    max_tokens: 20,
    temperature: 0.7,
    top_p: None,
    top_k: None,
};

// Schedule (find worker)
let result = scheduler.schedule(request.clone()).await?;
println!("Selected worker: {}", result.worker_id);

// Execute (run inference and stream)
scheduler.execute_inference(result, request, |line| {
    println!("{}", line);
    Ok(())
}).await?;
```

### Rhai Scheduler (M2+)

```rust
// TODO M2: Rhai scheduler example
use queen_rbee_inference_scheduler::RhaiScheduler;

let scheduler = RhaiScheduler::from_file("scheduler.rhai")?;
let result = scheduler.schedule(request).await?;
```

**Example Rhai script:**

```rhai
// scheduler.rhai
fn schedule_job(job, pools, workers) {
    // Route large models to multi-GPU setups
    if job.model.contains("70b") {
        return workers.filter(|w| w.gpus > 1).first();
    }
    
    // Route to cheapest available
    return workers.min_by(|w| w.cost_per_token);
}
```

---

## API

### InferenceScheduler Trait

```rust
#[async_trait]
pub trait InferenceScheduler: Send + Sync {
    async fn schedule(
        &self,
        request: InferenceRequest
    ) -> Result<ScheduleResult, SchedulerError>;
}
```

### Types

**InferenceRequest:**
- `job_id` - Job ID for tracking
- `model` - Model to use
- `prompt` - Prompt text
- `max_tokens` - Maximum tokens to generate
- `temperature` - Sampling temperature
- `top_p` - Nucleus sampling (optional)
- `top_k` - Top-k sampling (optional)

**ScheduleResult:**
- `worker_id` - Selected worker ID
- `worker_url` - Worker base URL
- `worker_port` - Worker port
- `model` - Model being served
- `device` - Device worker is using

**SchedulerError:**
- `NoWorkersAvailable` - No workers for model
- `WorkerCommunicationFailed` - HTTP error
- `WorkerError` - Worker returned error
- `ParseError` - Failed to parse response
- `StreamConnectionFailed` - SSE connection failed
- `StreamReadError` - Error reading stream

---

## Future: Rhai Programmable Scheduler (M2+)

### Two Modes

**Platform Mode:**
- Immutable scheduler (users can't modify)
- Multi-tenant fairness
- SLA compliance
- Quota enforcement

**Home/Lab Mode:**
- User-written Rhai scripts
- YAML config support
- Web UI policy builder
- Custom routing logic

### 40+ Built-in Helpers

**Worker Selection:**
- `workers.least_loaded()` - Pick worker with lowest load
- `workers.most_vram_free()` - Pick worker with most free VRAM
- `workers.round_robin()` - Distribute evenly
- `workers.filter(predicate)` - Filter by condition
- `workers.min_by(key)` - Pick minimum by key

**GPU Queries:**
- `gpu_vram_total(gpu_id)` - Total VRAM
- `gpu_vram_free(gpu_id)` - Free VRAM
- `gpu_device_name(gpu_id)` - Device name

**Quota Checks (Platform Mode):**
- `tenant_over_quota(tenant_id)` - Check quota exceeded
- `tenant_remaining_quota(tenant_id)` - Remaining quota

### Example Policies

**Cost Optimizer:**
```rhai
fn schedule_job(job, pools, workers) {
    return workers.min_by(|w| w.cost_per_token);
}
```

**Latency Optimizer:**
```rhai
fn schedule_job(job, pools, workers) {
    return workers.min_by(|w| 
        distance(job.user.geo, w.geo)
    );
}
```

**Multi-Modal Router:**
```rhai
fn schedule_job(job, pools, workers) {
    if job.type == "image-gen" {
        return workers
            .filter(|w| w.capability == "image-gen")
            .least_loaded();
    } else if job.model.contains("70b") {
        return workers.filter(|w| w.gpus > 1).first();
    } else {
        return workers.min_by(|w| w.cost_per_token);
    }
}
```

---

## Implementation Timeline

**M0 (Current):** ✅ SimpleScheduler  
**M1:** ✅ SimpleScheduler (stable)  
**M2 (Q2 2026):** ✅ RhaiScheduler  
**M3 (Q3 2026):** ✅ Platform mode  
**M4 (Q4 2026):** ✅ Web UI policy builder

---

## Testing

```bash
cargo test -p queen-rbee-inference-scheduler
```

---

## Dependencies

- `queen-rbee-worker-registry` - Worker tracking
- `observability-narration-core` - Logging
- `reqwest` - HTTP client
- `futures` - Async streaming
- `rhai` (M2+) - Scripting engine

---

**TEAM-275:** Created stub crate for future Rhai scheduler  
**See:** `.business/stakeholders/RHAI_PROGRAMMABLE_SCHEDULER.md`
