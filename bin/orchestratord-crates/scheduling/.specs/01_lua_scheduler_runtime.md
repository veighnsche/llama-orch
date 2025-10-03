# Lua Scheduler Runtime Environment Specification

**Status**: Draft  
**Version**: 0.1.0  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 1. Executive Summary

The Lua scheduler runtime provides a **rich, preloaded environment** with all data and functions needed to make informed scheduling decisions. The runtime is designed to give users complete visibility into system state and powerful primitives for custom scheduling logic.

### Design Principles

1. **Complete State Access**: All system state available at runtime (pools, workers, GPUs, models, queue)
2. **Rich Built-in Functions**: Comprehensive API for common scheduling operations
3. **Compile-Time Catalog**: Model catalog and GPU capabilities preloaded
4. **Real-Time Pool State**: Fresh pool manager data fetched on each schedule cycle
5. **Zero External Dependencies**: Everything needed is built-in (no require/import)

---

## 2. Scheduler Entry Point

Every Lua scheduler MUST implement the `schedule()` function:

```lua
-- Main entry point called by orchestratord
-- @param ctx: SchedulerContext - Complete system state
-- @return SchedulerDecision - What action to take
function schedule(ctx)
  -- User's custom scheduling logic here
  
  -- Example: dispatch first job to least-loaded worker
  local job = ctx.queue[1]
  if job then
    local worker = ctx.find_least_loaded_worker(job.model_ref)
    if worker then
      return {
        action = "dispatch",
        job_id = job.id,
        worker_id = worker.id
      }
    end
  end
  
  -- No action
  return { action = "wait" }
end
```

---

## 3. SchedulerContext Object (ctx)

The `ctx` object is passed to every `schedule()` call and contains complete system state.

### 3.1 Queue State (`ctx.queue`)

Array of jobs waiting to be scheduled, ordered by priority and arrival time.

```lua
ctx.queue = {
  {
    id = "job-abc123",
    tenant_id = "tenant-1",  -- nil in home/lab mode
    session_id = "sess-xyz",
    priority = "interactive",  -- "interactive" or "batch"
    model_ref = "hf:meta-llama/Llama-3.1-8B@main::file=model.gguf",
    prompt_hash = "sha256:...",  -- SHA256 of prompt (privacy)
    max_tokens = 100,
    seed = 42,  -- nil if not provided
    temperature = 0.7,
    
    -- Timestamps (Unix milliseconds)
    created_at = 1696348800000,
    queued_at = 1696348801000,
    
    -- Metadata
    retry_count = 0,
    wait_time_ms = 5000,  -- calculated: now - queued_at
    
    -- Estimated resource needs (from model catalog)
    estimated_vram_bytes = 8589934592,  -- 8GB
    estimated_duration_ms = 2000,
  },
  -- ... more jobs
}
```

**Helper methods**:
```lua
-- Get job by ID
local job = ctx.get_job("job-abc123")

-- Filter jobs by criteria
local interactive_jobs = ctx.filter_jobs({ priority = "interactive" })
local seeded_jobs = ctx.filter_jobs(function(job) return job.seed ~= nil end)
```

---

### 3.2 Pool State (`ctx.pools`)

Array of pool managers with their workers and GPUs. **Fetched fresh** from pool manager heartbeats.

```lua
ctx.pools = {
  {
    id = "pool-1",
    endpoint = "http://192.168.1.100:9200",
    status = "healthy",  -- "healthy", "degraded", "offline"
    last_heartbeat_ms = 1696348805000,
    heartbeat_age_ms = 500,  -- now - last_heartbeat_ms
    
    -- GPU inventory (from NVML)
    gpus = {
      {
        id = 0,
        device_name = "NVIDIA RTX 4090",
        compute_capability = "8.9",
        vram_total_bytes = 25769803776,  -- 24GB
        vram_allocated_bytes = 17179869184,  -- 16GB
        vram_free_bytes = 8589934592,  -- 8GB
        utilization_percent = 75.0,
        temperature_celsius = 68,
        power_draw_watts = 350,
        
        -- Capability flags
        supports_fp16 = true,
        supports_int8 = true,
        supports_tensor_cores = true,
      },
      {
        id = 1,
        device_name = "NVIDIA RTX 4090",
        -- ... similar structure
      }
    },
    
    -- Workers running on this pool
    workers = {
      {
        id = "worker-1",
        status = "ready",  -- "ready", "busy", "starting", "failed"
        model_ref = "hf:meta-llama/Llama-3.1-8B@main::file=model.gguf",
        gpu_ids = {0},  -- GPUs this worker uses
        vram_bytes = 8589934592,  -- 8GB allocated
        uri = "http://192.168.1.100:8001",
        
        -- Current job (if busy)
        current_job_id = nil,  -- or "job-xyz" if busy
        job_started_at = nil,
        
        -- Performance stats
        jobs_completed = 42,
        avg_tokens_per_sec = 85.3,
        uptime_ms = 3600000,  -- 1 hour
      },
      -- ... more workers
    },
    
    -- Cached models in RAM (for fast worker starts)
    cached_models = {
      "hf:meta-llama/Llama-3.1-8B@main::file=model.gguf",
      "hf:meta-llama/Llama-3.1-13B@main::file=model.gguf",
    }
  },
  -- ... more pools
}
```

**Helper methods**:
```lua
-- Get pool by ID
local pool = ctx.get_pool("pool-1")

-- Find pools with available VRAM
local pools_with_vram = ctx.filter_pools(function(pool)
  return pool.gpus[1].vram_free_bytes > 8 * 1024^3  -- 8GB
end)

-- Get all workers across all pools
local all_workers = ctx.get_all_workers()

-- Find workers by model
local llama_workers = ctx.find_workers_by_model("hf:meta-llama/Llama-3.1-8B@main")
```

---

### 3.3 Model Catalog (`ctx.catalog`)

**Preloaded at compile time** - complete model catalog with metadata.

```lua
ctx.catalog = {
  models = {
    ["hf:meta-llama/Llama-3.1-8B@main::file=model.gguf"] = {
      model_ref = "hf:meta-llama/Llama-3.1-8B@main::file=model.gguf",
      family = "llama",
      size = "8B",
      quantization = "Q4_K_M",
      
      -- Resource requirements
      vram_required_bytes = 8589934592,  -- 8GB
      context_length = 8192,
      
      -- Capabilities
      supports_determinism = true,
      supports_streaming = true,
      
      -- Performance estimates
      estimated_tokens_per_sec = {
        ["NVIDIA RTX 4090"] = 90,
        ["NVIDIA A100"] = 120,
        default = 50
      },
      
      -- File metadata
      file_size_bytes = 4831838208,  -- 4.5GB
      file_hash = "sha256:...",
    },
    -- ... all models in catalog
  },
  
  -- Aliases
  aliases = {
    ["llama-8b"] = "hf:meta-llama/Llama-3.1-8B@main::file=model.gguf",
    ["llama-13b"] = "hf:meta-llama/Llama-3.1-13B@main::file=model.gguf",
  }
}
```

**Helper methods**:
```lua
-- Resolve alias to canonical model_ref
local model_ref = ctx.resolve_model("llama-8b")

-- Get model metadata
local model = ctx.get_model("hf:meta-llama/Llama-3.1-8B@main")

-- Check if model exists
if ctx.model_exists("llama-8b") then
  -- ...
end

-- Find models by criteria
local small_models = ctx.filter_models(function(model)
  return model.vram_required_bytes < 16 * 1024^3  -- < 16GB
end)
```

---

### 3.4 Tenant Quotas (`ctx.tenants`)

**Platform mode only** - tenant quota and usage information.

```lua
ctx.tenants = {
  ["tenant-1"] = {
    id = "tenant-1",
    name = "Acme Corp",
    
    -- Quotas
    quota_vram_bytes = 51539607552,  -- 48GB
    quota_max_concurrent_jobs = 10,
    quota_tokens_daily = 1000000,
    
    -- Current usage
    vram_used_bytes = 34359738368,  -- 32GB
    concurrent_jobs = 5,
    tokens_used_today = 450000,
    
    -- Quota reset
    quota_reset_at = 1696435200000,  -- midnight UTC
    
    -- Status
    quota_exceeded = false,
  },
  -- ... more tenants
}
```

**Helper methods**:
```lua
-- Check if tenant can run job
local can_run, reason = ctx.check_tenant_quota("tenant-1", job)
-- returns: true, nil  OR  false, "quota_exceeded"

-- Get tenant usage
local usage = ctx.get_tenant_usage("tenant-1")
```

---

### 3.5 System Metrics (`ctx.metrics`)

Real-time system metrics for informed decisions.

```lua
ctx.metrics = {
  -- Queue metrics
  queue_depth = 15,
  queue_depth_interactive = 10,
  queue_depth_batch = 5,
  avg_wait_time_ms = 3500,
  
  -- Worker metrics
  total_workers = 8,
  ready_workers = 3,
  busy_workers = 5,
  failed_workers = 0,
  
  -- Pool metrics
  total_pools = 2,
  healthy_pools = 2,
  degraded_pools = 0,
  offline_pools = 0,
  
  -- GPU metrics
  total_gpus = 4,
  total_vram_bytes = 103079215104,  -- 96GB
  allocated_vram_bytes = 68719476736,  -- 64GB
  free_vram_bytes = 34359738368,  -- 32GB
  avg_gpu_utilization = 72.5,
  
  -- Throughput metrics
  jobs_completed_last_minute = 42,
  avg_tokens_per_sec = 85.3,
  
  -- Timestamp
  timestamp_ms = 1696348805000,
}
```

---

### 3.6 Configuration (`ctx.config`)

Current orchestrator configuration (read-only).

```lua
ctx.config = {
  mode = "agentic",  -- "agentic" or "platform"
  deployment_mode = "lab",  -- "home", "lab", "platform"
  
  queue = {
    capacity = -1,  -- unbounded
    policy = "queue",
  },
  
  timeout = {
    default_ms = 300000,
    max_ms = 1800000,
  },
  
  retry = {
    enabled = true,
    max_attempts = 5,
    initial_delay_ms = 1000,
    multiplier = 2.0,
    max_delay_ms = 60000,
  },
  
  eviction = {
    model_cache_policy = "lru",
    worker_policy = "lru",
    vram_threshold = 0.9,
  }
}
```

---

## 4. Built-in Functions

### 4.1 Worker Selection Functions

```lua
-- Find least-loaded worker for a model
-- @param model_ref: string - Model reference
-- @param pool_id: string|nil - Optional pool filter
-- @return Worker|nil
worker = ctx.find_least_loaded_worker(model_ref, pool_id)

-- Find worker with most free VRAM
-- @param model_ref: string - Model reference
-- @return Worker|nil
worker = ctx.find_most_vram_free_worker(model_ref)

-- Find any ready worker for model
-- @param model_ref: string - Model reference
-- @return Worker|nil
worker = ctx.find_ready_worker(model_ref)

-- Round-robin worker selection
-- @param model_ref: string - Model reference
-- @return Worker|nil
worker = ctx.find_worker_round_robin(model_ref)

-- Custom worker filter
-- @param predicate: function(Worker) -> boolean
-- @return Worker[]
workers = ctx.filter_workers(function(worker)
  return worker.status == "ready" and worker.gpu_ids[1] == 0
end)
```

---

### 4.2 Pool State Functions

```lua
-- Fetch fresh pool state from pool managers
-- Forces immediate heartbeat query (normally cached)
-- @return boolean - success
success = ctx.refresh_pool_state()

-- Get pool by ID
-- @param pool_id: string
-- @return Pool|nil
pool = ctx.get_pool("pool-1")

-- Find pools with available capacity
-- @param vram_needed: number - Bytes needed
-- @return Pool[]
pools = ctx.find_pools_with_capacity(8 * 1024^3)  -- 8GB

-- Check if pool is healthy
-- @param pool_id: string
-- @return boolean
is_healthy = ctx.is_pool_healthy("pool-1")

-- Get pool utilization
-- @param pool_id: string
-- @return number - 0.0 to 1.0
utilization = ctx.get_pool_utilization("pool-1")
```

---

### 4.3 GPU Query Functions

```lua
-- Find GPUs with available VRAM
-- @param vram_needed: number - Bytes needed
-- @return GPU[]
gpus = ctx.find_gpus_with_vram(8 * 1024^3)

-- Get GPU by pool and device ID
-- @param pool_id: string
-- @param gpu_id: number
-- @return GPU|nil
gpu = ctx.get_gpu("pool-1", 0)

-- Check GPU compatibility with model
-- @param gpu: GPU
-- @param model_ref: string
-- @return boolean, string|nil - compatible, reason
compatible, reason = ctx.check_gpu_compatibility(gpu, model_ref)

-- Estimate model performance on GPU
-- @param model_ref: string
-- @param gpu: GPU
-- @return number - estimated tokens/sec
tokens_per_sec = ctx.estimate_performance(model_ref, gpu)
```

---

### 4.4 Model Catalog Functions

```lua
-- Resolve model alias
-- @param model: string - Alias or full ref
-- @return string - Canonical model_ref
model_ref = ctx.resolve_model("llama-8b")

-- Get model metadata
-- @param model_ref: string
-- @return Model|nil
model = ctx.get_model(model_ref)

-- Check if model exists in catalog
-- @param model_ref: string
-- @return boolean
exists = ctx.model_exists(model_ref)

-- Find models by family
-- @param family: string - e.g., "llama", "mistral"
-- @return Model[]
models = ctx.find_models_by_family("llama")

-- Get VRAM requirement for model
-- @param model_ref: string
-- @return number - bytes
vram_needed = ctx.get_model_vram(model_ref)
```

---

### 4.5 Job Analysis Functions

```lua
-- Calculate job wait time
-- @param job: Job
-- @return number - milliseconds
wait_time = ctx.calculate_wait_time(job)

-- Estimate job duration
-- @param job: Job
-- @param worker: Worker|nil - Optional worker for accurate estimate
-- @return number - milliseconds
duration = ctx.estimate_job_duration(job, worker)

-- Check if job can fit on worker
-- @param job: Job
-- @param worker: Worker
-- @return boolean, string|nil - fits, reason
fits, reason = ctx.can_job_fit(job, worker)

-- Get job priority score (higher = more urgent)
-- @param job: Job
-- @return number
score = ctx.get_job_priority_score(job)
```

---

### 4.6 Quota Functions (Platform Mode)

```lua
-- Check tenant quota
-- @param tenant_id: string
-- @param job: Job
-- @return boolean, string|nil - allowed, reason
allowed, reason = ctx.check_tenant_quota(tenant_id, job)

-- Get tenant usage
-- @param tenant_id: string
-- @return TenantUsage
usage = ctx.get_tenant_usage(tenant_id)

-- Estimate quota impact
-- @param tenant_id: string
-- @param job: Job
-- @return number - VRAM bytes that would be used
vram_impact = ctx.estimate_quota_impact(tenant_id, job)
```

---

### 4.7 Eviction Functions

```lua
-- Find worker to evict (LRU)
-- @param pool_id: string|nil - Optional pool filter
-- @return Worker|nil
worker = ctx.find_worker_to_evict_lru(pool_id)

-- Find worker to evict (LFU - least frequently used)
-- @return Worker|nil
worker = ctx.find_worker_to_evict_lfu()

-- Check if eviction needed
-- @param pool_id: string
-- @return boolean, number - needed, vram_threshold_exceeded
needed, exceeded = ctx.should_evict(pool_id)

-- Find cached model to evict from RAM
-- @param pool_id: string
-- @return string|nil - model_ref
model_ref = ctx.find_model_to_evict(pool_id)
```

---

### 4.8 Utility Functions

```lua
-- Logging
ctx.log_info("Scheduling job-123 to worker-1")
ctx.log_warn("Pool pool-1 utilization high: 95%")
ctx.log_error("Failed to find worker for model: llama-70b")
ctx.log_debug("Queue depth: " .. ctx.metrics.queue_depth)

-- Metrics emission (custom metrics)
ctx.emit_metric("custom_scheduler_decision_time_ms", 15.3)
ctx.emit_metric("custom_jobs_skipped", 5)

-- Time utilities
local now_ms = ctx.now()  -- Unix timestamp milliseconds
local elapsed = ctx.elapsed_since(job.queued_at)  -- milliseconds

-- Math utilities
local avg = ctx.average({10, 20, 30})  -- 20
local max_val = ctx.max({10, 20, 30})  -- 30
local min_val = ctx.min({10, 20, 30})  -- 10

-- String utilities
local contains = ctx.string_contains("llama-8b", "llama")  -- true
local starts = ctx.string_starts_with("hf:meta-llama", "hf:")  -- true
```

---

### 4.9 Platform Scheduler Access

```lua
-- Call platform scheduler (fallback)
-- @param ctx: SchedulerContext
-- @return SchedulerDecision
decision = ctx.platform_schedule(ctx)

-- Get platform scheduler source code (if allowed)
-- @return string|nil - Lua source code
source = ctx.get_platform_scheduler_source()
```

---

## 5. Return Value (SchedulerDecision)

The `schedule()` function MUST return a table with action and parameters:

### 5.1 Dispatch Action

```lua
return {
  action = "dispatch",
  job_id = "job-abc123",
  worker_id = "worker-1",
}
```

### 5.2 Wait Action

```lua
return {
  action = "wait",
  reason = "no_available_workers",  -- optional
  retry_after_ms = 1000,  -- optional, hint for next schedule cycle
}
```

### 5.3 Start Worker Action

```lua
return {
  action = "start_worker",
  pool_id = "pool-1",
  gpu_id = 0,
  model_ref = "hf:meta-llama/Llama-3.1-8B@main",
  reason = "preload_for_queued_jobs",  -- optional
}
```

### 5.4 Evict Worker Action

```lua
return {
  action = "evict_worker",
  worker_id = "worker-3",
  reason = "lru_eviction",  -- optional
}
```

### 5.5 Multiple Actions (Batch)

```lua
return {
  action = "batch",
  actions = {
    { action = "dispatch", job_id = "job-1", worker_id = "worker-1" },
    { action = "dispatch", job_id = "job-2", worker_id = "worker-2" },
    { action = "start_worker", pool_id = "pool-1", gpu_id = 1, model_ref = "llama-13b" },
  }
}
```

---

## 6. Sandbox Restrictions

### 6.1 Disabled Lua Features

The following Lua features are **DISABLED** for security:

```lua
-- File I/O
io.*           -- DISABLED
os.execute     -- DISABLED
os.remove      -- DISABLED
os.rename      -- DISABLED

-- Network
socket.*       -- DISABLED (if LuaSocket present)

-- Process control
os.exit        -- DISABLED

-- Dangerous functions
load           -- DISABLED (code injection)
loadfile       -- DISABLED
dofile         -- DISABLED
require        -- DISABLED (use preloaded modules only)
```

### 6.2 Execution Limits

```lua
-- Timeout: 50ms per schedule() call
-- Memory limit: 64MB heap
-- No infinite loops (detected and terminated)
```

### 6.3 Allowed Standard Library

```lua
-- Math
math.*         -- ALLOWED

-- String
string.*       -- ALLOWED

-- Table
table.*        -- ALLOWED

-- Safe OS functions
os.time        -- ALLOWED
os.date        -- ALLOWED
os.clock       -- ALLOWED

-- Debugging (limited)
debug.traceback  -- ALLOWED (for error reporting)
```

---

## 7. Example Schedulers

### 7.1 Simple Priority Scheduler

```lua
function schedule(ctx)
  -- Process interactive jobs first
  for _, job in ipairs(ctx.queue) do
    if job.priority == "interactive" then
      local worker = ctx.find_least_loaded_worker(job.model_ref)
      if worker then
        ctx.log_info("Dispatching interactive job: " .. job.id)
        return {
          action = "dispatch",
          job_id = job.id,
          worker_id = worker.id
        }
      end
    end
  end
  
  -- Then batch jobs
  for _, job in ipairs(ctx.queue) do
    if job.priority == "batch" then
      local worker = ctx.find_most_vram_free_worker(job.model_ref)
      if worker then
        return {
          action = "dispatch",
          job_id = job.id,
          worker_id = worker.id
        }
      end
    end
  end
  
  return { action = "wait" }
end
```

### 7.2 GPU-Specific Scheduler

```lua
function schedule(ctx)
  -- User has 2x RTX 4090 (GPU 0,1) and 1x A100 (GPU 2)
  -- Strategy: Small models on 4090s, large models on A100
  
  local job = ctx.queue[1]
  if not job then
    return { action = "wait" }
  end
  
  local model = ctx.get_model(job.model_ref)
  local vram_needed = model.vram_required_bytes
  
  -- Large model (>16GB) → A100 only
  if vram_needed > 16 * 1024^3 then
    local a100_worker = ctx.filter_workers(function(w)
      return w.status == "ready" and w.gpu_ids[1] == 2
    end)[1]
    
    if a100_worker then
      return { action = "dispatch", job_id = job.id, worker_id = a100_worker.id }
    else
      -- Start worker on A100
      return {
        action = "start_worker",
        pool_id = "pool-1",
        gpu_id = 2,
        model_ref = job.model_ref
      }
    end
  end
  
  -- Small model → 4090s (round-robin between GPU 0 and 1)
  local rtx_workers = ctx.filter_workers(function(w)
    return w.status == "ready" and (w.gpu_ids[1] == 0 or w.gpu_ids[1] == 1)
  end)
  
  if #rtx_workers > 0 then
    local worker = rtx_workers[1]
    return { action = "dispatch", job_id = job.id, worker_id = worker.id }
  end
  
  return { action = "wait" }
end
```

### 7.3 Determinism-First Scheduler

```lua
function schedule(ctx)
  -- Prioritize jobs with seeds (deterministic inference)
  local seeded_jobs = ctx.filter_jobs(function(job)
    return job.seed ~= nil
  end)
  
  for _, job in ipairs(seeded_jobs) do
    local model = ctx.get_model(job.model_ref)
    
    -- Only dispatch if model supports determinism
    if model.supports_determinism then
      local worker = ctx.find_ready_worker(job.model_ref)
      if worker then
        ctx.log_info("Dispatching deterministic job: " .. job.id .. " (seed: " .. job.seed .. ")")
        return { action = "dispatch", job_id = job.id, worker_id = worker.id }
      end
    end
  end
  
  -- Fallback to platform scheduler for non-seeded jobs
  return ctx.platform_schedule(ctx)
end
```

### 7.4 VRAM-Aware Eviction Scheduler

```lua
function schedule(ctx)
  -- Check if any pool is over VRAM threshold
  for _, pool in ipairs(ctx.pools) do
    local utilization = ctx.get_pool_utilization(pool.id)
    
    if utilization > 0.9 then
      -- Evict LRU worker to free VRAM
      local worker_to_evict = ctx.find_worker_to_evict_lru(pool.id)
      if worker_to_evict then
        ctx.log_warn("Pool " .. pool.id .. " over threshold, evicting worker: " .. worker_to_evict.id)
        return {
          action = "evict_worker",
          worker_id = worker_to_evict.id,
          reason = "vram_threshold_exceeded"
        }
      end
    end
  end
  
  -- Normal scheduling
  local job = ctx.queue[1]
  if job then
    local worker = ctx.find_least_loaded_worker(job.model_ref)
    if worker then
      return { action = "dispatch", job_id = job.id, worker_id = worker.id }
    end
  end
  
  return { action = "wait" }
end
```

---

## 8. Error Handling

### 8.1 Scheduler Errors

If the scheduler throws an error, orchestratord will:

1. Log the error with full traceback
2. Emit metric: `scheduler_error_total`
3. Fallback to platform scheduler
4. Continue operation (non-fatal)

```lua
function schedule(ctx)
  -- This will be caught and logged
  error("Something went wrong!")
end
```

### 8.2 Timeout Handling

If scheduler exceeds 50ms execution time:

1. Scheduler is terminated
2. Warning logged: "Scheduler timeout exceeded"
3. Fallback to platform scheduler
4. Metric emitted: `scheduler_timeout_total`

---

## 9. Performance Considerations

### 9.1 Caching

```lua
-- Pool state is cached for 500ms (heartbeat interval)
-- To force refresh:
ctx.refresh_pool_state()

-- Model catalog is preloaded at startup (no runtime cost)
local model = ctx.get_model("llama-8b")  -- O(1) lookup
```

### 9.2 Optimization Tips

```lua
-- ✅ GOOD: Early return
function schedule(ctx)
  if #ctx.queue == 0 then
    return { action = "wait" }
  end
  -- ... rest of logic
end

-- ❌ BAD: Iterating all workers every time
function schedule(ctx)
  for _, pool in ipairs(ctx.pools) do
    for _, worker in ipairs(pool.workers) do
      -- expensive operation
    end
  end
end

-- ✅ GOOD: Use helper functions (optimized in Rust)
local worker = ctx.find_least_loaded_worker(model_ref)
```

---

## 10. Testing & Debugging

### 10.1 Dry-Run Mode

```bash
# Test scheduler without executing actions
orchestratord scheduler test --file my-scheduler.lua --dry-run
```

### 10.2 Trace Mode

```lua
-- Enable detailed logging
ctx.log_debug("Queue depth: " .. #ctx.queue)
ctx.log_debug("Available workers: " .. #ctx.get_all_workers())
```

### 10.3 Simulation

```bash
# Simulate with sample data
orchestratord scheduler simulate \
  --file my-scheduler.lua \
  --queue-depth 10 \
  --pools 2 \
  --workers-per-pool 4
```

---

## 11. Implementation Notes

### 11.1 Rust Integration

```rust
// bin/orchestratord-crates/scheduling/src/lua_runtime.rs

pub struct LuaSchedulerRuntime {
    lua: Lua,
    scheduler_code: String,
}

impl LuaSchedulerRuntime {
    pub fn new(scheduler_code: String) -> Result<Self> {
        let lua = Lua::new();
        
        // Preload context API
        Self::register_context_api(&lua)?;
        
        // Load scheduler code
        lua.load(&scheduler_code).exec()?;
        
        Ok(Self { lua, scheduler_code })
    }
    
    pub fn schedule(&self, context: SchedulerContext) -> Result<SchedulerDecision> {
        // Convert Rust context to Lua table
        let ctx_table = self.context_to_lua(&context)?;
        
        // Call schedule() function
        let schedule_fn: Function = self.lua.globals().get("schedule")?;
        let result: Table = schedule_fn.call(ctx_table)?;
        
        // Convert Lua result to Rust
        self.lua_to_decision(result)
    }
}
```

---

## 12. Future Enhancements

### 12.1 Machine Learning Integration

```lua
-- Predict job duration using ML model (future)
local predicted_duration = ctx.predict_job_duration(job)

-- Predict GPU utilization
local predicted_util = ctx.predict_gpu_utilization(pool_id, 60000)  -- next 60s
```

### 12.2 Historical Data

```lua
-- Access historical metrics (future)
local history = ctx.get_historical_metrics({
  metric = "queue_depth",
  duration_ms = 3600000,  -- last hour
  interval_ms = 60000     -- 1-minute buckets
})
```

### 12.3 A/B Testing

```lua
-- Run multiple schedulers and compare (future)
ctx.enable_ab_test({
  schedulers = {"platform", "custom"},
  split = 0.5  -- 50/50 split
})
```

---

## 13. Summary

The Lua scheduler runtime provides:

✅ **Complete state access**: Queue, pools, workers, GPUs, models, tenants  
✅ **Rich built-in functions**: 40+ helper functions for common operations  
✅ **Compile-time catalog**: Model metadata preloaded  
✅ **Real-time pool state**: Fresh data from heartbeats  
✅ **Zero dependencies**: Everything built-in  
✅ **Sandboxed execution**: Secure, timeout-protected  
✅ **Performance**: <50ms execution, cached data  
✅ **Fallback safety**: Platform scheduler always available  

Users can write powerful, custom scheduling logic with full system visibility while maintaining security and performance guarantees.

---

**Related Specs**:
- `00_programmable_scheduler.md` - Overall scheduler design
- `02_yaml_compiler.md` - YAML → Lua compilation
- `platform-scheduler.lua` - Reference implementation

**Implementation**:
- Rust runtime: `bin/orchestratord-crates/scheduling/src/lua_runtime.rs`
- Context API: `bin/orchestratord-crates/scheduling/src/context_api.rs`
- Built-in functions: `bin/orchestratord-crates/scheduling/src/builtins.rs`
