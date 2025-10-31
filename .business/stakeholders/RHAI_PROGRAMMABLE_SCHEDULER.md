# 🎭 Rhai Programmable Scheduler: The Smart Brain

**Pronunciation:** rbee (pronounced "are-bee")  
**Date:** 2025-10-31 (Updated)  
**Status:** M2 Feature (Planned) | M0/M1: SimpleScheduler (Active)  
**Spec:** `bin/.specs/00_llama-orch.md` [SYS-6.1.5]

---

## What is the Rhai Programmable Scheduler?

**Rhai** is an embedded Rust scripting language that will power rbee's intelligent scheduling decisions in M2+.

**Think of it as:**
- The "brain" of queen-rbee
- A policy execution engine
- User-programmable orchestration logic

---

## Current Implementation (M0/M1): SimpleScheduler

**Status:** ✅ ACTIVE (TEAM-275, TEAM-374)

**Location:** `bin/15_queen_rbee_crates/scheduler/`

### Architecture

```rust
// Current scheduler trait
#[async_trait]
pub trait JobScheduler: Send + Sync {
    async fn schedule(&self, request: JobRequest) 
        -> Result<ScheduleResult, SchedulerError>;
}

// M0/M1 implementation
pub struct SimpleScheduler {
    worker_registry: Arc<TelemetryRegistry>, // TEAM-374
}
```

### How It Works (Real-Time Telemetry)

**Data Flow:**
```
Worker Process (llama-cli)
    ↓ cgroup stats (CPU, RAM, uptime)
    ↓ nvidia-smi (GPU util, VRAM)
    ↓ /proc/pid/cmdline (model name)
Hive Monitor (rbee_hive_monitor::collect_all_workers)
    ↓ ProcessStats every 1s
Hive Heartbeat (POST /v1/hive-heartbeat)
    ↓ HiveHeartbeatEvent
Queen TelemetryRegistry (TEAM-374)
    ↓ stores workers by hive_id
SimpleScheduler.schedule()
    ↓ finds best worker
Worker Selection (first idle worker with model)
```

### ProcessStats Structure (Real-Time Telemetry)

```rust
pub struct ProcessStats {
    // Process info
    pub pid: u32,
    pub group: String,        // e.g., "llm"
    pub instance: String,     // e.g., "8080" (port)
    
    // CPU & Memory
    pub cpu_pct: f64,        // CPU usage %
    pub rss_mb: u64,         // RAM in MB
    pub uptime_s: u64,       // Uptime in seconds
    
    // GPU telemetry (TEAM-360)
    pub gpu_util_pct: f64,   // 0.0 = idle, >0 = busy
    pub vram_mb: u64,        // VRAM used
    pub total_vram_mb: u64,  // Total VRAM (TEAM-364)
    
    // Model detection (TEAM-360)
    pub model: Option<String>, // From --model arg
}
```

### TelemetryRegistry API (TEAM-374)

**Available NOW for scheduling:**

```rust
// Worker queries (used by SimpleScheduler)
registry.find_best_worker_for_model(model)  // First idle worker with model
registry.find_idle_workers()                 // gpu_util_pct == 0.0
registry.find_workers_with_model(model)      // All workers with model
registry.find_workers_with_capacity(vram_mb) // VRAM capacity check
registry.list_online_workers()               // All workers

// Hive queries
registry.list_online_hives()                 // Recent heartbeats
registry.list_available_hives()              // Online + Ready status
registry.get_hive(hive_id)                   // Single hive info
```

### Current Scheduling Algorithm

**SimpleScheduler (M0/M1):**

1. Find workers serving requested model
2. Filter by idle status (`gpu_util_pct == 0.0`)
3. Return **first match** (no load balancing)

**Code:**
```rust
let worker = self.worker_registry
    .find_best_worker_for_model(model)
    .ok_or(SchedulerError::NoWorkersAvailable)?;
```

**Limitations:**
- ❌ No load balancing
- ❌ No cost optimization
- ❌ No latency optimization
- ❌ No custom policies
- ❌ No multi-modal routing

**These will be solved by RhaiScheduler in M2+**

---

## Future Implementation (M2+): RhaiScheduler

**Status:** 📋 PLANNED

**The Rhai scheduler will have access to the SAME TelemetryRegistry data, but with programmable logic.**

---

## Two Modes: Platform vs Home/Lab

### Platform Mode (Multi-Tenant Marketplace)

**Purpose:** Secure, fair, multi-tenant GPU marketplace

**Scheduler:** `platform-scheduler.rhai` (built-in, immutable)

**Characteristics:**
- ✅ **Immutable** - Cannot be modified by users
- ✅ **Multi-tenant fairness** - Fair resource allocation
- ✅ **SLA compliance** - Guarantees service levels
- ✅ **Security-first** - Sandboxed execution
- ✅ **Quota enforcement** - Per-tenant limits
- ✅ **Capacity management** - Rejects with 429 when full

**Use Case:** Commercial GPU marketplace (api.yourplatform.com)

**Example Logic:**
```rhai
// Platform scheduler (immutable)
fn schedule_job(job, pools, workers) {
    // 1. Check tenant quota
    if tenant_over_quota(job.tenant_id) {
        return reject("quota_exceeded");
    }
    
    // 2. Priority-based scheduling
    if job.priority == "interactive" {
        // Route to least-loaded worker
        return workers.least_loaded();
    } else {
        // Route to cheapest available
        return workers.min_by(|w| w.cost_per_token);
    }
}
```

### Home/Lab Mode (Self-Hosted)

**Purpose:** Custom orchestration for personal/team use

**Scheduler:** User-written Rhai scripts or YAML configs

**Characteristics:**
- ✅ **Customizable** - Write your own logic
- ✅ **No recompilation** - Update scripts live
- ✅ **Unbounded queues** - No artificial limits
- ✅ **Custom policies** - Your rules, your way
- ✅ **YAML support** - Compiles to Rhai internally
- ✅ **Web UI builder** - Visual policy editor

**Use Case:** Homelab, research lab, small team

**Example Logic:**
```rhai
// Custom home scheduler
fn schedule_job(job, pools, workers) {
    // Route large models to multi-GPU setups
    if job.model.contains("70b") {
        return workers.filter(|w| w.gpus > 1).first();
    }
    
    // Route image generation to CUDA
    if job.type == "image-gen" {
        return workers
            .filter(|w| w.capability == "image-gen" && w.backend == "cuda")
            .least_loaded();
    }
    
    // Everything else to cheapest
    return workers.min_by(|w| w.cost_per_token);
}
```

---

## The Rhai Scheduler API (M2+)

### Complete System State Access

**Available to Rhai scripts (will use TelemetryRegistry):**
- `registry` - TelemetryRegistry with real-time worker telemetry
- `workers` - All active workers (ProcessStats)
- `hives` - All online hives
- `job` - Current job request
- `tenants` - Tenant quotas (platform mode)

### Built-in Helper Functions (Planned M2+)

**These will wrap the existing TelemetryRegistry API:**

**Worker Selection (wraps TelemetryRegistry):**
- `workers.idle()` - Wraps `registry.find_idle_workers()`
- `workers.with_model(model)` - Wraps `registry.find_workers_with_model()`
- `workers.with_capacity(vram_mb)` - Wraps `registry.find_workers_with_capacity()`
- `workers.least_loaded()` - Sort by `cpu_pct` ascending
- `workers.most_vram_free()` - Sort by `(total_vram_mb - vram_mb)` descending
- `workers.round_robin()` - Distribute evenly
- `workers.filter(predicate)` - Filter by condition
- `workers.min_by(key)` - Pick minimum by key
- `workers.max_by(key)` - Pick maximum by key

**ProcessStats Fields (available in Rhai):**
- `worker.pid` - Process ID
- `worker.group` - Service group (e.g., "llm")
- `worker.instance` - Port number
- `worker.cpu_pct` - CPU usage %
- `worker.rss_mb` - RAM in MB
- `worker.gpu_util_pct` - GPU utilization (0.0 = idle)
- `worker.vram_mb` - VRAM used
- `worker.total_vram_mb` - Total VRAM available
- `worker.model` - Model name (from --model arg)
- `worker.uptime_s` - Uptime in seconds

**Hive Queries (wraps TelemetryRegistry):**
- `hives.online()` - Wraps `registry.list_online_hives()`
- `hives.available()` - Wraps `registry.list_available_hives()`
- `hives.get(hive_id)` - Wraps `registry.get_hive()`

**Quota Checks (Platform Mode - M3+):**
- `tenant_over_quota(tenant_id)` - Check quota exceeded
- `tenant_remaining_quota(tenant_id)` - Remaining quota
- `tenant_usage(tenant_id)` - Current usage

**Model Queries (Future):**
- `model_size_bytes(model_ref)` - Model file size
- `model_vram_required(model_ref)` - Estimated VRAM needed
- `model_exists(model_ref)` - Check if model available

---

## Real-World Examples

### Example 1: Cost Optimizer (Home Mode)

```rhai
// Route to cheapest provider that meets deadline
fn schedule_job(job, pools, workers) {
    let deadline_ok = workers.filter(|w| 
        w.estimated_time < job.deadline
    );
    
    if deadline_ok.is_empty() {
        return reject("no_workers_meet_deadline");
    }
    
    return deadline_ok.min_by(|w| w.cost_per_token);
}
```

**Value:** Saves 20-30% on costs by always choosing cheapest option

### Example 2: Latency Optimizer (Home Mode)

```rhai
// Route to geographically closest worker
fn schedule_job(job, pools, workers) {
    let user_lat = job.user.geo.latitude;
    let user_lon = job.user.geo.longitude;
    
    return workers.min_by(|w| {
        distance(user_lat, user_lon, w.geo.latitude, w.geo.longitude)
    });
}
```

**Value:** 30-50% lower latency by minimizing network distance

### Example 3: Multi-Modal Smart Router (Home Mode)

```rhai
// Route different AI tasks using real telemetry
fn schedule_job(job, registry, workers) {
    // Route large models to workers with high VRAM
    if job.model.contains("70b") {
        let high_vram = workers.filter(|w| w.total_vram_mb > 40000);
        if !high_vram.is_empty() {
            return high_vram.idle().first();
        }
    }
    
    // Route to idle workers with the model loaded
    let with_model = workers.with_model(job.model).idle();
    if !with_model.is_empty() {
        return with_model.least_loaded();
    }
    
    // Fallback: any idle worker with capacity
    let with_capacity = workers.with_capacity(8192).idle();
    if !with_capacity.is_empty() {
        return with_capacity.first();
    }
    
    return reject("no_available_workers");
}
```

**Value:** Uses real GPU telemetry for intelligent routing

### Example 4: EU-Only Compliance Router (Home Mode)

```rhai
// Route only to EU providers for GDPR compliance
fn schedule_job(job, pools, workers) {
    let eu_workers = workers.filter(|w| 
        w.geo.region == "EU"
    );
    
    if eu_workers.is_empty() {
        return reject("no_eu_workers_available");
    }
    
    return eu_workers.least_loaded();
}
```

**Value:** Automatic GDPR compliance by geo-filtering

---

## YAML Support (Declarative Mode)

**For non-programmers, YAML compiles to Rhai:**

```yaml
# scheduler.yaml
rules:
  - name: "Large models to multi-GPU"
    condition:
      model_contains: "70b"
    action:
      select: "workers.filter(gpus > 1).first()"
  
  - name: "Image gen to CUDA"
    condition:
      task_type: "image-gen"
    action:
      select: "workers.filter(backend == 'cuda').least_loaded()"
  
  - name: "Default to cheapest"
    condition: true
    action:
      select: "workers.min_by(cost_per_token)"
```

**Compiles to Rhai internally, same performance**

---

## Web UI Mode (Visual Policy Builder)

**For visual users:**

1. **Drag-and-drop policy builder**
   - Add conditions (if model contains "70b")
   - Add actions (route to multi-GPU)
   - Visual flow diagram

2. **Live preview**
   - Test with sample jobs
   - See routing decisions
   - Verify correctness

3. **Export to Rhai or YAML**
   - Save as script file
   - Deploy to queen-rbee
   - Update without restart

---

## Security & Sandboxing

### Sandboxed Execution

**Safety guarantees:**
- ✅ **50ms timeout** - Scripts can't hang
- ✅ **Memory limits** - Can't exhaust RAM
- ✅ **No unsafe operations** - No file I/O, no network
- ✅ **Type safety** - Rhai is type-safe
- ✅ **No FFI** - Can't call external code

### Platform Mode Security

**Immutable scheduler:**
- ❌ Users cannot modify platform scheduler
- ❌ Users cannot inject malicious code
- ❌ Users cannot bypass quotas
- ✅ Platform operator controls all logic
- ✅ Audited and tested scheduler

---

## Why Rhai (Not Python/JavaScript/Lua)?

| Feature | Rhai | Python | JavaScript | Lua |
|---------|------|--------|------------|-----|
| Rust integration | ✅ Native | ❌ FFI | ❌ FFI | ❌ FFI |
| Type safety | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Sandboxed | ✅ Yes | ❌ No | ⚠️ Partial | ⚠️ Partial |
| Performance | ✅ Fast | ❌ Slow | ⚠️ Medium | ✅ Fast |
| Zero dependencies | ✅ Yes | ❌ No | ❌ No | ⚠️ Partial |
| Embedded-first | ✅ Yes | ❌ No | ❌ No | ✅ Yes |

**Rhai is the ONLY choice for embedded Rust scripting.**

---

## Implementation Timeline

**M0 (Oct 2025):** ✅ **SimpleScheduler ACTIVE** (TEAM-275)
- First-available worker selection
- TelemetryRegistry integration (TEAM-374)
- Real-time worker telemetry (GPU, CPU, RAM, model)
- ProcessStats from cgroup + nvidia-smi
- Idle worker detection (gpu_util_pct == 0.0)

**M1 (Current):** ✅ **Production-ready SimpleScheduler**
- Worker lifecycle management
- Hive discovery and heartbeats
- SSE-based telemetry streaming
- Automatic stale worker cleanup

**M2 (Q2 2026):** 📋 **Rhai scheduler engine**
- Rhai script execution
- Access to TelemetryRegistry via Rhai API
- Custom scheduling logic
- YAML config support
- Hot-reload scripts without restart

**M3 (Q3 2026):** 📋 **Platform mode**
- Immutable platform scheduler
- Multi-tenant fairness
- Quota enforcement
- SLA compliance

**M4 (Q4 2026):** 📋 **Web UI policy builder**
- Visual policy editor
- Drag-and-drop conditions
- Live preview with sample jobs
- Export to Rhai/YAML

---

## Competitive Advantage

### vs. Competitors

**Runpod/Vast.ai:**
- ❌ No routing customization
- ❌ Fixed algorithms
- ❌ No user control

**Together.ai/Replicate:**
- ❌ Vendor lock-in
- ❌ Black box routing
- ❌ No customization

**AWS Bedrock:**
- ❌ Vendor lock-in
- ❌ No scripting
- ❌ Opaque routing

**rbee:**
- ✅ User-scriptable (Home/Lab mode)
- ✅ Transparent (open source)
- ✅ Customizable (Rhai + YAML + Web UI)
- ✅ Platform mode for marketplace
- ✅ Best of both worlds

---

## The Vision

**M0/M1 (Today):** ✅ SimpleScheduler with real-time telemetry  
**M2 (Q2 2026):** Programmable Rhai scheduler  
**M3 (Q3 2026):** Platform mode for marketplace  
**M4 (Q4 2026):** Visual policy builder  
**Future:** Community-shared scheduler templates

**The result:** Maximum flexibility without sacrificing security. 🐝🎭

---

## Key Takeaways

1. **Current (M0/M1):** SimpleScheduler is ACTIVE with real-time GPU telemetry
2. **TelemetryRegistry:** Stores ProcessStats from all workers (CPU, GPU, RAM, model)
3. **Real-time data:** Hives send telemetry every 1s via heartbeats
4. **Idle detection:** Uses `gpu_util_pct == 0.0` to find available workers
5. **Future (M2+):** Rhai will wrap TelemetryRegistry with programmable logic
6. **Two modes:** Platform (immutable) vs Home/Lab (customizable)
7. **Sandboxed:** 50ms timeout, memory limits, type-safe
8. **Web UI (M4):** Visual policy builder for non-programmers

**The foundation is READY. Rhai will add programmability on top.** 🎭🐝

---

*Based on: bin/.specs/00_llama-orch.md [SYS-6.1.5] Programmable Scheduler*
